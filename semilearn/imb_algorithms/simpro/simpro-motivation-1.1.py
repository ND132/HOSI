# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import torch.nn as nn
from collections import Counter
import matplotlib.pyplot as plt
import torch.nn.functional as F

from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument


@IMB_ALGORITHMS.register('simpro')
class SimPro(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(SimPro, self).__init__(args, net_builder, tb_log, logger)
        
        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None

        lb_class_dist = [0 for _ in range(self.num_classes)]
        if args.noise_ratio > 0:
            for c in self.dataset_dict['train_lb'].noised_targets:
                lb_class_dist[c] += 1
        else:
            for c in self.dataset_dict['train_lb'].targets:
                lb_class_dist[c] += 1
        lb_class_dist = np.array(lb_class_dist)
        self.lb_class_dist = torch.from_numpy(lb_class_dist / lb_class_dist.sum())
        
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)
        
        self.scale_ratio = args.ulb_dest_len / args.lb_dest_len
        
        self.ema_u = args.ema_u
        
        self.tau = args.tau

        self.py_con = self.lb_class_dist.cuda(args.gpu)
        self.py_uni = torch.ones(self.num_classes).cuda(args.gpu) / self.num_classes
        self.py_rev = torch.flip(self.py_con, dims=[0])

        self.adapt_dis = self.py_con
        self.estimate_dis = self.py_uni

        self.adapt_dis_adjustment = torch.log(self.adapt_dis ** self.tau + 1e-12)
        self.estimate_dis_adjustment = torch.log(self.estimate_dis ** self.tau + 1e-12)


    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

            if self.epoch > 0 and self.epoch % 5 == 0:
                select_ulb_idx_to_label = {}
                select_ulb_idx_to_pseudo_label = {}

                for ulb_idx, ulb_pseudo_label, ulb_label in zip(self.select_ulb_idx, self.select_ulb_pseudo_label, self.select_ulb_label):
                    if ulb_idx.item() in select_ulb_idx_to_label:
                        select_ulb_idx_to_label[ulb_idx.item()].append(ulb_label.item())
                    else:
                        select_ulb_idx_to_label[ulb_idx.item()] = [ulb_label.item()]

                    if ulb_idx.item() in select_ulb_idx_to_pseudo_label:
                        select_ulb_idx_to_pseudo_label[ulb_idx.item()].append(ulb_pseudo_label.item())
                    else:
                        select_ulb_idx_to_pseudo_label[ulb_idx.item()] = [ulb_pseudo_label.item()]

                select_ulb_unique_idx = torch.unique(self.select_ulb_idx)

                select_ulb_unique_label = []
                select_ulb_unique_pseudo_label = []

                for ulb_unique_idx in select_ulb_unique_idx:
                    ulb_unique_label = select_ulb_idx_to_label[ulb_unique_idx.item()]
                    ulb_unique_pseudo_label = select_ulb_idx_to_pseudo_label[ulb_unique_idx.item()]
                    if len(ulb_unique_label) > 1:
                        most_common_label = Counter(ulb_unique_label).most_common(1)[0][0]
                        most_common_number = Counter(ulb_unique_label).most_common(1)[0][1]
                        if most_common_number > 1:
                            select_ulb_unique_label.append(torch.tensor([most_common_label]))
                        else:
                            select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))
                    else:
                        select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))

                    if len(ulb_unique_pseudo_label) > 1:
                        most_common_label = Counter(ulb_unique_pseudo_label).most_common(1)[0][0]
                        most_common_number = Counter(ulb_unique_pseudo_label).most_common(1)[0][1]
                        if most_common_number > 1:
                            select_ulb_unique_pseudo_label.append(torch.tensor([most_common_label]))
                        else:
                            select_ulb_unique_pseudo_label.append(torch.tensor([ulb_unique_label[0]]))
                    else:
                        select_ulb_unique_pseudo_label.append(torch.tensor([ulb_unique_label[0]]))

                select_ulb_unique_label = torch.cat(select_ulb_unique_label)
                select_ulb_unique_pseudo_label = torch.cat(select_ulb_unique_pseudo_label)

                self.select_ulb_idx = select_ulb_unique_idx
                self.select_ulb_label = select_ulb_unique_label
                self.select_ulb_pseudo_label = select_ulb_unique_pseudo_label
            
                if not os.path.exists(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch))):
                    os.makedirs(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch)))

                class_indices = np.arange(self.num_classes)

                bar_width = 0.3
                index = class_indices - bar_width / 2

                record_mask_true = torch.zeros(self.num_classes).cpu()
                record_mask_false = torch.zeros(self.num_classes).cpu()

                record_mask_true.index_add_(0, self.select_ulb_pseudo_label[self.select_ulb_pseudo_label==self.select_ulb_label], torch.ones_like(self.select_ulb_pseudo_label[self.select_ulb_pseudo_label==self.select_ulb_label], dtype=record_mask_true.dtype))
                record_mask_false.index_add_(0, self.select_ulb_pseudo_label[self.select_ulb_pseudo_label!=self.select_ulb_label], torch.ones_like(self.select_ulb_pseudo_label[self.select_ulb_pseudo_label!=self.select_ulb_label], dtype=record_mask_false.dtype))

                self.print_fn('ulb_dist:\n' + np.array_str(np.array(self.ulb_dist.cpu())))
                self.print_fn('record_mask_true:\n' + np.array_str(np.array(record_mask_true)))
                self.print_fn('record_mask_false:\n' + np.array_str(np.array(record_mask_false)))
                self.print_fn('record_mask:\n' + np.array_str(np.array(record_mask_true + record_mask_false)))

                fig = plt.figure(figsize=(8, 6), dpi=1000)

                ax = fig.add_subplot(111)

                bar0 = ax.bar(index, torch.log(self.ulb_dist).tolist(), width=bar_width, color='#ffffff', edgecolor='black', label='GT')
                bar1 = ax.bar(index + bar_width, torch.log(record_mask_true).tolist(), width=bar_width, color='#e37663', edgecolor='black', label='TP')
                for i in range(self.num_classes):
                    if i == 0:
                        bar2 = ax.bar(index[i] + bar_width, (record_mask_false * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], width=bar_width, bottom=(record_mask_true * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], color='#76a4bc', edgecolor='black', label='FP')
                    else:
                        bar2 = ax.bar(index[i] + bar_width, (record_mask_false * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], width=bar_width, bottom=(record_mask_true * torch.log(record_mask_false + record_mask_true) / (record_mask_false + record_mask_true)).tolist()[i], color='#76a4bc', edgecolor='black')

                ax.set_ylim(0, max(max(np.array(torch.log(self.ulb_dist).cpu())), max(np.array(torch.log(record_mask_true + record_mask_false)))) + 1)

                ax.set_xlabel('Class index', fontsize=18)
                ax.set_ylabel('Number of samples', fontsize=18)

                ax.set_xticks(class_indices)
                ax.set_xticklabels([f'{i}' for i in class_indices], fontsize=18, rotation=0)

                sample_indices = np.arange(0, max(max(np.array(torch.log(self.ulb_dist).cpu())), max(np.array(torch.log(record_mask_true + record_mask_false)))) + 1, 3)

                ax.set_yticks(sample_indices)
                ax.set_yticklabels([f'$e^{{{int(i)}}}$' for i in sample_indices], fontsize=18, rotation=0)

                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.5, handleheight=0.8, borderpad=0.2, columnspacing=0.8, fontsize=16, framealpha=0.2)

                plt.subplots_adjust(left=0.15, bottom=0.15)

                plt.savefig(os.path.join(self.args.save_dir, self.args.save_name, str(self.epoch), 'mask_true_false.pdf'))
                plt.clf()
                plt.close()
            
            self.select_ulb_idx = None
            self.select_ulb_label = None
            self.select_ulb_pseudo_label = None

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")
            
            self.adapt_dis_adjustment = torch.log(self.adapt_dis ** self.tau + 1e-12)
            self.estimate_dis_adjustment = torch.log(self.estimate_dis ** self.tau + 1e-12)
            
            self.count_labeled_dataset = torch.zeros(self.num_classes).cuda(self.args.gpu)
            self.dis_unlabeled_dataset = torch.zeros(self.num_classes).cuda(self.args.gpu)

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")


    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb
        num_lb = lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x = outputs['logits'][:num_lb]
                logits_u_w, logits_u_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb_w) 
                logits_x = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_u_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_u_w = outs_x_ulb_w['logits']

            pseudo_label = torch.softmax(logits_u_w.detach() + self.estimate_dis_adjustment, dim=-1)
            
            max_probs, max_indexs = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(self.p_cutoff)
            
            if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, max_indexs[mask]], dim=0)
            else:
                self.select_ulb_idx = idx_ulb[mask]
                self.select_ulb_label = y_ulb[mask]
                self.select_ulb_pseudo_label = max_indexs[mask]

            self.count_labeled_dataset += torch.bincount(lb, minlength=self.num_classes)
            self.dis_unlabeled_dataset += torch.sum(pseudo_label[mask], dim=0)
            
            Lx = (F.cross_entropy(logits_x + self.adapt_dis_adjustment, lb, reduction="mean")) / self.scale_ratio

            Lu = (F.cross_entropy(logits_u_s + self.adapt_dis_adjustment, pseudo_label, reduction="none") * mask.float()).mean()
            
            estimate_dis = self.dis_unlabeled_dataset / (self.dis_unlabeled_dataset.sum() + 1)

            self.estimate_dis = self.estimate_dis * self.ema_u + (estimate_dis) * (1 - self.ema_u)

            count_forward = self.count_labeled_dataset + self.dis_unlabeled_dataset

            self.adapt_dis = self.adapt_dis * self.ema_u + (count_forward / count_forward.sum()) * (1 - self.ema_u)
            
            total_loss = Lx + Lu
            
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=Lx.item(), 
                                         unsup_loss=Lu.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())

        return out_dict, log_dict


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tau', float, 1),
            SSL_Argument('--ema_u', float, 0.9),
        ]
