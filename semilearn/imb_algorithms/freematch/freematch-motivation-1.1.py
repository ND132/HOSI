# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from semilearn.core import ImbAlgorithmBase
from .utils import FreeMatchThresholdingHook
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


@IMB_ALGORITHMS.register('freematch')
class FreeMatch(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(FreeMatch, self).__init__(args, net_builder, tb_log, logger)
        self.lambda_e = args.ent_loss_ratio
        self.clip_thresh = args.clip_thresh
        self.use_quantile = args.use_quantile

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        
        ulb_class_dist = [0 for _ in range(self.num_classes)]
        for c in self.dataset_dict['train_ulb'].targets:
            ulb_class_dist[c] += 1
        ulb_class_dist = np.array(ulb_class_dist)

        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).cuda(args.gpu)

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

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes), "MaskingHook")

    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb_w, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb_w)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, lb, reduction='mean')

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=logits_x_ulb_w)
            
            if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, pseudo_label[mask]], dim=0)
            else:
                self.select_ulb_idx = idx_ulb[mask]
                self.select_ulb_label = y_ulb[mask]
                self.select_ulb_pseudo_label = pseudo_label[mask]

            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--ent_loss_ratio', float, 0.05),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
        ]