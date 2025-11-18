# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from collections import deque
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .utils import SoftMatchWeightingHook, DistAlignEMAHook


@IMB_ALGORITHMS.register('scomatch')
class SCOMatch(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # SCOMatch 特有参数
        self.selected_ood_maxlength = max(8 * args.num_classes, 256)
        self.selected_ood_update_length = args.Km
        self.selected_ood_count = 0
        self.selected_ood_scores = deque(maxlen=self.selected_ood_maxlength)
        self.selected_ood_labels = deque(maxlen=self.selected_ood_maxlength)
        self.selected_ood_images = deque(maxlen=self.selected_ood_maxlength)
        self.all_sample_scores = [[] for _ in range(args.num_classes + 1)]
        self.ood_threshold = args.ood_threshold
        self.start_fix = args.start_fix
        self.T = args.scoT
        self.mu = args.mu
        self.batch_size = args.batch_size
        self.threshold = args.scothreshold
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.threshold_update_freq = (self.train_ulb_len) // int(args.batch_size * args.mu * 2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pseudo_label_stats = {
            'total_correct': 0,
            'total_selected': 0,
            'class_correct': torch.zeros(self.num_classes).to(self.device),
            'class_total': torch.zeros(self.num_classes).to(self.device),
            'epoch_correct': 0,
            'epoch_selected': 0,
            'epoch_class_correct': torch.zeros(self.num_classes).to(self.device),
            'epoch_class_total': torch.zeros(self.num_classes).to(self.device)
        }

    # def set_hooks(self):
    #     super().set_hooks()
    #     self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
    #     self.register_hook(DistAlignEMAHook(num_classes=self.num_classes, momentum=0.999), "DistAlignHook")
    #     self.register_hook(OODThresholdHook(num_classes=self.num_classes,start_fix=self.start_fix,ood_threshold=self.ood_threshold),"OODThresholdHook")
    # def set_hooks(self):
    #     super().set_hooks()
    #     # 注册 SCOMatch 特有的 Hook
    #     self.register_hook(
    #         PseudoLabelingHook(), 
    #         "PseudoLabelingHook"
    #     )
    #     self.register_hook(
    #         DistAlignEMAHook(
    #             num_classes=self.num_classes,
    #             momentum=self.args.ema_p,
    #             p_target_type='uniform' if self.args.dist_uniform else 'model'
    #         ), 
    #         "DistAlignHook"
    #     )
    #     self.register_hook(
    #         SCOMatchWeightingHook(
    #             num_classes=self.num_classes,
    #             n_sigma=self.args.n_sigma,
    #             momentum=self.args.ema_p,
    #             per_class=self.args.per_class,
    #             ood_threshold=self.ood_threshold,
    #             selected_ood_maxlength=self.selected_ood_maxlength,
    #             Km=self.selected_ood_update_length
    #         ),
    #         "SCOMatchMaskingHook"
    #     )
    

    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")
        #self.train_ulb_len = len(self.dataset_dict['train_ulb'])
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch

                # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
                

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb, data_ulb_all in zip(self.loader_dict['train_lb'],self.loader_dict['train_ulb'],self.loader_dict['train_ulb_all']):
                    # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                combined_data = {
                    'idx_lb': data_lb['idx_lb'],
                    'x_lb_w': data_lb['x_lb_w'],
                    'x_lb_s': data_lb['x_lb_s'],
                    'y_lb': data_lb['y_lb'],
                    'idx_ulb': data_ulb['idx_ulb'],
                    'x_ulb_w': data_ulb['x_ulb_w'],
                    'x_ulb_s': data_ulb['x_ulb_s'],
                    'y_ulb': data_ulb['y_ulb'],
                    'idx_ulb_all': data_ulb_all['idx_ulb'],
                    'x_ulb_all_w': data_ulb_all['x_ulb_w'],
                    'x_ulb_all_s': data_ulb_all['x_ulb_s'],
                    'y_ulb_all': data_ulb_all['y_ulb']}
                    #self.out_dict, self.log_dict = self.train_step_sco(**self.process_batch_sco(**data_lb, **data_ulb, **data_ulb_all))
                self.out_dict, self.log_dict = self.train_step_sco(**self.process_batch_sco(**combined_data))
                self.call_hook("after_train_step")
                self.it += 1
                
            self.print_pseudo_label_accuracy(epoch_wise=True)#################
            self.pseudo_label_stats['epoch_correct'] = 0
            self.pseudo_label_stats['epoch_selected'] = 0
            self.pseudo_label_stats['epoch_class_correct'].zero_()
            self.pseudo_label_stats['epoch_class_total'].zero_()

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")

    def train_step_sco(self, idx_lb, x_lb_w, x_lb_s, y_lb, idx_ulb, x_ulb_w, x_ulb_s, y_ulb, idx_ulb_all, x_ulb_all_w, x_ulb_all_s, y_ulb_all):
        num_lb = y_lb.shape[0]


        if self.it % self.threshold_update_freq == 0 and self.it > 0 and self.epoch >= self.start_fix:
            max_len = sum([len(self.all_sample_scores[i]) for i in range(self.num_classes)])
            ood_len = len(self.all_sample_scores[-1])
            if max_len > 0:
                ratio = ood_len / (max_len)
                ood_threshold = self.threshold * (ratio)
                ood_threshold = min(0.95, max(0.75, ood_threshold))
            else:
                ood_threshold = self.ood_threshold
            self.all_sample_scores = [[] for i in range(self.num_classes + 1)]

        # 合并输入（包含有标签数据、无标签弱增强、无标签强增强）
        inputs = torch.cat([x_lb_w, x_ulb_all_w, x_ulb_all_s, x_ulb_w, x_ulb_s], dim=0)
        inputs = inputs.to(self.device)
        
        # 前向传播
        with self.amp_cm():

            logits_all, logits_p, feats_all, _ = self.model(inputs)
            logits_id_lb = logits_p[:num_lb]
            
            # 拆分不同增强版本的结果
            logits_open_w, logits_open_s, logits_close_w, logits_close_s = logits_p[num_lb:].chunk(4)

            # 监督损失
            L_sup_close = F.cross_entropy(logits_id_lb, y_lb.to(self.device))

            # OOD 监督损失（如果已选择 OOD 样本）
            if self.selected_ood_count >= self.batch_size:
                # ood_samples = self._get_ood_samples()
                #logits_ood_lb = self.model(ood_samples)[1]
                # L_sup_open = self._compute_ood_loss(logits_ood_lb)
                indices = torch.randperm(len(self.selected_ood_images))[:self.batch_size]
                ood_samples = torch.stack(list(self.selected_ood_images))[indices]
                ood_samples = ood_samples.to(self.device) 
                _, logits_ood_lb, _, _ = self.model(ood_samples)
                #ood_label = torch.full((self.batch_size,), self.num_classes).to(self.device)
                ood_label = (torch.ones(self.batch_size) * self.num_classes).to(self.device).long()
                ood_mask = (torch.tensor(self.selected_ood_scores)[indices] < self.threshold).to(self.device)
                L_sup_open = (F.cross_entropy(logits_ood_lb, ood_label, reduction='none') * ood_mask).mean()             
            else:
                L_sup_open = torch.tensor(0.0).to(self.device)

            # 无监督损失
            # L_unsup_close, L_unsup_open = self._compute_unsup_loss(
            #     logits_open_w, logits_open_s,
            #     logits_close_w, logits_close_s,
            #     x_ulb_w, x_ulb_s
            # )
             # 伪标签生成
            pseudo_label_open = torch.softmax(logits_open_w.detach() / self.T, dim=-1)
            max_probs, targets_u_all = torch.max(pseudo_label_open, dim=-1)

            # 更新 OOD 缓存和阈值
            for prob, target in zip(max_probs, targets_u_all):
                if prob > self.threshold:
                    self.all_sample_scores[target.item()].append(prob.item())

            max_probs_open, _ = torch.max(pseudo_label_open[:, :self.num_classes], dim=-1)
            _, indices = torch.sort(max_probs_open)
            indices = indices[:self.selected_ood_update_length]
            if self.selected_ood_count < self.selected_ood_maxlength:
                self.selected_ood_count += self.selected_ood_update_length
            for prob, img, ulab in zip(max_probs_open[indices], x_ulb_all_w[indices], y_ulb_all[indices]):
                self.selected_ood_scores.append(prob.item())
                self.selected_ood_images.append(img)
                self.selected_ood_labels.append(ulab.item())

        
             # 计算开放集和封闭集损失
            max_probs_open, targets_u_all_open = torch.max(pseudo_label_open, dim=-1)
            mask_pos = max_probs_open.ge(self.threshold) & (targets_u_all_open < self.num_classes)
            mask_pos = mask_pos | ((max_probs_open.ge(self.ood_threshold)) & (targets_u_all_open == self.num_classes))
            if self.dataset == 'cifar10':
                L_unsup_open = (
                        F.cross_entropy(
                            torch.cat([logits_open_s], dim=0), targets_u_all_open,
                            reduction='none') * mask_pos
                ).mean()
            else:
               L_unsup_open = (
                        F.cross_entropy(
                            torch.cat([logits_open_w, logits_open_s], dim=0), targets_u_all_open.repeat(2),
                            reduction='none') * mask_pos.repeat(2)
                ).mean()

            logits_p_u_close_w = logits_close_w[:, :self.num_classes]
            logits_p_u_close_s = logits_close_s[:, :self.num_classes]

            pseudo_close = torch.softmax(logits_p_u_close_w.detach() / self.T, dim=-1)
            pseudo_open = torch.softmax(logits_close_w.detach() / self.T, dim=-1)

            max_probs_close, targets_close = torch.max(pseudo_close, dim=-1)
            max_probs_open, targets_open = torch.max(pseudo_open, dim=-1)

            mask = max_probs.ge(self.threshold).float()
            id_mask = (targets_open < self.num_classes)
            L_unsup_close = (F.cross_entropy(logits_p_u_close_s,
                                     targets_close,
                                     reduction='none') * (mask * id_mask)).mean()

            if self.epoch < self.start_fix:
                L_unsup_open = torch.zeros(1).to(self.device).mean()
                L_sup_open = torch.zeros(1).to(self.device).mean()
            # 总损失
            total_loss = L_sup_close + L_sup_open + L_unsup_close + L_unsup_open
            
            self.update_pseudo_label_accuracy(mask * id_mask, targets_close, y_ulb)

        # 更新 OOD 缓存和阈值
        # self._update_ood_selection(logits_open_w, x_ulb_w, y_ulb)
        # if (self.it % self.args.update_freq) == 0:
        #     self._update_ood_threshold()

        # 返回损失和日志
        out_dict = self.process_out_dict(loss=total_loss, feats=feats_all)
        log_dict = self.process_log_dict(
            sup_close_loss=L_sup_close.item(),
            sup_open_loss=L_sup_open.item(),
            unsup_close_loss=L_unsup_close.item(),
            unsup_open_loss=L_unsup_open.item(),
            total_loss=total_loss.item()
        )
        return out_dict, log_dict



    def _compute_unsup_loss(self, logits_open_w, logits_open_s, logits_close_w, logits_close_s, x_ulb_w, x_ulb_s):
        # 伪标签生成
        pseudo_label_open = torch.softmax(logits_open_w.detach() / self.T, dim=-1)
        max_probs, targets_u_all = torch.max(pseudo_label_open, dim=-1)
        
         # 计算开放集和封闭集损失
        mask_pos = max_probs_open.ge(self.threshold) & (targets_u_all_open < self.num_classes)
        mask_pos = mask_pos | ((max_probs_open.ge(self.ood_threshold)) & (targets_u_all_open == self.num_classes))
        if self.dataset == 'cifar10':
            L_unsup_open = (
                    F.cross_entropy(
                        torch.cat([logits_open_s], dim=0), targets_u_all_open,
                        reduction='none') * mask_pos
            ).mean()
        else:
            L_unsup_open = (
                    F.cross_entropy(
                        torch.cat([logits_open_w, logits_open_s], dim=0), targets_u_all_open.repeat(2),
                        reduction='none') * mask_pos.repeat(2)
            ).mean()

        logits_p_u_close_w = logits_close_w[:, :num_classes]
        logits_p_u_close_s = logits_close_s[:, :num_classes]

        pseudo_close = torch.softmax(logits_p_u_close_w.detach() / self.T, dim=-1)
        pseudo_open = torch.softmax(logits_close_w.detach() / self.T, dim=-1)

        max_probs_close, targets_close = torch.max(pseudo_close, dim=-1)
        max_probs_open, targets_open = torch.max(pseudo_open, dim=-1)

        mask = max_probs.ge(self.threshold).float()
        id_mask = (targets_open < self.num_classes)
        L_unsup_close = (F.cross_entropy(logits_p_u_close_s,
                                 targets_close,
                                 reduction='none') * (mask * id_mask)).mean()

       
        return L_unsup_close, L_unsup_open

    def _update_ood_selection(self, logits_open_w, x_ulb_w, y_ulb):
        max_probs_open, _ = torch.max(logits_open_w[:, :self.num_classes], dim=-1)
        _, indices = torch.sort(max_probs_open)
        indices = indices[:self.selected_ood_update_length]
        
        for prob, img, ulab in zip(max_probs_open[indices], x_ulb_w[indices], y_ulb[indices]):
            self.selected_ood_scores.append(prob.item())
            self.selected_ood_images.append(img.cpu())
            self.selected_ood_labels.append(ulab.item())
            

    def update_pseudo_label_accuracy(self, mask, pseudo_labels, true_labels):

        if mask.sum() == 0:
            return
            
      
        selected_mask = mask > 0.5
        selected_pseudo = pseudo_labels[selected_mask]
        selected_true = true_labels[selected_mask]
        
  
        correct_mask = (selected_pseudo == selected_true)
        
 
        total_correct = correct_mask.sum().item()
        total_selected = selected_mask.sum().item()
        
        self.pseudo_label_stats['total_correct'] += total_correct
        self.pseudo_label_stats['total_selected'] += total_selected
        self.pseudo_label_stats['epoch_correct'] += total_correct
        self.pseudo_label_stats['epoch_selected'] += total_selected
        

        for cls_idx in range(self.num_classes):

            cls_mask = (selected_pseudo == cls_idx)
            if cls_mask.sum() > 0:
                cls_correct = (selected_true[cls_mask] == cls_idx).sum().item()
                cls_total = cls_mask.sum().item()
                
                self.pseudo_label_stats['class_correct'][cls_idx] += cls_correct
                self.pseudo_label_stats['class_total'][cls_idx] += cls_total
                self.pseudo_label_stats['epoch_class_correct'][cls_idx] += cls_correct
                self.pseudo_label_stats['epoch_class_total'][cls_idx] += cls_total
    
    def print_pseudo_label_accuracy(self, epoch_wise=True):

        if epoch_wise:
            total_correct = self.pseudo_label_stats['epoch_correct']
            total_selected = self.pseudo_label_stats['epoch_selected']
            class_correct = self.pseudo_label_stats['epoch_class_correct']
            class_total = self.pseudo_label_stats['epoch_class_total']
            prefix = "Epoch"
        else:
            total_correct = self.pseudo_label_stats['total_correct']
            total_selected = self.pseudo_label_stats['total_selected']
            class_correct = self.pseudo_label_stats['class_correct']
            class_total = self.pseudo_label_stats['class_total']
            prefix = "Cumulative"
            
        if total_selected == 0:
            self.print_fn(f"{prefix} Pseudo Label Accuracy: No samples selected")
            return
            

        overall_accuracy = total_correct / total_selected
        
        self.print_fn(f"{prefix} Pseudo Label Accuracy:")
        self.print_fn(f"  Overall: {overall_accuracy:.4f} ({total_correct}/{total_selected})")
        

        self.print_fn("  Per-class accuracy:")
        for cls_idx in range(self.num_classes):
            if class_total[cls_idx] > 0:
                cls_accuracy = class_correct[cls_idx].item() / class_total[cls_idx].item()
                self.print_fn(f"    Class {cls_idx}: {cls_accuracy:.4f} ({int(class_correct[cls_idx])}/{int(class_total[cls_idx])})")
            else:
                self.print_fn(f"    Class {cls_idx}: No samples")
        

        if class_total.sum() > 0:
            weighted_accuracy = (class_correct.sum() / class_total.sum()).item()
            self.print_fn(f"  Weighted average: {weighted_accuracy:.4f}")

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--ood_threshold', float, 0.95),
            SSL_Argument('--scothreshold', float, 0.95),
            SSL_Argument('--Km', int, 1),  # OOD 更新长度
            SSL_Argument('--start_fix', int, 0),  # 开始固定阈值的 epoch
            SSL_Argument('--scoT', float, 1.0),  # 温度系数
            SSL_Argument('--mu', int, 2),  # 无标签数据增强倍数
        ] #+ super(SCOMatch, SCOMatch).get_argument()