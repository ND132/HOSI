# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


@IMB_ALGORITHMS.register('supervise')
class SuperviseLogitAdj(ImbAlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(SuperviseLogitAdj, self).__init__(args, net_builder, tb_log, logger)
        
        # 初始化logits调整参数
        self.logit_adjustment = args.logit_adjustment
        self.tau = args.tau if hasattr(args, 'tau') else 1.0
        
        # 初始化类别先验概率(将在第一次训练步骤中计算)
        self.class_prior = None
    
    def compute_class_prior(self):
        """计算并更新类别先验概率"""
        y = torch.tensor(self.dataset_dict['targets'][self.dataset_dict['lb_idx']], dtype=torch.long)
        class_counts = torch.bincount(y, minlength=self.num_classes)+10
        self.class_prior = (class_counts.float() / class_counts.sum()).clamp(min=0.05, max=0.5)
        self.class_prior = self.class_prior.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def adjust_logits(self, logits):
        """
        Logits调整方法
        支持两种调整方式:
        1. 基于类别先验的减法调整
        2. 基于类别先验的除法调整
        """
        if not self.logit_adjustment or self.class_prior is None:
            return logits
            
        if self.logit_adjustment == 'subtract':
            # 减法调整: logits -= tau * log(prior)
            adjustment = torch.log(self.class_prior + 1e-12)
            adjusted_logits = logits - self.tau * adjustment
        elif self.logit_adjustment == 'divide':
            # 除法调整: logits /= prior^tau
            adjustment = torch.pow(self.class_prior, self.tau) + 1e-12
            adjustment = adjustment.clamp(min=0.05)
            adjusted_logits = logits / adjustment
            
            adjusted_logits = torch.where(
            torch.isnan(adjusted_logits), 
            logits,  # 回退原始值
            adjusted_logits)
        else:
            raise ValueError(f"Unknown logit adjustment method: {self.logit_adjustment}")
        
        return adjusted_logits
    
    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        
        # 第一次迭代时计算类别先验
        if self.class_prior is None:
            self.compute_class_prior()
        
        if self.args.noise_ratio > 0:
            lb = y_lb_noised
        else:
            lb = y_lb

        # inference and calculate sup losses
        with self.amp_cm():
            inputs = x_lb_w
            outputs = self.model(inputs)
            logits_x_lb = outputs['logits']
            feats_x_lb = outputs['feat']
            
            # 应用logits调整
            adjusted_logits = self.adjust_logits(logits_x_lb)
            
            feat_dict = {'x_lb': feats_x_lb}

            # 使用调整后的logits计算损失
            sup_loss = self.ce_loss(adjusted_logits, lb, reduction='mean')

            total_loss = sup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            total_loss=total_loss.item(),
            class_prior=self.class_prior.cpu().numpy().tolist() if self.class_prior is not None else None
        )
        return out_dict, log_dict