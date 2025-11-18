# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from semilearn.core import ImbAlgorithmBase
from semilearn.core.utils import IMB_ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from .utils import *

class FixMatchNet(nn.Module):
    def __init__(self, base, num_classes):
        super(FixMatchNet, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        
        self.register_buffer('prototypes', torch.zeros(num_classes, self.feat_planes))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.01)
        
        
        self.register_buffer('true_prototypes', torch.zeros(num_classes, self.feat_planes))
        
        self.register_buffer('global_center', torch.zeros(1, self.feat_planes))

        self.evidence_classifier = nn.Linear(self.feat_planes, num_classes)

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        evidence = F.softplus(self.evidence_classifier(feat))
        return {'feat':feat, 'logits': logits, 'evidence': evidence}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher
    

@IMB_ALGORITHMS.register('fixmatch')
class FixMatch(ImbAlgorithmBase):
    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super(FixMatch, self).__init__(args, net_builder, tb_log, logger)

        self.num_ood_classes = args.num_ood_classes

        self.warm_up_epoch = 60
        self.proupdate = 20
        self.trust_update = 20
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.current_epoch_pseudo_count = torch.zeros(self.num_classes).to(self.device)
        
        self.prototype_momentum = 0.99  
        self.true_prototype_momentum = 0.99  
        self.radial_loss_weight = 0.1  
        self.offset_aware_weight = 1.0  
        self.inter_class_loss_weight = 0.1  
        self.global_class_loss_weight = 0.1
        self.trust_loss_weight = 0.1
        self.lambda_ood_center = 0.1
        self.trustcontrast_loss = 0.1
        
    def train(self):
        """
        train function
        """
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch


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
                if self.epoch < self.warm_up_epoch:
                    self.out_dict, self.log_dict = self.train_warmup_step(**self.process_batch(**data_lb, **data_ulb))
                else:
                    self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
        

    def set_hooks(self):
        super().set_hooks()
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook1")
        
        
    def set_model(self):
        model = super().set_model()  # backbone
        model = FixMatchNet(model, num_classes=self.num_classes)  # including ova classifiers
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = FixMatchNet(ema_model, num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model
        
    def update_true_prototypes(self, feats, labels):
        
        with torch.no_grad():
            for cls_idx in range(self.num_classes):
                cls_mask = (labels == cls_idx)
                if cls_mask.sum() > 0:
                    cls_mean = feats[cls_mask].mean(dim=0)
                    
                    self.model.true_prototypes[cls_idx] = (
                        self.true_prototype_momentum * self.model.true_prototypes[cls_idx] +
                        (1 - self.true_prototype_momentum) * cls_mean
                    )
    def update_ood_prototypes(self, feats, pseudo_labels, evidential_ood_mask):
        with torch.no_grad():
            for cls_idx in range(self.num_classes):
                cls_ood_mask = (pseudo_labels == cls_idx) & evidential_ood_mask
                if cls_ood_mask.sum() > 0:
                    cls_ood_feats = feats[cls_ood_mask]
                    cls_ood_mean = cls_ood_feats.mean(dim=0)
                
                    self.model.ood_prototypes[cls_idx] = (
                        self.ood_prototype_momentum * self.model.ood_prototypes[cls_idx] +
                        (1 - self.ood_prototype_momentum) * cls_ood_mean
                    )
                
    def update_true_prototypes_morelb(self, feats, labels, high_conf_feats=None, high_conf_labels=None):
        with torch.no_grad():
            all_feats = feats
            all_labels = labels
        
        
            if high_conf_feats is not None and high_conf_labels is not None:
                all_feats = torch.cat([all_feats, high_conf_feats], dim=0)
                all_labels = torch.cat([all_labels, high_conf_labels], dim=0)
        
            for cls_idx in range(self.num_classes):
                cls_mask = (all_labels == cls_idx)
                if cls_mask.sum() > 0:
                    cls_mean = all_feats[cls_mask].mean(dim=0)
                    self.model.true_prototypes[cls_idx] = (
                        self.true_prototype_momentum * self.model.true_prototypes[cls_idx] +
                        (1 - self.true_prototype_momentum) * cls_mean
                    )

                    
    def update_global_center(self):
        with torch.no_grad():
            self.model.global_center = self.model.true_prototypes.mean(dim=0, keepdim=True)
            
    
    def calculate_prototype_offset_weights(self, feats, pseudo_labels):
        
        with torch.no_grad():
            
            offsets = self.model.prototypes - self.model.true_prototypes
            
            
            weights = torch.ones(len(feats), device=feats.device)
             
            for i, (feat, label) in enumerate(zip(feats, pseudo_labels)):
                if label < self.num_classes:  
                   
                    proto_to_feat =  feat - self.model.prototypes[label]
                    
                    offset_dir = offsets[label]
                    if torch.norm(offset_dir) > 1e-6 and torch.norm(proto_to_feat) > 1e-6:
                        cosine_sim = F.cosine_similarity(
                            proto_to_feat.unsqueeze(0), 
                            offset_dir.unsqueeze(0)
                        )
                        
                        
                        if cosine_sim > 0:  
                            weights[i] = 1.0 + self.offset_aware_weight * abs(cosine_sim)

            return weights
            
        
    def calculate_prototype_offset_weights_ood(self, feats, pseudo_labels):

        with torch.no_grad():
            weights = torch.ones(len(feats), device=feats.device)
        
            for i, (feat, label) in enumerate(zip(feats, pseudo_labels)):
                if label < self.num_classes: 
                    offsets = self.model.prototypes - self.model.true_prototypes
                    offset_dir = offsets[label]
                    proto_to_feat = feat - self.model.prototypes[label]
                
                    original_weight = 1.0
                    if torch.norm(offset_dir) > 1e-6 and torch.norm(proto_to_feat) > 1e-6:
                        cosine_sim = F.cosine_similarity(
                            proto_to_feat.unsqueeze(0), 
                            offset_dir.unsqueeze(0)
                        )
                        if cosine_sim > 0:  
                            original_weight = 1.0 + self.offset_aware_weight * abs(cosine_sim)
                
                    ood_proto_valid = torch.norm(self.model.ood_prototypes[label]) > 1e-6
                
                    if ood_proto_valid:
                        dist_to_in_proto = torch.norm(feat - self.model.prototypes[label])
                    
                        dist_to_ood_proto = torch.norm(feat - self.model.ood_prototypes[label])
                    
                        if (dist_to_ood_proto < dist_to_in_proto) :
                            weights[i] = 1.0
                        else:
                            weights[i] = original_weight
                    else:
                        weights[i] = original_weight
        
        return weights
            
    def radial_uniformity_loss(self, feats, labels, prototypes):
        
        loss = 0.0
        valid_count = 0
        
        for cls_idx in range(self.num_classes):
            cls_mask = (labels == cls_idx)
            if cls_mask.sum() > 1:  
                cls_feats = feats[cls_mask]
                
                
                centered_feats = cls_feats - prototypes[cls_idx].unsqueeze(0)
                
                
                radii = torch.norm(centered_feats, dim=1, keepdim=True)
                radial_dirs = centered_feats / (radii + 1e-8)
                
                
                if len(radial_dirs) > 1:
                    similarities = torch.mm(radial_dirs, radial_dirs.t())
                    
                    
                    n = len(similarities)
                    mask = ~torch.eye(n, dtype=torch.bool, device=similarities.device)
                    off_diag_similarities = similarities[mask].view(n, n-1)
                    
                    
                    mean_similarity = off_diag_similarities.mean()
                    
                    
                    loss += mean_similarity.pow(2)
                    valid_count += 1
        
        return loss / (valid_count + 1e-8)
        
    def inter_class_distance_loss(self, prototypes):
        normalized_prototypes = F.normalize(prototypes, dim=1)
        similarity_matrix = torch.mm(normalized_prototypes, normalized_prototypes.t())
        
        
        n = len(prototypes)
        mask = ~torch.eye(n, dtype=torch.bool, device=prototypes.device)
        off_diag_similarities = similarity_matrix[mask]
        
        
        target_similarity = -torch.ones_like(off_diag_similarities)
        loss = F.mse_loss(off_diag_similarities, target_similarity)
        
        return loss
    
    def global_class_distribution_loss(self, prototypes, global_center):

        distances = torch.norm(prototypes - global_center, dim=1)
        
        distance_loss = -torch.log(distances.mean() + 1e-8)
        
        variance_loss = distances.var()
        
        return distance_loss + variance_loss

    
    def train_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
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
                evidence_x_lb = outputs['evidence'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                evidence_x_ulb_w, evidence_x_ulb_s = outputs['evidence'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}
    
            #sup_loss
            sup_closed_loss = self.ce_loss(logits_x_lb, lb, reduction='mean')
            
            if lb.dim() == 1:  
                lb = F.one_hot(lb, num_classes=self.num_classes).float()
            trust_sup_loss = self.l_trust_loss(evidence_x_lb, lb, self.epoch, self.warm_up_epoch)
            trust_sup_loss = self.trust_loss_weight * trust_sup_loss
            
            sup_loss = sup_closed_loss 
            
            
            #update true_prototypes
            with torch.no_grad():
                self.update_true_prototypes(feats_x_lb, y_lb)
            self.update_global_center()

            with torch.no_grad():
                p = F.softmax(logits_x_ulb_w, dim=-1)
                targets_p = p.detach()
                if self.registered_hook("DistAlignHook"):
                    targets_p = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_p)

  
            
            p_mask = self.call_hook("masking", "MaskingHook1", logits_x_ulb=targets_p, softmax_x_ulb=False)


            targets_p = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", logits=targets_p,
                                          use_hard_label=self.use_hard_label, T=self.T, softmax=False)


            
            #cauculate evidential mask
            if self.epoch >= self.warm_up_epoch + self.proupdate:

                S = torch.sum(evidence_x_ulb_w + 1, dim=1) 
                uncertainty = self.num_classes / S  
    
                p_mask_indices = torch.where(p_mask)[0]
                if len(p_mask_indices) > 0:
                    p_uncertainty = uncertainty[p_mask_indices]
        
                    k = max(1, int(len(p_uncertainty) * 0.1))
                    threshold = torch.topk(p_uncertainty, k, largest=True)[0][-1]
                    
                    # evidential_ood_mask
                    evidential_ood_mask = torch.zeros_like(p_mask, dtype=torch.bool)
                    evidential_ood_mask[p_mask_indices] = uncertainty[p_mask_indices] >= threshold
        
                    final_mask = (p_mask > 0.5) & ~evidential_ood_mask
                else:
                    evidential_ood_mask = torch.zeros_like(p_mask, dtype=torch.bool)
                   
                    final_mask = p_mask
            else:
                evidential_ood_mask = torch.zeros_like(p_mask, dtype=torch.bool)
                final_mask = p_mask

            mask = final_mask

            #ood_center_loss
            ood_center_loss = torch.tensor(0.0).to(self.device)
            if evidential_ood_mask.sum() > 0 and self.epoch >= self.warm_up_epoch + self.proupdate:
                ood_feats = feats_x_ulb_s[evidential_ood_mask] 
                global_center = self.model.global_center.detach()  
    
                ood_center_loss = F.mse_loss(ood_feats, global_center.expand(ood_feats.size(0), -1), reduction='mean')
                ood_center_loss = self.lambda_ood_center * ood_center_loss
            


            if self.epoch >= self.warm_up_epoch + self.proupdate:
                offset_weights = self.calculate_prototype_offset_weights(feats_x_ulb_w, targets_p)
                
                weighted_mask = mask * offset_weights
                
            else:
                weighted_mask = mask
                

            #ui_loss
            ui_loss = self.consistency_loss(logits_x_ulb_s, targets_p, 'ce', mask=weighted_mask)
            
            #prototype_loss/radial_loss
            mask = mask > 0.5
            prototype_loss = torch.tensor(0.0).to(self.device)
            radial_loss = torch.tensor(0.0).to(self.device)
            inter_class_loss = torch.tensor(0.0).to(self.device)
            global_class_loss = torch.tensor(0.0).to(self.device)
            trust_unsup_loss = torch.tensor(0.0).to(self.device)
            if mask.sum() > 0:
                selected_feats = feats_x_ulb_s[mask]
                selected_labels = targets_p[mask].long()
                valid_mask = (selected_labels < self.num_classes) & (selected_labels >= 0)
                selected_feats = selected_feats[valid_mask]
                selected_labels = selected_labels[valid_mask]
                
                if len(selected_labels) > 0 and self.epoch >= self.warm_up_epoch + self.proupdate:                        
                    # caculate loss
                    prototype_loss = self.prototype_ppp_loss(
                        selected_feats, 
                        selected_labels, 
                        self.model.prototypes,
                        temperature1=0.1
                    )
                    
                    radial_loss = self.radial_uniformity_loss(
                            selected_feats, 
                            selected_labels, 
                            self.model.true_prototypes
                        )
                    
                    
                    inter_class_loss = self.inter_class_distance_loss(self.model.true_prototypes)
                        
                    global_class_loss = self.global_class_distribution_loss(self.model.true_prototypes, self.model.global_center)
                    
                   
                    if self.epoch >= self.warm_up_epoch + self.trust_update:
                        selected_evidence = evidence_x_ulb_s[mask]
                        selected_pseudo_labels = targets_p[mask]
                
                        if selected_pseudo_labels.dim() == 1:  
                            selected_pseudo_labels = F.one_hot(selected_pseudo_labels, num_classes=self.num_classes).float()
                
                        trust_unsup_loss = self.l_trust_loss(selected_evidence, selected_pseudo_labels, self.epoch, self.warm_up_epoch + self.trust_update)
                        trust_unsup_loss = self.trust_loss_weight * trust_unsup_loss
                    
                    
                    # update
                    with torch.no_grad():
                        for cls_idx in range(self.num_classes):
                            cls_mask = (selected_labels == cls_idx)
                            if cls_mask.sum() > 0:
                                cls_mean = selected_feats[cls_mask].mean(dim=0)
                                self.model.prototypes[cls_idx] = (self.prototype_momentum * self.model.prototypes[cls_idx] + (1 - self.prototype_momentum) * cls_mean)
                elif len(selected_labels) > 0:
                    # update
                    with torch.no_grad():
                        for cls_idx in range(self.num_classes):
                            cls_mask = (selected_labels == cls_idx)
                            if cls_mask.sum() > 0:
                                cls_mean = selected_feats[cls_mask].mean(dim=0)
                                self.model.prototypes[cls_idx] = (self.prototype_momentum * self.model.prototypes[cls_idx] + (1 - self.prototype_momentum) * cls_mean)       
                            
            prototype_loss = 0.1 * prototype_loss 
            if isinstance(radial_loss, torch.Tensor):
                radial_loss = self.radial_loss_weight * radial_loss
            else:
                radial_loss = torch.tensor(self.radial_loss_weight * radial_loss, device=self.device)
 
            inter_class_loss = self.inter_class_loss_weight * inter_class_loss 
            global_class_loss = self.global_class_loss_weight * global_class_loss


            similarity2  = pairwise_similarity(evidence_x_ulb_w, evidence_x_ulb_s)
            contrast_loss  =  NT_xent(similarity2)
            contrast_loss = self.trustcontrast_loss * contrast_loss

            
            unsup_loss = self.lambda_u * ui_loss 
            

            total_loss = sup_loss + trust_sup_loss + trust_unsup_loss + unsup_loss + prototype_loss + radial_loss + inter_class_loss + global_class_loss + contrast_loss + ood_center_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         prototype_loss=prototype_loss.item(),
                                         radial_loss = radial_loss.item(),
                                         inter_class_loss = inter_class_loss.item(),
                                         global_class_loss = global_class_loss.item(),
                                         trust_sup_loss = trust_sup_loss.item(),
                                         trust_unsup_loss = trust_unsup_loss.item(),
                                         contrast_loss = contrast_loss.item(),
                                         ood_center_loss = ood_center_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
    
    
    
    def edl_loss(self, evidence, y):

        alpha = evidence + 1  
        S = torch.sum(alpha, dim=1, keepdim=True) 
    
        
        loss_per_sample = torch.sum(y * (torch.log(S) - torch.log(alpha)), dim=1)
        loss = torch.mean(loss_per_sample)
    
        return loss
    
    def l_trust_loss(self, evidence, pseudo_labels, epoch, trust_start):
        alpha = evidence + 1
        
        S = torch.sum(alpha, dim=1, keepdim=True)
        
      
        annealing_coef = min(0.6, max(0, (epoch - trust_start)/10) )###########ori_0.6
        
        A = torch.sum(pseudo_labels * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        
        alp = evidence * (1 - pseudo_labels) + 1
        B = annealing_coef * self.kl_trust(alp)
        
       
        loss = torch.mean(A) + torch.mean(B)
        
        return loss#, torch.mean(A), torch.mean(B)
    
    def kl_trust(self, alpha):
        beta = torch.ones((1, alpha.shape[1])).to(alpha.device)
        
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        
        return kl
        
    def compute_ood_evidence_loss(self, evidence, evidential_ood_mask):
        if evidential_ood_mask.sum() == 0:
            return torch.tensor(0.0).to(self.device)
    
        ood_evidence = evidence[evidential_ood_mask]  # [n_ood, num_classes]
    
        evidence_loss = torch.mean(torch.sum(ood_evidence ** 2, dim=1))
    
        return evidence_loss
        
    def prototype_ppp_loss(self, features, labels, prototypes, temperature1=0.1):
    
    
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(prototypes.detach(), dim=1)
    
        sim_matrix = torch.matmul(features, prototypes.t()) / temperature1
    
        class_sim = sim_matrix[torch.arange(features.size(0)), labels]
    
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        mask[torch.arange(features.size(0)), labels] = False
    
       
        exp_sim = torch.exp(sim_matrix)
        denominator = exp_sim[mask].view(features.size(0), -1).sum(dim=1)
    
     
        log_probs = class_sim - torch.log(denominator)
        loss = -log_probs.mean()
    
        return loss

    def train_warmup_step(self, idx_lb, x_lb_w, x_lb_s, y_lb, y_lb_noised, idx_ulb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
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
                evidence_x_lb = outputs['evidence'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                evidence_x_ulb_w, evidence_x_ulb_s = outputs['evidence'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}
 
            with torch.no_grad():
                self.update_true_prototypes(feats_x_lb, y_lb)
            
            inter_class_loss = self.inter_class_distance_loss(self.model.true_prototypes)
                        
            global_class_loss = self.global_class_distribution_loss(self.model.true_prototypes, self.model.global_center)
            
            inter_class_loss = self.inter_class_loss_weight * inter_class_loss 
            global_class_loss = self.global_class_loss_weight * global_class_loss
            
            
            similarity1  = pairwise_similarity(logits_x_ulb_w, logits_x_ulb_s) 
            similarity2  = pairwise_similarity(evidence_x_ulb_w, evidence_x_ulb_s)
            unsup_loss        = NT_xent(similarity1) +   (self.trustcontrast_loss * NT_xent(similarity2))
  

            total_loss = unsup_loss  + inter_class_loss + global_class_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(unsup_loss=unsup_loss.item(),
                                         global_class_loss = global_class_loss.item(),
                                         inter_class_loss = inter_class_loss.item(),
                                         total_loss=total_loss.item(),
                                         )
        return out_dict, log_dict


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--mb_loss_ratio', float, 1.0),
         ]