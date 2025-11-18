import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Optional
from semilearn.algorithms.hooks import MaskingHook


import sys
import time
import math
import torch.nn.init as init


   
def mb_unsup_loss(logits_ova, ood_mask):
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, dim=1)
    
    # 
    #known_mask = ~ood_mask
    
    # 
    #if torch.any(known_mask):
    #    label_sp_neg = torch.ones((batch_size, num_classes)).to(ood_mask.device)
    #    neg_log_probs = -torch.log(probs_ova[known_mask, 0, :] + 1e-8
    #    open_loss_neg = torch.mean(torch.max(neg_log_probs * label_sp_neg[known_mask], dim=1)[0])
    #else:
    #    open_loss_neg = torch.tensor(0.0).to(ood_mask.device)
    
    # 
    if torch.any(ood_mask):
        ood_neg_probs = probs_ova[ood_mask, 1, :]
        ood_loss = torch.mean(-torch.log(1 - ood_neg_probs + 1e-8))
    else:
        ood_loss = torch.tensor(0.0).to(ood_mask.device)
    
    # 
    total_loss = ood_loss #+ open_loss_neg 
    return total_loss

def mb_kl_unsup_loss(logits_ova, ood_mask):
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, dim=1)
    
    if torch.any(ood_mask):
        ood_pos_probs = probs_ova[ood_mask, 1, :]
        #target = torch.full_like(ood_neg_probs, 0.5)
        #ood_loss = F.mse_loss(ood_neg_probs, target)
        
        log_ood_pos_probs = torch.log(ood_pos_probs + 1e-8)
        target = torch.full_like(ood_pos_probs, 0.5)
        ood_loss = F.kl_div(log_ood_pos_probs, target, reduction='batchmean')
    else:
        ood_loss = torch.tensor(0.0).to(ood_mask.device)
    
    total_loss = ood_loss
    return total_loss

# Reference: https://github.com/VisionLearningGroup/OP_Match/blob/main/utils/misc.py
def mb_sup_loss(logits_ova, label):
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, 1)
    label_s_sp = torch.zeros((batch_size, num_classes)).long().to(label.device)
    label_range = torch.arange(0, batch_size).long().to(label.device)
    label_s_sp[label_range[label < num_classes], label[label < num_classes]] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(probs_ova[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(probs_ova[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova_sup = open_loss_neg + open_loss
    return l_ova_sup

def pairwise_similarity(outputs_1, outputs_2,temperature=0.5):
    '''
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity
    '''  
    outputs=torch.cat((outputs_1,outputs_2),dim=0)
    B   = outputs.shape[0]
    outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    similarity_matrix = (1./temperature) * torch.mm(outputs_norm,outputs_norm.transpose(0,1))
    return similarity_matrix
 
 
def NT_xent(similarity_matrix):
    '''
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    ''' 

    N2  = len(similarity_matrix)
    N   = int(len(similarity_matrix) / 2)

    # Removing diagonal #
    similarity_matrix_exp = torch.exp(similarity_matrix)
    similarity_matrix_exp = similarity_matrix_exp * (1 - torch.eye(N2,N2)).cuda()

    NT_xent_loss        = - torch.log(similarity_matrix_exp/(torch.sum(similarity_matrix_exp,dim=1).view(N2,1) + 1e-8) + 1e-8)
    NT_xent_loss_total  = (1./float(N2)) * torch.sum(torch.diag(NT_xent_loss[0:N,N:]) + torch.diag(NT_xent_loss[N:,0:N]))

    return NT_xent_loss_total
