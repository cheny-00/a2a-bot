# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import torch
import typing as tp
import torch.nn.functional as F


def xnet(*args, **kwargs):
    return F.cross_entropy(*args, **kwargs)



def audio_mask_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    # multi layer
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook: tp.List[torch.Tensor] = []
    for k in range(K):
        logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        ce += q_ce
        ce_per_codebook.append(q_ce.detach())
    # average cross entropy across codebooks
    ce = ce / K
    return ce, ce_per_codebook


def text_mask_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    # Check shapes
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape

    # Flatten logits, targets, and mask to 1D arrays
    logits_flat = logits.view(-1, logits.size(-1))  # Flatten to [B * T, card]
    targets_flat = targets.view(-1)  # Flatten to [B * T]
    mask_flat = mask.view(-1)  # Flatten to [B * T]
    
    # Apply mask to logits and targets (use mask to filter out invalid tokens)
    logits_masked = logits_flat[mask_flat]
    targets_masked = targets_flat[mask_flat]
    
    # Calculate cross-entropy loss only for valid tokens
    ce = F.cross_entropy(logits_masked, targets_masked)
    return ce
