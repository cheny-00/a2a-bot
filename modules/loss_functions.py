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
    logits_flat = logits.view(-1, logits.size(-1))  # Flatten to [B * T, D]
    targets_flat = targets.view(-1)  # Flatten to [B * T]
    mask_flat = mask.view(-1)  # Flatten to [B * T]
    
    # Apply mask to logits and targets (use mask to filter out invalid tokens)
    logits_masked = logits_flat[mask_flat]
    targets_masked = targets_flat[mask_flat]
    # Calculate cross-entropy loss only for valid tokens
    ce = chunked_cross_entropy(logits_masked, targets_masked)
    print(f"ce: {ce}", logits_masked, targets_masked, logits_masked.shape, targets_masked.shape)
    return ce


# from https://github.com/Lightning-AI/litgpt/blob/main/litgpt/utils.py#L294
def chunked_cross_entropy(
    logits: tp.Union[torch.Tensor, tp.List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        # See [non_masked_elems div note]
        return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))

    # no chunking at all
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

    # lm_head wasn't chunked, chunk cross entropy
    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    non_masked_elems = (targets != ignore_index).sum()
    # [non_masked_elems div note]:
    #   max(1, non_masked_elems) would be more ergonomic to avoid a division by zero. However that
    #   results in a python int which is then passed back to torch division. By using the
    #   `x.maximum(torch.ones_like(x))` pattern we avoid a cudaStreamSynchronize.
    return torch.cat(loss_chunks).sum() / non_masked_elems.maximum(torch.ones_like(non_masked_elems))