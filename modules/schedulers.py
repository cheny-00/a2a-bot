# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy


import torch
import math
import torch.nn as nn
import pytorch_lightning as pl



def litgpt_get_lr(learning_rate, it, warmup_iters, max_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
