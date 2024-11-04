# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import torch.nn as nn
import torch.nn.functional as F


def xnet(*args, **kwargs):
    return F.cross_entropy(*args, **kwargs)

