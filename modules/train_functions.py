# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import pytorch_lightning as pl
from utils.utils import log_loss

from modules.cal_loss_process import cal_stage_1_loss, cal_stage_2_loss, cal_stage_3_loss


def omni_stage_1_training(self: pl.LightningModule, batch, batch_idx):
    losses, _, _ = cal_stage_1_loss(self, batch)
    train_losses = {f"train_{k}": v for k, v in losses.items()}
    log_loss(self, train_losses, len(batch["task"]), self.is_distributed)

    return train_losses["train_loss"]


def omni_stage_2_training(self: pl.LightningModule, batch, batch_idx):
    losses, _, _ = cal_stage_2_loss(self, batch)
    train_losses = {f"train_{k}": v for k, v in losses.items()}
    log_loss(self, train_losses, len(batch["task"]), self.is_distributed)
    return train_losses["train_loss"]

def omni_stage_3_training(self: pl.LightningModule, batch, batch_idx):
    losses, _, _ = cal_stage_3_loss(self, batch, self.hparams.alpha)
    train_losses = {f"train_{k}": v for k, v in losses.items()}
    log_loss(self, train_losses, len(batch["task"]), self.is_distributed)
    return train_losses["train_loss"]