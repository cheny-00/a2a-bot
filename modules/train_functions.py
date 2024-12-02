# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import torch
from mini_omni.litgpt.generate.base import sample
import pytorch_lightning as pl
from modules.loss_functions import text_mask_cross_entropy, audio_mask_cross_entropy

from modules.cal_loss_process import cal_stage_1_loss, cal_stage_2_loss, cal_stage_3_loss

def omni_stage_1_training(self: pl.LightningModule, batch, batch_idx):
    loss, _, _ = cal_stage_1_loss(self, batch)

    self.log(
        f"{self.task}/train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        batch_size=len(batch["task"]),
        sync_dist=self.is_distributed
    )

    return loss


def omni_stage_2_training(self: pl.LightningModule, batch, batch_idx):
    loss, _, _ = cal_stage_2_loss(self, batch)
    
    self.log(
        f"{self.task}/train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        batch_size=len(batch["task"]),
        sync_dist=self.is_distributed
    )

    return loss
    
def omni_stage_3_training(self: pl.LightningModule, batch, batch_idx):
    alpha = 0.5
    losses, _, _ = cal_stage_3_loss(self, batch, alpha)
    loss = losses["loss"]
    audio_loss = losses["audio_loss"]
    text_loss = losses["text_loss"]
    
    self.log(
        f"{self.task}/text_loss",
        text_loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )
    self.log(
        f"{self.task}/audio_loss",
        audio_loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )
    self.log(
        f"{self.task}/train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )
    
    return loss
