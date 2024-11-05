# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import torch
from mini_omni.litgpt.generate.base import sample
import pytorch_lightning as pl


def omni_stage_1_training(self: pl.LightningModule, batch, batch_idx):
    audio_feature = batch['audio_feature']
    input_ids = batch['input_ids']
    audio_length = batch['audio_length']
    target = batch['text']

    logit_a, logit_t = self(audio_feature, input_ids, whisper_lens=audio_length, task='asr')

    loss = self.loss_function["xnet"](logit_t.reshape(-1, logit_t.size(-1)), target.reshape(-1))

    self.log(
        "train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )
    if self.metrics is not None and "train_text_acc" in self.metrics:
        pred_ids = torch.argmax(logit_t, dim=-1)
        text_acc = self.train_text_acc.update(pred_ids, target)
        self.log("train_text_acc", text_acc, on_step=True, on_epoch=True, prog_bar=True)

    return loss
