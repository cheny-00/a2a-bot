# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch

def omni_stage_1_validation(self: pl.LightningModule, batch, batch_idx):
    audio_feature = batch['audio_feature']
    input_ids = batch['input_ids']
    audio_length = batch['audio_length']
    target = batch['text']

    logit_a, logit_t = self(audio_feature, input_ids, whisper_lens=audio_length, task='asr')

    val_loss = self.loss_function["xnet"](logit_t.reshape(-1, logit_t.size(-1)), target.reshape(-1))
    
    # Log both loss and WER
    self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
    if self.metrics is not None and "val_text_acc" in self.metrics:
        pred_ids = sample(logit_t)
        text_acc = self.val_text_acc.update(pred_ids, target)
        self.log("val_text_acc", text_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    return val_loss