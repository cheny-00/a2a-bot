# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch
from modules.loss_functions import text_mask_cross_entropy
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


def omni_stage_2_validation(self: pl.LightningModule, batch, batch_idx):
    question_audio_feature = batch['question_audio_feature']
    input_ids = batch['input_ids']
    question_audio_length = batch['question_audio_length']
    answer_token = batch['answer_token']
    answer_token_length = batch['answer_token_length']
    
    _, logit_t = self(question_audio_feature, input_ids, whisper_lens=question_audio_length, task='AT')
    max_length = answer_token.size(1)  # T (max sequence length)
    text_indices = torch.arange(max_length, device=answer_token.device).unsqueeze(0)  # [1, T]
    text_mask = text_indices < answer_token_length.unsqueeze(1)
    
    val_loss = text_mask_cross_entropy(logit_t, answer_token, text_mask)
    
    self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
    if self.metrics is not None and "val_text_acc" in self.metrics:
        pred_ids = sample(logit_t)
        text_acc = self.val_text_acc.update(pred_ids, answer_token)
        self.log("val_text_acc", text_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    return val_loss
