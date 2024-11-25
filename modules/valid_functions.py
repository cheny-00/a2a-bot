# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch
from modules.loss_functions import text_mask_cross_entropy
from modules.loss_functions import audio_mask_cross_entropy

def omni_stage_1_validation(self: pl.LightningModule, batch, batch_idx):
    audio_feature = batch['audio_feature']
    input_ids = batch['input_ids']
    audio_length = batch['audio_length']
    target_token = batch['text_token']
    target_token_mask = batch['text_token_mask']
    target_text = batch['text']
    task = batch['task']
    pad_t = self.token_config["pad_t"]
    
    logit_a, logit_t = self(audio_feature, input_ids, whisper_lens=audio_length, task=task)

    shifted_target_token = target_token[..., 1:].contiguous()
    shifted_logits = logit_t[..., :-1, :].contiguous() 
    shifted_target_token_mask = target_token_mask[..., 1:].contiguous()
    loss_text = text_mask_cross_entropy(shifted_logits, shifted_target_token, shifted_target_token_mask, ignore_index=pad_t)
    val_loss = loss_text
    
    self.log(
        f"{self.task}/val_loss",
        val_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    if self.metrics is not None and "val_text_acc" in self.metrics:
        pred_ids = sample(logit_t)
        text_acc = self.val_text_acc.update(pred_ids, target_token)
        self.log(
            f"{self.task}/val_text_acc",
            text_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
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
    
    self.log(
        f"{self.task}/val_loss",
        val_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    if self.metrics is not None and "val_text_acc" in self.metrics:
        pred_ids = sample(logit_t)
        text_acc = self.val_text_acc.update(pred_ids, answer_token)
        self.log(
            f"{self.task}/val_text_acc",
            text_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    return val_loss

def omni_stage_3_validation(self: pl.LightningModule, batch, batch_idx):
    question_audio_feature = batch['question_audio_feature']
    input_ids = batch['input_ids']
    question_audio_length = batch['question_audio_length']
    answer_token = batch['answer_token']
    answer_token_mask = batch['answer_token_mask']
    answer_snac_tokens = batch['answer_snac_tokens']
    answer_padding_mask = batch['answer_padding_mask']
    
    logit_a, logit_t = self(question_audio_feature, input_ids, whisper_lens=question_audio_length, task='AT')
    
    
    val_text_loss = text_mask_cross_entropy(logit_t, answer_token, answer_token_mask)
    val_audio_loss, _ = audio_mask_cross_entropy(logit_a, answer_snac_tokens, answer_padding_mask)
    
    alpha = 0.5
    val_loss = alpha * val_text_loss + (1 - alpha) * val_audio_loss

    self.log(
        f"{self.task}/val_loss",
        val_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    if self.metrics is not None and "val_text_acc" in self.metrics:
        pred_ids = sample(logit_t)
        text_acc = self.val_text_acc.update(pred_ids, answer_token)
        self.log(
            f"{self.task}/val_text_acc",
            text_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    
    return val_loss
