# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch
from modules.loss_functions import text_mask_cross_entropy
from modules.loss_functions import audio_mask_cross_entropy
from modules.cal_loss_process import cal_stage_1_loss

def omni_stage_1_validation(self: pl.LightningModule, batch, batch_idx):
    val_loss, _, logit_t = cal_stage_1_loss(self, batch)
    self.log(
        f"{self.task}/val_loss",
        val_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=len(batch["task"]),
        sync_dist=self.is_distributed
    )
    if self.metrics is not None:
        pred_ids = sample(logit_t)
        target_text_token = batch["target_text_token"]
        if "val_text_acc" in self.metrics:
            text_acc = self.val_text_acc.update(pred_ids, target_text_token)
            current_acc = self.val_text_acc.compute()
            self.log(
                f"{self.task}/val_text_acc",
                current_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.is_distributed
            )
        if "val_text_wer" in self.metrics:
            pred_texts = self.tokenizer.batch_decode(pred_ids)
            target_texts = self.tokenizer.batch_decode(target_text_token)
            self.val_text_wer.update(pred_texts, target_texts)
            current_wer = self.val_text_wer.compute()
            self.log(
                f"{self.task}/val_wer",
                current_wer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(batch["task"]),
                sync_dist=self.is_distributed
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
