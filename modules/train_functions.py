# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import torch
from mini_omni.litgpt.generate.base import sample
import pytorch_lightning as pl
from modules.loss_functions import text_mask_cross_entropy, audio_mask_cross_entropy


def omni_stage_1_training(self: pl.LightningModule, batch, batch_idx):
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
    loss = loss_text

    self.log(
        f"{self.task}/train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )

    # result_tokens = torch.argmax(torch.softmax(logit_t, dim=-1), dim=-1)
    # for _i, _result_tokens in enumerate(result_tokens):
    #     result_text = self.tokenizer.decode(_result_tokens)
    if self.metrics is not None and "train_text_acc" in self.metrics:
        pred_ids = torch.argmax(logit_t, dim=-1)
        text_acc = self.train_text_acc.update(pred_ids, target_token)
        self.log(
            f"{self.task}/train_text_acc",
            text_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    return loss


def omni_stage_2_training(self: pl.LightningModule, batch, batch_idx):
    question_audio_feature = batch['question_audio_feature']
    question_audio_length = batch['question_audio_length']
    input_ids = batch['input_ids']
    answer_token = batch['answer_token']
    answer_token_mask = batch['answer_token_mask']
    
    _, logit_t = self(question_audio_feature, input_ids, whisper_lens=question_audio_length, task='AT')
    # logit_a = torch.stack(logit_a)
    # TODO: add loss function
    # logit_a shape: (batch_size, 7, seq_length, vocab_size)
    # logit_t shape: (batch_size, seq_length, vocab_size)   
    text_loss = text_mask_cross_entropy(logit_t, answer_token, answer_token_mask)
    loss = text_loss
    self.log(
        f"{self.task}/train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )

    return loss
    
def omni_stage_3_training(self: pl.LightningModule, batch, batch_idx):
    question_audio_feature = batch['question_audio_feature']
    question_audio_length = batch['question_audio_length']
    input_ids = batch['input_ids']
    answer_token = batch['answer_token']
    answer_token_mask = batch['answer_token_mask']
    
    answer_snac_tokens = batch['answer_snac_tokens']
    answer_padding_mask = batch['answer_padding_mask']
    
    logit_a, logit_t = self(question_audio_feature, input_ids, whisper_lens=question_audio_length, task='AT')
    text_loss = text_mask_cross_entropy(logit_t, answer_token, answer_token_mask)
    audio_loss, _ = audio_mask_cross_entropy(logit_a, answer_snac_tokens, answer_padding_mask)
    
    alpha = 0.5
    loss = alpha * text_loss + (1 - alpha) * audio_loss
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
