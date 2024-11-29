
import torch
from mini_omni.litgpt.generate.base import sample
import pytorch_lightning as pl
from modules.loss_functions import text_mask_cross_entropy, audio_mask_cross_entropy

def cal_stage_1_loss(self: pl.LightningModule, batch):
    audio_feature = batch['audio_feature']
    input_ids = batch['input_ids']
    audio_length = batch['audio_length']
    target_text_token = batch['target_text_token']
    target_text_token_mask = batch['target_text_token_mask']
    task = batch['task']
    pad_t = self.token_config["pad_t"]
    
    logit_a, logit_t = self(audio_feature, input_ids, whisper_lens=audio_length, task=task)
    
    shifted_target_text_token = target_text_token[..., 1:].contiguous()
    shifted_logits = logit_t[..., :-1, :].contiguous() 
    shifted_target_text_token_mask = target_text_token_mask[..., 1:].contiguous()
    loss_text = text_mask_cross_entropy(shifted_logits, shifted_target_text_token, shifted_target_text_token_mask, ignore_index=pad_t)
    loss = loss_text

    return loss, logit_a, logit_t