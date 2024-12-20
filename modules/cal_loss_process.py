
import torch
import pytorch_lightning as pl
from modules.loss_functions import text_mask_cross_entropy, audio_mask_cross_entropy

def cal_mini_omni_logits(self: pl.LightningModule, batch):
    audio_feature = batch['audio_feature']
    input_ids = batch['input_ids']
    audio_length = batch['audio_length']
    task = batch['task']
    
    # logit_a shape: (batch_size, 7, seq_length, vocab_size)
    # logit_t shape: (batch_size, seq_length, vocab_size)   
    logit_a, logit_t = self(audio_feature, input_ids, whisper_lens=audio_length, task=task)
    return logit_a, logit_t
    
def cal_text_loss(batch, logit_t, pad_t=-100):
    target_text_token = batch['target_text_token']
    target_text_token_mask = batch['target_text_token_mask']
    
    shifted_target_text_token = target_text_token[..., 1:].contiguous()
    shifted_logits = logit_t[..., :-1, :].contiguous() 
    shifted_target_text_token_mask = target_text_token_mask[..., 1:].contiguous()
    loss_text = text_mask_cross_entropy(shifted_logits, shifted_target_text_token, shifted_target_text_token_mask, ignore_index=pad_t)
    return loss_text

def cal_audio_loss(batch, logit_a, pad_t=-100):
    answer_snac_tokens = batch['target_snac_token']
    answer_snac_padding_mask = batch['target_snac_token_mask']
        
    loss_audio, _ = audio_mask_cross_entropy(logit_a, answer_snac_tokens, answer_snac_padding_mask, ignore_index=pad_t)
    return loss_audio




def cal_stage_1_loss(self: pl.LightningModule, batch):
    logit_a, logit_t = cal_mini_omni_logits(self, batch)
    loss_text = cal_text_loss(batch, logit_t, self.pad_t)
    losses = {"text_loss": loss_text, "loss": loss_text}
    return losses, logit_a, logit_t


def cal_stage_2_loss(self: pl.LightningModule, batch):
    logit_a, logit_t = cal_mini_omni_logits(self, batch)
    loss_text = cal_text_loss(batch, logit_t, self.pad_t)
    losses = {"text_loss": loss_text, "loss": loss_text}
    return losses, logit_a, logit_t


def cal_stage_3_loss(self: pl.LightningModule, batch, alpha=0.5):
    logit_a, logit_t = cal_mini_omni_logits(self, batch)
    loss_text = cal_text_loss(batch, logit_t, self.pad_t)
    loss_audio = cal_audio_loss(batch, logit_a, self.pad_a)
    
    loss = alpha * loss_text + (1 - alpha) * loss_audio
    losses = {"text_loss": loss_text, "audio_loss": loss_audio, "loss": loss}
    
    return losses, logit_a, logit_t