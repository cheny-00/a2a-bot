# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch
from modules.loss_functions import text_mask_cross_entropy
from modules.loss_functions import audio_mask_cross_entropy
from modules.cal_loss_process import cal_stage_1_loss, cal_stage_2_loss, cal_stage_3_loss



def omni_stage_1_validation(self: pl.LightningModule, batch, batch_idx):
    val_loss, _, logit_t = cal_stage_1_loss(self, batch)
    log_validation_metrics(self, val_loss, logit_t, batch, batch_idx)
    return val_loss

def omni_stage_2_validation(self: pl.LightningModule, batch, batch_idx):
    val_loss, _, logit_t = cal_stage_2_loss(self, batch)
    log_validation_metrics(self, val_loss, logit_t, batch, batch_idx)
    return val_loss

def omni_stage_3_validation(self: pl.LightningModule, batch, batch_idx):
    alpha = 0.5
    losses, _, logit_t = cal_stage_3_loss(self, batch, alpha)
    val_loss = losses["loss"]
    log_validation_metrics(self, val_loss, logit_t, batch, batch_idx)
    return val_loss

def log_validation_metrics(self, val_loss, logit_t, batch, batch_idx):
    """Shared validation logging function for all stages
    Args:
        val_loss: validation loss value
        logit_t: text logits from model output
        batch: input batch dictionary
    """
    # Log validation loss
    self.log(
        f"{self.task}/val_loss",
        val_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=len(batch["task"]) if "task" in batch else None,
        sync_dist=getattr(self, "is_distributed", False)
    )

    if self.metrics is not None:
        pred_ids = sample(logit_t)
        assert "target_text_token" in batch, "target_text_token is not in batch"
        tokens_to_check = batch["target_text_token"]
        
        # Log accuracy if metric exists
        if "val_text_acc" in self.metrics:
            text_acc = self.val_text_acc.update(pred_ids, tokens_to_check)
            current_acc = self.val_text_acc.compute()
            self.log(
                f"{self.task}/val_text_acc",
                current_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=getattr(self, "is_distributed", False)
            )
        
        # Log WER if metric exists
        if "val_text_wer" in self.metrics and hasattr(self, "tokenizer"):
            pred_texts = self.tokenizer.batch_decode(pred_ids)
            target_texts = self.tokenizer.batch_decode(tokens_to_check)
            self.val_text_wer.update(pred_texts, target_texts)

            add_text_samples(
                self,
                pred_texts,
                target_texts,
                batch_idx,
                prefix=f"{self.task}/"
            )


def add_text_samples(self, pred_texts, target_texts, batch_idx, prefix=""):
    """Helper function to log text samples to tensorboard
    Args:
        pred_texts (List[str]): List of predicted texts
        target_texts (List[str]): List of target texts
        batch_idx (int): Current batch index
        prefix (str): Prefix for the log name (e.g., "stage1/")
    """
    if self.logger is not None and batch_idx % 100 == 0:  # Log every 100 batches
        num_samples = min(3, len(pred_texts))  # Show up to 3 examples
        text_table = ""
        for i in range(num_samples):
            text_table += f"\n\nExample {i+1}:\n"
            text_table += f"Target:    {target_texts[i]}\n"
            text_table += f"Predicted: {pred_texts[i]}"
        
        self.logger.experiment.add_text(
            f"{prefix}samples",
            text_table,
            self.global_step
        )