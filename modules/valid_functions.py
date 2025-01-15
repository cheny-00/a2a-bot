# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch
from modules.cal_loss_process import cal_stage_1_loss, cal_stage_2_loss, cal_stage_3_loss
from mini_omni.litgpt.generate.base import generate_AT
from utils.utils import log_loss

from rich.console import Console
from rich.table import Table
from rich import box


def omni_stage_1_validation(self: pl.LightningModule, batch, batch_idx):
    losses, _, logit_t = cal_stage_1_loss(self, batch)
    log_validation_metrics(self, losses, logit_t, batch, batch_idx)
    return losses["loss"]

def omni_stage_2_validation(self: pl.LightningModule, batch, batch_idx):
    losses, _, logit_t = cal_stage_2_loss(self, batch)
    log_validation_metrics(self, losses, logit_t, batch, batch_idx)
    return losses["loss"]

def omni_stage_3_validation(self: pl.LightningModule, batch, batch_idx):
    alpha = 0.5
    losses, _, logit_t = cal_stage_3_loss(self, batch, alpha)
    log_validation_metrics(self, losses, logit_t, batch, batch_idx)
    return losses["loss"]

def log_validation_metrics(self, losses, logit_t, batch, batch_idx):
    """Shared validation logging function for all stages
    Args:
        losses: validation losses
        logit_t: text logits from model output
        batch: input batch dictionary
    """
    # Log validation loss
    val_losses = {f"val_{k}": v for k, v in losses.items()}
    log_loss(self, val_losses, len(batch["task"]), self.is_distributed)
    self.log("validation_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch["task"]), sync_dist=self.is_distributed)

    # text metrics
    text_batch_mask = [1 if _task[2] == "T" else 0 for _task in batch["task"]]
    
    if self.metrics is not None and sum(text_batch_mask) > 0:
        text_batch_mask = torch.tensor(text_batch_mask, device=logit_t.device)
        logit_t = logit_t * text_batch_mask.view(-1, 1, 1)
        assert "target_text_token" in batch, "target_text_token is not in batch"
        tokens_to_check = batch["target_text_token"] * text_batch_mask.view(-1, 1)
        
        if "val_text_acc" in self.metrics:
            pred_ids = torch.argmax(logit_t, dim=-1)
            self.val_text_acc.update(pred_ids, tokens_to_check)

        if "val_text_perplexity" in self.metrics:
            self.val_text_perplexity.update(logit_t, tokens_to_check)
            current_perplexity = self.val_text_perplexity.compute()
            self.log(
                f"{self.task}/val_text_perplexity",
                current_perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(batch["task"]) if "task" in batch else None,
                sync_dist=getattr(self, "is_distributed", False)
            )
        
        n_show_text_times = self.hparams.n_show_text_times
        if not hasattr(self, '_show_text_count'):
            self._show_text_count = 0
        if self._show_text_count < n_show_text_times and hasattr(self, "val_text_wer"):
            # device = self.device
            device = "cpu"
            self.model.to(device)    
            self.model.set_kv_cache(batch_size=1, device=device) 
            
            add_text_samples(
                self,
                batch,
                batch_idx,
                prefix=f"{self.task}/",
            )
            self._show_text_count += 1
                
            self.model.clear_kv_cache()
            self.model.to(self.device)


def add_text_samples(self, batch, batch_idx, prefix="", sample_every_n=200):
    """Helper function to log text samples to tensorboard
    Args:
        batch (dict): Input batch dictionary
        batch_idx (int): Current batch index
        prefix (str): Prefix for the log name (e.g., "stage1/")
        sample_every_n (int): Log samples every N batches
    """
    # if batch_idx % sample_every_n != 0:  # Only process every N batches
    #     return
    tasks = batch["task"]
    text_batch_mask = [1 if _task[2:].startswith("T") else 0 for _task in tasks]
    if sum(text_batch_mask) == 0:
        return 
    device = batch["audio_feature"].device
    text_batch_mask = torch.tensor(text_batch_mask, device=device)
    text_batch_id = torch.argmax((text_batch_mask == 1).float(), dim=-1)
    single_sample = {
        "audio_feature": batch["audio_feature"][text_batch_id].unsqueeze(0),
        "input_ids": [_b[text_batch_id].unsqueeze(0) for _b in batch["input_ids"]],
        "audio_length": batch["audio_length"][text_batch_id].unsqueeze(0),
        "task": [batch["task"][text_batch_id]]
    }
    precision = self.config["precision"]
    pred_texts = self.predict_step(single_sample, batch_idx, dataloader_idx=0, display_result=False, precision=precision)
    target_text = batch["target_text"][text_batch_id].strip()
    
    self.val_text_wer.update([pred_texts], [target_text])
    current_wer = self.val_text_wer.compute()
    
    # Create Rich table for console output
    console = Console()
    
    # Create results table
    results_table = Table(
        title="Speech Recognition Results",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    # Add columns
    results_table.add_column("Type", style="cyan", no_wrap=True)
    results_table.add_column("Text", style="green")
    results_table.add_column("Metrics", style="yellow")
    
    # Add rows
    results_table.add_row(
        "Target",
        target_text,
        ""
    )
    results_table.add_row(
        "Predicted",
        pred_texts,
        f"WER: {current_wer:.4f}" if hasattr(self, "val_text_wer") else "N/A"
    )
    
    # Print to console using Rich
    console.print("\n")
    console.print(results_table)
    
    # Create markdown version for TensorBoard
    text_table = "## Speech Recognition Results\n\n"
    
    # Add results table in markdown
    text_table += "| Type | Text | Metrics |\n"
    text_table += "|------|------|---------||\n"
    text_table += f"| Target | `{target_text}` | |\n"
    text_table += f"| Predicted | `{pred_texts}` | WER: {current_wer:.4f} |\n"
    
    # Log to TensorBoard
    self.logger.experiment.add_text(
        f"{prefix}samples",
        text_table,
        self.global_step
    )