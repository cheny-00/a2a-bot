# -- coding: utf-8 --
# @Time    :   2024/11/1
# @Author  :   chy

import pytorch_lightning as pl
from mini_omni.litgpt.generate.base import sample
import torch
from modules.loss_functions import text_mask_cross_entropy
from modules.loss_functions import audio_mask_cross_entropy
from modules.cal_loss_process import cal_stage_1_loss, cal_stage_2_loss, cal_stage_3_loss
from mini_omni.litgpt.generate.base import generate_AT, generate_ASR

from rich.console import Console
from rich.table import Table
from rich import box


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

        # Log Perplexity if metric exists
        if "val_text_perplexity" in self.metrics:
            self.val_text_perplexity.update(logit_t, tokens_to_check)
            current_perplexity = self.val_text_perplexity.compute()
            self.log(
                f"{self.task}/val_text_perplexity",
                current_perplexity,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=getattr(self, "is_distributed", False)
            )
        
        n_show_text_times = 5
        if not hasattr(self, '_show_text_count'):
            self._show_text_count = 0
        if self._show_text_count < n_show_text_times and hasattr(self, "val_text_wer"):
            device = self.device
            if str(self.device).startswith("mps"):
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
            if str(self.device).startswith("mps"):
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
        
    single_sample = {
        "audio_feature": batch["audio_feature"][:1],
        "input_ids": [_b[:1] for _b in batch["input_ids"]],
        "audio_length": batch["audio_length"][:1],
        "task": batch["task"][:1]
    }
    pred_tokens = generate_AT(self.model, single_sample["audio_feature"], 
                           single_sample["input_ids"], single_sample["audio_length"], 
                           single_sample["task"], max_returned_tokens=2048,
                           generate_text=True,
                           include_prompt=False,
                           temperature=0.9,
                           eos_id_t=self.token_config["eot"],
                           eos_id_a=self.token_config["eoa"],
                           pad_id_t=self.token_config["pad_t"],
                           shift=self.token_config["padded_text_vocab_size"],
                           top_k=1,
                           tqdm_disable=True)
    pred_texts = self.tokenizer.decode(torch.tensor(pred_tokens)).strip()
    target_text = self.tokenizer.decode(torch.tensor(batch["target_text_token"][0])).strip()
    
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