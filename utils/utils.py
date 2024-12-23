# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import os
import re
import inspect
import pytorch_lightning as pl
from lightning.pytorch.cli import instantiate_class
from pathlib import Path

def get_mapped_kwargs(func, kwargs):
    sig = inspect.signature(func)
    class_args = [
        param.name for param in sig.parameters.values() if param.name != "self"
    ]
    mapped_args = {arg: kwargs[arg] for arg in class_args if arg in kwargs}

    return mapped_args


def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    # From: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/utils.py#L561
    # Special care taken where some optimizers do not have some parameters referenced in some of the code, for example "fused" in the pretrain.py script:
    #   bnb.optim.AdamW8bit
    #   grokadamw.GrokAdamW
    #   torch.optim.RMSprop

    if isinstance(optimizer, str):
        if "." in optimizer:
            class_module, class_name = optimizer.rsplit(".", 1)
        else:
            class_module, class_name = "torch.optim", optimizer

        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}
        optimizer = optimizer_cls(model_parameters, **kwargs)
    elif isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        class_module, class_name = optimizer["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}

        optimizer["init_args"].update(kwargs)
        optimizer = instantiate_class(model_parameters, optimizer)
    else:
        raise ValueError(f'Unrecognized "optimizer" value: {optimizer}')

    return optimizer

def get_group_parameters(args, group_name):
    grouped_params = {}
    pop_list = list()
    for key, value in vars(args).items():
        if not key.startswith(group_name) or value is None:
            continue
        grouped_params[key[len(group_name) :]] = value
        pop_list.append(key)
    for key in pop_list:
        args.__dict__.pop(key)
    return grouped_params


def log_loss(self: pl.LightningModule, losses, batch_size, sync_dist):
    for key, value in losses.items():
        self.log(
            f"{self.task}/{key}",
            value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=sync_dist
        )
        

def fix_version_path(args, checkpoint_path):
    if checkpoint_path is None or not checkpoint_path.startswith("version"):
        return checkpoint_path
    
    current_path = Path(os.getcwd())
    checkpoint_path = current_path / args.log_dir / args.model_name / checkpoint_path / "checkpoints"
    
    # Confirm the existence of the directory
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} does not exist.")
    
    last_ckpt = checkpoint_path / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    checkpoint_files = list(checkpoint_path.glob("best-*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path} following the filename pattern.")
    
    def parse_validation_loss(file):
        match = re.search(r"validation_loss=([0-9]+\.[0-9]+)", file.name)
        if match:
            return float(match.group(1))
        return float("inf")

    best_ckpt = min(checkpoint_files, key=parse_validation_loss)

    return best_ckpt