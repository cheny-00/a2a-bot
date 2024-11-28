# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy
from typing import Any

import torch
import importlib
import torchmetrics
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs

import modules.train_functions as train_funcs
import modules.valid_functions as valid_funcs
import modules.loss_functions as loss_funcs
from modules import schedulers

from utils.utils import get_mapped_kwargs
from utils.utils import instantiate_torch_optimizer

from mini_omni.litgpt.model import GPT

from functools import partial

import pytorch_lightning as pl

from torchmetrics import WordErrorRate




class ModelInterface(pl.LightningModule):
    def __init__(
            self,
            model_name,
            loss_fn_name,
            lr,
            optimizer_name,
            scheduler_name,
            train_func_name,
            valid_func_name,
            task,
            config,
            snac_model=None,
            tokenizer=None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "config", "tokenizer", "snac_model"])

        self.task = task
        self.config = config
        self.metrics = None
        self.snac_model = snac_model
        self.tokenizer = tokenizer
        self.token_config = self.config["token_config"]
        self.total_text_vocab_size = self.token_config["padded_text_vocab_size"]
        
        self.loss_function = {}
        

        self.model = self.prepare_model()
        self.configure_loss()
        self.configure_metrics()

        self.__set_self_function__("training_step", train_funcs, train_func_name)
        self.__set_self_function__("validation_step", valid_funcs, valid_func_name)


    def __set_self_function__(self, self_func_name, source, func_name):
        func = getattr(source, func_name)
        self_func = partial(func, self)
        self.__setattr__(self_func_name, self_func)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def get_scheduler(self, optimizer, scheduler_name):
        def lr_lambda(it):
            """ This lambda function encapsulates your custom learning rate logic. """
            if self.hparams.scheduler_interval == "epoch":
                max_iters = self.hparams.max_iters
                warmup_iters = self.hparams.warmup_iters
            else:
                max_iters = self.hparams.max_steps
                warmup_iters = self.hparams.warmup_steps
                if warmup_iters < 1:
                    warmup_iters = warmup_iters * max_iters
                warmup_iters = int(warmup_iters)

            return schedulers.litgpt_get_lr(
                learning_rate=self.hparams.lr,
                it=it,
                warmup_iters=warmup_iters,
                max_iters=max_iters,
                min_lr=self.hparams.min_lr
            )

        scheduler_function = lrs.LambdaLR(optimizer, lr_lambda)


        return scheduler_function



    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_name
        optimizer_params = self.config.get(optimizer_name, {})
        print("Optimizer params:", optimizer_params)
        optimizer_params["lr"] = self.hparams.lr
        optimizer = instantiate_torch_optimizer(optimizer_name, self.parameters(), **optimizer_params)
        scheduler = self.get_scheduler(optimizer, self.hparams.scheduler_name)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.hparams.scheduler_interval,  # `interval` specifies when the scheduler is applied.
                "frequency": 1  # `frequency` defines how often the scheduler is applied.
            }
        }

    
    def get_metrics(self, is_train=False):
        prefix = "train" if is_train else "val"
        for metric_name in self.metrics:
            if not metric_name.startswith(prefix):
                continue
            _score = self.__getattr__(metric_name).compute()
            print(f"\n {metric_name}: {_score}")
            self.log(f"{prefix}_{metric_name}_epoch", _score, prog_bar=False)
            self.__getattr__(metric_name).reset()


    def on_train_epoch_end(self):
        
        # Make the Progress Bar leave there
        self.print("")
        if self.metrics is None:
            return
        self.get_metrics(is_train=True)


    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print("")
        if self.metrics is None:
            return
        self.get_metrics(is_train=False)
    
    
    def initialize_metrics(self, metrics):
        for metric_name in metrics:
            if metric_name.endswith("_acc"):
                metric = torchmetrics.Accuracy(task="multiclass", num_classes=self.total_text_vocab_size)
            elif metric_name.endswith("_wer"):
                metric = WordErrorRate()
            self.__setattr__(metric_name, metric)
    
    
    def configure_metrics(self):
        task = self.task
        if task == "stage_1":
            self.metrics = ["val_text_wer"]
            self.initialize_metrics(self.metrics)
        
        

    def configure_loss(self):
        loss_func_name = self.hparams.loss_fn_name
        if loss_func_name is None or loss_func_name == "":
            return
        if isinstance(loss_func_name, str):
            loss_func_name = loss_func_name.split(",")

        for _loss_func_name in loss_func_name:
            _loss_func_name = _loss_func_name.lower()
            if _loss_func_name in self.loss_function:
                continue
            if _loss_func_name in loss_funcs.__dict__:
                self.loss_function[_loss_func_name] = getattr(
                    loss_funcs, _loss_func_name
                )
            else:
                camel_name = "".join(
                    [i.capitalize() for i in _loss_func_name.split("_")]
                )
                loss_class = getattr(loss_funcs, camel_name)
                loss_params = get_mapped_kwargs(loss_class.__init__, self.kwargs)
                loss_func = loss_class(**loss_params).to(self.device)
                self.loss_function[_loss_func_name] = loss_func
    
    def configure_gradient_clipping(self, optimizer: lrs.Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        if "gradient_clip_val" in self.hparams:
            gradient_clip_val = self.hparams.gradient_clip_val
        if "gradient_clip_algorithm" in self.hparams:
            gradient_clip_algorithm = self.hparams.gradient_clip_algorithm
        self.clip_gradients(optimizer, gradient_clip_val, gradient_clip_algorithm)

    @staticmethod
    def _freeze_the_layer(layer):
        """Freeze the parameters of a given layer."""
        for param in layer.parameters():
            param.requires_grad = False


    def _model_state(self, model, task):
        post_adapter_available = self.config[self.hparams.model_name].post_adapter
        if task == "stage_1":
            self._freeze_the_layer(model.transformer.wte)
            self._freeze_the_layer(model.transformer.h)
            self._freeze_the_layer(model.transformer.ln_f)
        elif task == "stage_2":
            self._freeze_the_layer(model.whisper_adapter)
            if post_adapter_available:
                self._freeze_the_layer(model.transformer.post_adapter)
                self._freeze_the_layer(model.transformer.post_audio_adapter_ln)
                self._freeze_the_layer(model.transformer.post_audio_adapter_lm_head)
        # stage 3 no need freeze

    def prepare_model(self):
        model_name = self.hparams.model_name
        model_config = self.config[model_name]
        model = GPT(model_config)
        self._model_state(model, self.task)

        return model




