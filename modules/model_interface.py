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
            train_config,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "config"])

        self.task = task

        self.model = self.prepare_model()

        self.__set_self_function__("training_step", train_funcs, train_func_name)
        self.__set_self_function__("validation_step", valid_funcs, valid_func_name)


    def __set_self_function__(self, self_func_name, source, func_name):
        func = getattr(source, func_name)
        self_func = partial(func, self)
        self.__setattr__(self_func_name, self_func)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def get_scheduler(self, scheduler_name):
        def lr_lambda(epoch):
            """ This lambda function encapsulates your custom learning rate logic. """
            return schedulers.litgpt_get_lr(
                learning_rate=self.hparams.learning_rate,
                it=epoch,
                warmup_iters=self.hparams.warmup_iters,
                max_iters=self.hparams.max_iters,
                min_lr=self.hparams.min_lr
            )

        scheduler_function = lr_lambda


        return scheduler_function



    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer_name = self.hparams.optimizer_name
        optimizer_params = self.config[optimizer_name]
        optimizer = instantiate_torch_optimizer(optimizer_name, self.parameters(), **optimizer_params)
        scheduler = self.get_scheduler(self.hparams.scheduler_name)


        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # `interval` specifies when the scheduler is applied.
                "frequency": 1  # `frequency` defines how often the scheduler is applied.
            }
        }



    def on_train_epoch_end(self):
        ...

    def configure_loss(self):
        loss_func_name = self.hparams.loss_fn_name
        if isinstance(loss_func_name, str):
            loss_func_name = [loss_func_name]

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

    @staticmethod
    def _freeze_the_layer(layer):
        """Freeze the parameters of a given layer."""
        for param in layer.parameters():
            param.requires_grad = False


    def _model_state(self, model, task):
        if task == "stage_1":
            self._freeze_the_layer(model.transformer.wta)
            self._freeze_the_layer(model.transformer.h)
            self._freeze_the_layer(model.transformer.ln_f)
        elif task == "stage_2":
            self._freeze_the_layer(model.whisper_adapter)
            self._freeze_the_layer(model.transformer.post_adapter)
            self._freeze_the_layer(model.transformer.post_audio_adapter_ln)
            self._freeze_the_layer(model.transformer.post_audio_adapter_lm_head)
        # stage 3 no need freeze

    def prepare_model(self):
        model_name = self.hparams.model_name
        model_config = self.config[model_name]
        model = GPT(model_config)
        self._model_state(model, self.hparams.task)

        return model






