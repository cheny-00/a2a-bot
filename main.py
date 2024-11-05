# -- coding: utf-8 --
# @Time    :   2024/11/4
# @Author  :   chy


import torch
import tomllib

import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger


from modules.model_interface import ModelInterface
from data_modules.data_interface import DataInterface
from mini_omni.litgpt.config import Config
from params import get_config, get_args
from utils.model_utils import load_models
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from pathlib import Path


def load_callbacks(model_name, version):
    callbacks = list()
    # callbacks.append(
    #     plc.EarlyStopping(
    #         monitor="valid_acc_epoch", mode="max", patience=10, min_delta=0.001
    #     )
    # )

    callbacks.append(
        plc.ModelCheckpoint(
            monitor="valid_acc_epoch",
            # dirpath=f"checkpoints/{model_name}/version_{version}",
            filename="best-{epoch}-{valid_acc_epoch:.3f}-{valid_loss_epoch:.2f}",
            save_top_k=1,
            mode="max",
            save_last=True,
        )
    )

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))
    return callbacks



def main(args):
    
    config = get_config(args.config_path)
    model_config = Config.from_file(Path(args.ckpt_dir) / "model_config.yaml")
    config[args.model_name] = model_config
    
    model = ModelInterface(
        model_name=args.model_name,
        config=config,
        **args.train_params
    )
    
    model_state_dict = lazy_load(args.ckpt_dir + "/lit_model.pth")
    model.model.load_state_dict(model_state_dict)
    
    
    snac_model, whisper_model, tokenizer = load_models(config, args.ckpt_dir)

    
    data_module = DataInterface(
        config=config,
        whisper_model=whisper_model,
        tokenizer=tokenizer,
        **args.data_params,
    )
    
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model_name)
    args.pl_trainer_params["callbacks"] = load_callbacks(args.model_name, logger.version)
    args.pl_trainer_params["logger"] = logger
    
    trainer = Trainer(**args.pl_trainer_params)
    trainer.fit(model, data_module)

    



if __name__ == "__main__":
    args = get_args()
    main(args)
