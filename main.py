# -- coding: utf-8 --
# @Time    :   2024/11/4
# @Author  :   chy

import os
import json
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy


from modules.model_interface import ModelInterface
from data_modules.data_interface import DataInterface
from mini_omni.litgpt.config import Config
from params import get_config, get_args, get_task_config, update_deepspeed_config
from utils.model_utils import load_models
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from pathlib import Path


def load_callbacks(model_name, version, task):
    callbacks = list()
    # callbacks.append(
    #     plc.EarlyStopping(
    #         monitor="valid_acc_epoch", mode="max", patience=10, min_delta=0.001
    #     )
    # )

    callbacks.append(
        plc.ModelCheckpoint(
            monitor=f"{task}/val_loss",
            # dirpath=f"checkpoints/{model_name}/version_{version}",
            filename="best-{epoch}-{valid_loss_epoch:.2f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        )
    )

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))
    return callbacks



def main(args):
    
    config = get_config(args.config_path)
    model_config = Config.from_file(Path(args.ckpt_dir) / "model_config.yaml")
    print("model_config", model_config)
    config[args.model_name] = model_config
    get_task_config(args.train_params, config)
    task = args.train_params["task"]
    if task in config and "dataset" in config[task]:
        args.data_params["dataset"] = config[task]["dataset"]
        print(f"========= Use dataset: {args.data_params['dataset']} for task: {task} =========")
    
    snac_model, whisper_model, tokenizer = load_models(config, args.ckpt_dir)
    
    model = ModelInterface(
        model_name=args.model_name,
        config=config,
        snac_model=snac_model,
        tokenizer=tokenizer,
        **args.train_params
    )
    
    strategy = None
    if args.deepspeed and os.path.exists(args.deepspeed_config_path):
        with open(args.deepspeed_config_path, "r") as f:
            deepspeed_config = json.load(f) 
        update_deepspeed_config(deepspeed_config, args.train_params)
        strategy = DeepSpeedStrategy(config=deepspeed_config)
    
    model_state_dict = lazy_load(args.ckpt_dir + "/lit_model.pth")
    model.model.load_state_dict(model_state_dict)
    
    

    
    data_module = DataInterface(
        config=config,
        whisper_model=whisper_model,
        tokenizer=tokenizer,
        strategy=strategy,
        **args.data_params,
    )
    
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model_name)
    args.pl_trainer_params["callbacks"] = load_callbacks(args.model_name, logger.version, task)
    args.pl_trainer_params["logger"] = logger
    
    trainer = Trainer(**args.pl_trainer_params)
    trainer.fit(model, data_module)

    



if __name__ == "__main__":
    args = get_args()
    main(args)
