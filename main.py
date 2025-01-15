# -- coding: utf-8 --
# @Time    :   2024/11/4
# @Author  :   chy

import os
import json
import torch
from torch.multiprocessing import set_start_method
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
try:
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint # type: ignore
except:
    print("Deepspeed is not installed, skipping zero checkpoint loading")


from modules.model_interface import ModelInterface
from data_modules.data_interface import DataInterface
from mini_omni.litgpt.config import Config
from params import get_config, get_args, get_task_config, update_deepspeed_config
from utils.model_utils import load_models
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from pathlib import Path
from utils.logging_utils import LightRichProgressBarTheme
from utils.utils import fix_version_path
from infer import infer_once

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
            filename="best-{epoch}-{validation_loss:.3f}",
            save_top_k=1,
            mode="min",
            save_last=True,
        )
    )
    callbacks.append(plc.LearningRateMonitor(logging_interval="step"))
    callbacks.append(plc.RichProgressBar(theme=LightRichProgressBarTheme()))
    callbacks.append(plc.RichModelSummary())

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def update_params(args, config):
    """
    Update the parameters with the specific task config
    """
    task = args.train_params["task"]
    if task not in config:
        return task
    if "datasets" in config[task]:
        args.data_params["datasets"] = config[task]["datasets"]
        print(f"========= Use datasets: {args.data_params['datasets']} for task: {task} =========")
    if "train_data_dir" not in args.data_params:
        if "train_data_dir" not in config[task]:
            raise ValueError(f"train_data_dir not found in config for task: {task}")
        args.data_params["train_data_dir"] = config[task]["train_data_dir"]
        print(f"========= Use train_data_dir: {args.data_params['train_data_dir']} for task: {task} =========") 
        
    if "valid_data_dir" not in args.data_params:
        if "valid_data_dir" not in config[task]:
            raise ValueError(f"valid_data_dir not found in config for task: {task}")
        args.data_params["valid_data_dir"] = config[task]["valid_data_dir"]
        print(f"========= Use valid_data_dir: {args.data_params['valid_data_dir']} for task: {task} =========") 
    return task



def main(args):
    
    config = get_config(args.config_path)
    model_config = Config.from_file(Path(args.ckpt_dir) / "model_config.yaml")
    print("model_config", model_config)
    config[args.model_name] = model_config
    config["model_name"] = args.model_name
    config["infer_params"] = args.infer_params
    config["precision"] = args.pl_trainer_params["precision"]
    get_task_config(args.train_params, config)
    task = update_params(args, config)
    
    snac_model, whisper_model, tokenizer = load_models(config, args.ckpt_dir)
    # TODO: fix resume from checkpoint(args.resume_from_checkpoint)
    model = ModelInterface(
        model_name=args.model_name,
        config=config,
        snac_model=snac_model,
        tokenizer=tokenizer,
        **args.train_params
    )
    
    strategy = "auto"
    # if args.deepspeed and os.path.exists(args.deepspeed_config_path):
    #     print(f"========= Use deepspeed config: {args.deepspeed_config_path} =========")
    #     with open(args.deepspeed_config_path, "r") as f:
    #         deepspeed_config = json.load(f) 
    #     update_deepspeed_config(deepspeed_config, args.train_params)
    #     strategy = DeepSpeedStrategy(config=deepspeed_config)
    if args.deepspeed:
        print("========= Use deepspeed strategy: deepspeed_stage_2 =========")
        args.pl_trainer_params.pop("gradient_clip_val", None)
        args.pl_trainer_params.pop("gradient_clip_algorithm", None)
        strategy = "deepspeed_stage_2"
    
    
    if args.resume_from_checkpoint:
        print(f"========= Resume from checkpoint: {args.resume_from_checkpoint} =========")
        print(f"========= Global step: {model.global_step} =========")
    elif args.reuse_state_dict:
        checkpoint_path = args.reuse_state_dict
        checkpoint_path = fix_version_path(args, checkpoint_path)
        if checkpoint_path.is_file():
            model_state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        else:
            model_state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
        model.load_state_dict(model_state_dict)
        print(f"========= Reuse state dict from: {checkpoint_path} =========")
    else:
        model_state_dict = lazy_load(args.ckpt_dir + "/lit_model.pth")
        model.model.load_state_dict(model_state_dict)
    
    
    data_module = DataInterface(
        config=config,
        whisper_model=whisper_model,
        tokenizer=tokenizer,
        **args.data_params,
    )
    
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.model_name)
    if not (args.infer_params["infer_once"] or args.infer_params["infer"]) and not args.debug:
        args.pl_trainer_params["callbacks"] = load_callbacks(args.model_name, logger.version, task)
        args.pl_trainer_params["logger"] = logger
    
    trainer = Trainer(**args.pl_trainer_params, strategy=strategy)
    assert int(args.infer_params["infer"]) + int(args.infer_params["infer_once"]) < 2, "Only one of infer or single_infer can be True"

    if args.infer_params["infer"]:
        print("========= Infer =========")
        trainer.predict(model, data_module)
    elif args.infer_params["infer_once"]:
        print("========= Infer once =========")
        infer_once(trainer, model, whisper_model, tokenizer, config)
    else:
        print("========= Train =========")
        trainer.fit(model, data_module)

    



if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available(): 
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass
    main(args)
