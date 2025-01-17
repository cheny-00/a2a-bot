# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import toml
from argparse import ArgumentParser
from typing import Dict
from utils.utils import get_group_parameters


def update_deepspeed_config(deepspeed_config: Dict, train_params: Dict):
    if "optimizer" in deepspeed_config:
        deepspeed_config["optimizer"]["params"]["lr"] = train_params["lr"]
    if "scheduler" in deepspeed_config:
        deepspeed_config["scheduler"]["params"]["warmup_min_lr"] = train_params["min_lr"]
        deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = train_params["lr"]
        deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = train_params["max_steps"] * train_params["warmup_steps"]


def get_token_config(token_config: Dict) -> Dict:
    # 读取基本配置
    text_vocab_size = token_config["text_vocab_size"]
    audio_vocab_size = token_config["audio_vocab_size"]
    text_special_tokens = token_config["text_special_tokens"]
    audio_special_tokens = token_config["audio_special_tokens"]

    # 计算扩展的词汇表大小
    token_config["padded_text_vocab_size"] = text_vocab_size + text_special_tokens
    token_config["padded_audio_vocab_size"] = audio_vocab_size + audio_special_tokens

    # 定义文本相关的特殊标记
    token_config["eot"] = text_vocab_size
    token_config["pad_t"] = text_vocab_size + 1
    token_config["input_t"] = text_vocab_size + 2
    token_config["answer_t"] = text_vocab_size + 3
    token_config["asr"] = text_vocab_size + 4

    # 定义音频相关的特殊标记
    token_config["eoa"] = audio_vocab_size
    token_config["pad_a"] = audio_vocab_size + 1
    token_config["input_a"] = audio_vocab_size + 2
    token_config["answer_a"] = audio_vocab_size + 3
    token_config["split"] = audio_vocab_size + 4

    return token_config

def get_task_config(train_params: Dict, task_config: Dict):
    task = train_params["task"]
    
    if train_params["valid_func_name"] == train_params["train_func_name"] == "omni":
        train_params["train_func_name"] = f"omni_{task}_training"
        train_params["valid_func_name"] = f"omni_{task}_validation"
        print(f"======= Update train_func_name and valid_func_name to omni_{task} =======")
    if task in task_config:
        print(f"======= Update {task}'s parameters from config.toml =======")
        for k, v in task_config[task].items():
            train_params[k] = v
    
    

def get_args():
    parser = ArgumentParser()

    parser.add_argument("--config_path", type=str, default="config.toml")
    parser.add_argument("--model_name", type=str, default="mini_omni")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/")
    parser.add_argument("--log_dir", type=str, default=".logs/")
    parser.add_argument("--reuse_state_dict", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    
    # deepspeed config
    parser.add_argument("--deepspeed_config_path", type=str, default="./deepspeed_config.json")
    parser.add_argument("--deepspeed", action="store_true", default=False)
    
    # pl_trainer config
    pl_trainer_config = parser.add_argument_group("pl_trainer config", "pl_trainer config")
    pl_trainer_config.add_argument("--max_epochs", type=int, default=200, dest="pl_trainer_max_epochs")
    pl_trainer_config.add_argument("--precision", type=str, default=32, dest="pl_trainer_precision")
    pl_trainer_config.add_argument("--fast_dev_run", type=int, default=False, dest="pl_trainer_fast_dev_run")
    
    ## accelerator
    pl_trainer_config.add_argument("--accelerator", type=str, default="auto", dest="pl_trainer_accelerator")
    # pl_trainer_config.add_argument("--strategy", type=str, default="ddp", dest="pl_trainer_strategy")
    pl_trainer_config.add_argument("--devices", type=str, default="auto", dest="pl_trainer_devices")
    ## gradient clip
    pl_trainer_config.add_argument("--gradient_clip_val", type=float, default=1.0, dest="pl_trainer_gradient_clip_val")
    pl_trainer_config.add_argument("--gradient_clip_algorithm", type=str, default="norm", dest="pl_trainer_gradient_clip_algorithm")
    ## gradient accumulation
    pl_trainer_config.add_argument("--accumulate_grad_batches", type=int, default=1, dest="pl_trainer_accumulate_grad_batches")

    # data config
    data_config = parser.add_argument_group("data config", "data config")
    data_config.add_argument("--dataset", type=str, default="asr_dataset", dest="data_config_dataset")
    data_config.add_argument("--num_workers", type=int, default=8, dest="data_config_num_workers")
    data_config.add_argument("--batch_size", type=int, default=16, dest="data_config_batch_size")
    ## load dataset
    data_config.add_argument("--train_data_dir", type=str, dest="data_config_train_data_dir")
    data_config.add_argument("--valid_data_dir", type=str, default="random", dest="data_config_valid_data_dir")

    # trian config
    train_config = parser.add_argument_group("train config", "train config")

    ## task & functions
    train_config.add_argument("--task", type=str, default="stage_1", dest="train_config_task")
    train_config.add_argument("--train_func_name", type=str, default="omni",
                              dest="train_config_train_func_name")
    train_config.add_argument("--valid_func_name", type=str, default="omni",
                              dest="train_config_valid_func_name")
    train_config.add_argument("--loss_fn_name", type=str, default=None, dest="train_config_loss_fn_name")

    ## params 
    train_config.add_argument("--lr", type=float, default=5.5e-4, dest="train_config_lr")
    train_config.add_argument("--optimizer_name", type=str, default="AdamW", dest="train_config_optimizer_name")
    train_config.add_argument("--scheduler_name", type=str, default="CosineAnnealing",
                              dest="train_config_scheduler_name")
    train_config.add_argument("--warmup_iters", type=int, default=3, dest="train_config_warmup_iters")
    train_config.add_argument("--max_iters", type=int, default=20, dest="train_config_max_iters")
    train_config.add_argument("--warmup_steps", type=float, default=0.1, dest="train_config_warmup_steps")
    train_config.add_argument("--max_steps", type=int, default=100000, dest="train_config_max_steps")
    
    train_config.add_argument("--min_lr", type=float, default=1e-6, dest="train_config_min_lr")
    train_config.add_argument("--scheduler_interval", type=str, default="step", dest="train_config_scheduler_interval")
    train_config.add_argument("--n_show_text_times", type=int, default=2, dest="train_config_n_show_text_times")

    # for infer
    infer_config = parser.add_argument_group("infer config", "infer config")
    infer_config.add_argument("--infer", action="store_true", default=False, dest="infer_config_infer")
    infer_config.add_argument("--infer_once", action="store_true", default=False, dest="infer_config_infer_once")
    infer_config.add_argument("--use_state_dict", action="store_true", default=False, dest="infer_config_use_state_dict")
    infer_config.add_argument("--max_seq_length", type=int, default=1200, dest="infer_config_max_seq_length")
    infer_config.add_argument("--temperature", type=float, default=1, dest="infer_config_temperature")
    infer_config.add_argument("--top_k", type=int, default=1, dest="infer_config_top_k")
    infer_config.add_argument("--top_p", type=float, default=1.0, dest="infer_config_top_p")
    infer_config.add_argument("--infer_device", type=str, default="cpu", dest="infer_config_infer_device")
    infer_config.add_argument("--out_dir", type=str, default=None, dest="infer_config_out_dir")
    args = parser.parse_args()

    args.data_params = get_group_parameters(args, "data_config_")
    
    args.train_params = get_group_parameters(args, "train_config_")
    args.infer_params = get_group_parameters(args, "infer_config_")
    
    if "loss_fn_name" not in args.train_params:
        args.train_params["loss_fn_name"] = None
    
        

    args.pl_trainer_params = get_group_parameters(args, "pl_trainer_")
    args.pl_trainer_params["devices"] = parse_devices(args.pl_trainer_params["devices"])

    return args


def get_config(config_path):
    config = toml.load(config_path)

    config["token_config"] = get_token_config(config["token_config"])

    return config


def parse_devices(devices):
    """
        devices (str): A comma-separated string of device indices or 'auto'.
    
    Returns:
        list or str: Returns 'auto' if devices is 'auto', a list of integers representing device indices otherwise.
    
    Raises:
        ValueError: If devices is an unrecognized string format.
    """
    # Ensure the input is a string
    if not isinstance(devices, str):
        raise ValueError(f"Expected string input, got {type(devices).__name__} instead.")

    # Handle 'auto' as a special case
    if devices.strip() == "auto":
        return "auto"

    # Strip any leading/trailing whitespace and check if the input is empty
    devices = devices.strip()
    if not devices:
        raise ValueError("Empty string provided for devices.")

    # Handle the case with commas for multiple devices
    if "," in devices:
        try:
            return [int(num.strip()) for num in devices.split(",") if num.strip()]
        except ValueError:
            raise ValueError(f"Error parsing numbers from '{devices}'.")

    # Handle the case with a single device number
    try:
        return [int(devices)]
    except ValueError:
        raise ValueError(f"Unrecognized devices value: {devices}")
