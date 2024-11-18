# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import toml
from argparse import ArgumentParser
from typing import Dict
from utils.utils import get_group_parameters


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
        print(f"======= Update {task} from config.toml lr and min_lr =======")
        for k, v in task_config[task].items():
            train_params[k] = v
    return train_params
    
    

def get_args():
    parser = ArgumentParser()

    parser.add_argument("--config_path", type=str, default="config.toml")
    parser.add_argument("--model_name", type=str, default="mini_omni")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoint/")
    parser.add_argument("--log_dir", type=str, default=".logs/")

    # pl_trainer config
    pl_trainer_config = parser.add_argument_group("pl_trainer config", "pl_trainer config")
    # pl_trainer_config.add_argument("--max_epochs", type=int, default=200, dest="pl_trainer_max_epochs")
    pl_trainer_config.add_argument("--precision", type=int, default=32, dest="pl_trainer_precision")
    ## accelerator
    pl_trainer_config.add_argument("--accelerator", type=str, default="auto", dest="pl_trainer_accelerator")
    # pl_trainer_config.add_argument("--strategy", type=str, default="ddp", dest="pl_trainer_strategy")
    pl_trainer_config.add_argument("--devices", type=str, default="auto", dest="pl_trainer_devices")

    # data config
    data_config = parser.add_argument_group("data config", "data config")
    data_config.add_argument("--dataset", type=str, default="asr_dataset", dest="data_config_dataset")
    data_config.add_argument("--num_workers", type=int, default=8, dest="data_config_num_workers")
    data_config.add_argument("--batch_size", type=int, default=16, dest="data_config_batch_size")
    ## load dataset
    data_config.add_argument("--train_data_dir", type=str, default="dataset/train", dest="data_config_train_data_dir")
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

    args = parser.parse_args()

    args.data_params = get_group_parameters(args, "data_config_")
    args.train_params = get_group_parameters(args, "train_config_")
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
    Parses the input to determine the list of device indices or 'auto'.
    
    Args:
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
