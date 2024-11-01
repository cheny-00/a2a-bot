# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import toml

from typing import Dict


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




def get_config(config_path):
    config = toml.load(config_path)

    config["token_config"] = get_config(config["token_config"])

    return config

