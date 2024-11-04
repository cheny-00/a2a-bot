# -- coding: utf-8 --
# @Time    :   2024/11/04
# @Author  :   chy

import whisper
from snac import SNAC
from mini_omni.litgpt.tokenizer import Tokenizer


def load_models(config, ckpt_dir):
    # snac_model = SNAC.from_pretrained(config["snac_model_name"])
    snac_model = None
    whisper_model = whisper.load_model(config["whisper_model_name"])
    tokenizer = Tokenizer(ckpt_dir)
    
    return snac_model, whisper_model, tokenizer