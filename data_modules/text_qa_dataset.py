# -- coding: utf-8 --
# @Time    :   2024/11/5
# @Author  :   chy

import numpy as np
import torch
from pathlib import Path
from typing import Dict, AnyStr

import pandas as pd
from torch.utils.data import Dataset
from utils.data_utils import (
    pad_text_tokens,
    get_input_template,
)
from data_modules.base_dataset import MiniOmniBaseDataset

class TextQaDataset(MiniOmniBaseDataset):
    
    def __init__(
        self,
        data_dir: AnyStr,
        whisper_model,
        tokenizer,
        config: Dict,
        train=True,
    ):
        super().__init__(data_dir, whisper_model, tokenizer, config, train)
    
    def __getitem__(self, item):
        data = self.data.iloc[item]
        self.target_text_name = "answer"
        task = "T1T2"
        
        features = self._collate_common_features(data, task)
        
        question_tokens = self.tokenizer.encode(data["question"])
        text_length = len(question_tokens)
        question_tokens = pad_text_tokens(self.token_config, question_tokens, self.max_seq_length)
        
        audio_input_ids = get_input_template(self.token_config, self.max_seq_length - 3, self.model_layers, speical_token_name="pad_a")
        input_ids = audio_input_ids + [question_tokens]
        features["input_ids"] = input_ids
        features["text_length"] = text_length
        features["task"] = task
        
        return features
    