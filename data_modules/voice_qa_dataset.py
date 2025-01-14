

import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.data_utils import (
    pad_text_tokens,
    get_input_template
)
from pathlib import Path
import typing as tp

from data_modules.base_dataset import MiniOmniBaseDataset


class VoiceQaDataset(MiniOmniBaseDataset):
    
    def __init__(
        self,
        data_dir: tp.AnyStr,
        whisper_model,
        tokenizer,
        config: tp.Dict,
        train=True,
    ):
        super().__init__(data_dir, whisper_model, tokenizer, config, train)
        
        
    def __getitem__(self, item):
        
        data = self.data.iloc[item]
        self.target_text_name = "answer"
        task = "A1A2"
        
        features = self._collate_common_features(data, task)
        
        question_tokens = self.tokenizer.encode(data["question"])
        question_tokens = pad_text_tokens(self.token_config, question_tokens, self.max_seq_length)
        
        input_ids = get_input_template(self.config["token_config"], self.max_seq_length - 3, self.model_layers, speical_token_name="answer_a")
        features["input_ids"] = input_ids

        features["task"] = task
        features["text_length"] = 0
        
        return features
        
        
