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
    load_audio_from_bytes, 
    get_whisper_embeddings, 
    pad_to_max_length, 
    get_audio_template,
    pad_text_tokens,
    get_target_text_token
)

class TextQaDataset(Dataset):
    
    def __init__(
        self,
        data_dir: AnyStr,
        whisper_model,
        tokenizer,
        config: Dict,
        train=True,
    ):
        data_dir = Path(data_dir)
        assert data_dir.exists(), f"{data_dir} NOT EXISTS!"

        self.data = self._load_data(data_dir)
        self.whisper_model = whisper_model
        self.tokenizer = tokenizer
        self.max_seq_length = config["max_seq_length"]
        self.config = config
        self.model_layers = config["model_layers"]
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _load_data(data_dir):
        df = pd.read_parquet(data_dir)
        return df
    
    
    def _get_audio_embeddings(self, audio_info):
        mel, leng = load_audio_from_bytes(audio_info)
        audio_feature = get_whisper_embeddings(self.whisper_model, mel)
        return audio_feature, leng

    
    
    def _get_input_position_ids(self, tokens):
        return torch.arange(len(tokens), dtype=torch.long)
    
    def __getitem__(self, item):
        data = self.data.iloc[item]
        
        question_tokens = self.tokenizer.encode(data["question"])
        question_token_length = question_tokens.size(0)
        
        question_audio = data["question_audio"]["bytes"]
        
        question_audio_feature, question_audio_length = self._get_audio_embeddings(question_audio)
        question_audio_feature = pad_to_max_length(question_audio_feature, self.max_seq_length)
        question_audio_feature = question_audio_feature.squeeze(0)
        
        audio_input_ids = get_audio_template(self.config["token_config"], self.max_seq_length, self.model_layers)
        question_tokens = pad_text_tokens(self.config["token_config"], question_tokens, self.max_seq_length)
        
        answer_token, answer_token_mask = get_target_text_token(data["answer"], self.tokenizer, self.config["token_config"], self.max_seq_length)
        
        features = dict()
        features["question_audio_feature"] = question_audio_feature
        features["question_audio_length"] = question_audio_length
        
        # features["question_text"] = question_tokens.to(torch.long)
        features["question_token_length"] = question_token_length
        features["answer_token"] = answer_token
        features["answer_token_mask"] = answer_token_mask
        
        input_ids = audio_input_ids + [question_tokens]
        features["input_ids"] = input_ids
        
        return features
    