# -- coding: utf-8 --
# @Time    :   2024/11/29
# @Author  :   chy


import os
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, AnyStr

from torch.utils.data import Dataset

from utils.data_utils import (
    load_audio_from_bytes,
    load_audio_from_path,
    get_whisper_embeddings,
    pad_to_max_length,
    get_target_text_token,
    construct_snac_tokens,
    pad_snac_tokens
)

class MiniOmniBaseDataset(Dataset):
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
        self.data_dir = data_dir

        self.data = self._load_data(data_dir)
        self.whisper_model = whisper_model
        self.tokenizer = tokenizer
        self.max_seq_length = config["max_seq_length"]
        self.config = config
        self.model_layers = config["model_layers"]
        self.token_config = config["token_config"]
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _load_data(data_dir) -> pd.DataFrame:
        df = pd.read_parquet(data_dir)
        return df
    
    
    
    def _get_audio_embeddings(self, data: pd.Series):
        if "question_audio" in data:
            audio_feature, audio_length = load_audio_from_bytes(data['question_audio']["bytes"])
        elif "question_audio_path" in data:
            q_audio_path = data['question_audio_path']
            audio_path = q_audio_path if os.path.exists(q_audio_path) else self.data_dir.parent / q_audio_path
            audio_feature, audio_length = load_audio_from_path(audio_path)
        else:
            raise ValueError("No question audio or question audio path found in data")
        audio_feature = get_whisper_embeddings(self.whisper_model, audio_feature, audio_length)
        return audio_feature, audio_length
    
    
    def _collate_common_features(self, data: pd.Series, task: str):
        features = dict()
        # audio feature, audio length
        if task.startswith("A"):
            audio_feature, audio_length = self._get_audio_embeddings(data)
            audio_feature = pad_to_max_length(audio_feature, self.max_seq_length, self.token_config["pad_a"])
            audio_feature = audio_feature.squeeze(0)
        elif task.startswith("T"):
            audio_feature = None
            audio_length = None
        else:
            raise ValueError(f"Invalid task: {task}")
        
        features["audio_feature"] = audio_feature
        features["audio_length"] = audio_length
        
        # target text token, target text token mask
        if task.startswith("T"):
            target_text_token, target_text_token_mask = get_target_text_token(data[self.target_text_name], self.tokenizer, self.config["token_config"], self.max_seq_length)
            features["target_token"] = target_text_token
            features["target_token_mask"] = target_text_token_mask
            features["target"] = data[self.target_text_name]
        elif task.startswith("A"):
            answer_snac = data["answer_snac"]
            answer_snac_tokens, _ = construct_snac_tokens(answer_snac)
            answer_snac_tokens, answer_snac_padding_mask = pad_snac_tokens(self.token_config, answer_snac_tokens, self.max_seq_length)
            features["target_token"] = answer_snac_tokens
            features["target_token_mask"] = answer_snac_padding_mask
            features["target"] = data["answer_snac"]
        else:
            raise ValueError(f"Invalid task: {task}")
        
        
        
        return features
