# -- coding: utf-8 --
# @Time    :   2024/11/29
# @Author  :   chy


import os
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, AnyStr, Union

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
        self._map_config(config)
    
    def _map_config(self, config: Dict):
        self.config = config
        self.max_seq_length = config["max_seq_length"]
        self.model_layers = config["model_layers"]
        self.token_config = config["token_config"]
        self.model_name = config["model_name"]
    
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _load_data(data_dir) -> pd.DataFrame:
        df = pd.read_parquet(data_dir)
        return df
    
    
    @classmethod
    def from_input(
        cls,
        input_data: Union[Dict, pd.DataFrame],
        whisper_model,
        tokenizer,
        config: Dict,
        task: str
    ):
        """_summary_

        Args:
            input_data (Union[Dict, pd.DataFrame]): include
            {
                question: text,
                question_audio: audio bytes,
            }
            whisper_model: whisper model
            tokenizer: tokenizer
            config: config
            task (str): task
        """
        instance = cls.__new__(cls)
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame(input_data)
        instance.data = input_data
        instance.task = task
        instance.whisper_model = whisper_model
        instance.tokenizer = tokenizer
        instance._map_config(config)
        instance._collate_common_features = instance._collate_source_features.__get__(instance, cls)
        return instance
        
    
    
    
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
    
    
    def _collate_source_features(self, data: pd.Series, task: str):
        features = dict()
        if task.startswith("A"):
            audio_feature, audio_length = self._get_audio_embeddings(data)
            audio_feature = pad_to_max_length(audio_feature, self.max_seq_length, self.token_config["pad_a"])
            audio_feature = audio_feature.squeeze(0)
        elif task.startswith("T"):
            features["question"] = data["question"]
            audio_feature = torch.zeros((self.max_seq_length, self.config[self.model_name].whisper_adapter_dim))
            audio_length = 0
        else:
            raise ValueError(f"Invalid task: {task}")
        features["audio_feature"] = audio_feature.to("cpu")
        features["audio_length"] = audio_length
        return features
        
    
    def _collate_common_features(self, data: pd.Series, task: str):
        features = dict()
        # audio feature, audio length
        source_features = self._collate_source_features(data, task)
        features.update(source_features)
        
        target_text_token, target_text_token_mask = get_target_text_token(data[self.target_text_name], self.tokenizer, self.config["token_config"], self.max_seq_length)
        answer_snac = data["answer_snac"]
        answer_snac_tokens, _ = construct_snac_tokens(answer_snac)
        answer_snac_tokens, answer_snac_padding_mask = pad_snac_tokens(self.config["token_config"], answer_snac_tokens, self.max_seq_length)
        answer_snac_tokens = torch.tensor(answer_snac_tokens, dtype=torch.long)
        answer_snac_padding_mask = torch.tensor(answer_snac_padding_mask, dtype=torch.bool)
        
        # target text token, target text token mask
        if task[2] == "T":
            features["target_text_token"] = target_text_token.to("cpu")
            features["target_text_token_mask"] = target_text_token_mask.to("cpu")
            features["target_text"] = data[self.target_text_name]
            
            features["target_snac_token"] = torch.ones_like(answer_snac_tokens) * self.token_config["pad_a"]
            features["target_snac_token_mask"] = torch.zeros_like(answer_snac_padding_mask)
            
        elif task[2] == "A":
            features["target_text_token"] = torch.ones_like(target_text_token) * self.token_config["pad_t"]
            features["target_text_token_mask"] = torch.zeros_like(target_text_token_mask)
            features["target_text"] = data[self.target_text_name]
            
            features["target_snac_token"] = answer_snac_tokens
            features["target_snac_token_mask"] = answer_snac_padding_mask
        else:
            raise ValueError(f"Invalid task: {task}")
        
        
        
        return features
