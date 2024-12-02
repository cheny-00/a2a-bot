# -- coding: utf-8 --
# @Time    :   2024/11/29
# @Author  :   chy


import os
import pandas as pd
from pathlib import Path
from typing import Dict, AnyStr

from torch.utils.data import Dataset

from utils.data_utils import (
    load_audio_from_bytes,
    load_audio_from_path,
    get_whisper_embeddings,
    pad_to_max_length,
    get_target_text_token
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
    
    
    def _collate_common_features(self, data: pd.Series):
        features = dict()
        # audio feature, audio length
        audio_feature, audio_length = self._get_audio_embeddings(data)
        audio_feature = pad_to_max_length(audio_feature, self.max_seq_length)
        audio_feature = audio_feature.squeeze(0)
        
        features["audio_feature"] = audio_feature
        features["audio_length"] = audio_length
        
        # target text token, target text token mask
        target_text_token, target_text_token_mask = get_target_text_token(data[self.target_text_name], self.tokenizer, self.config["token_config"], self.max_seq_length)
        features["target_text_token"] = target_text_token
        features["target_text_token_mask"] = target_text_token_mask
        features["target_text"] = data[self.target_text_name]
        
        return features
