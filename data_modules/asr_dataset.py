# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import torch
import whisper
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, AnyStr

from torch.utils.data import Dataset

from utils.data_utils import (
    load_audio_from_bytes,
    get_input_template,
    get_whisper_embeddings,
    pad_to_max_length,
    get_target_text_token
)



class AsrDataset(Dataset):
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

    @staticmethod
    def _load_data(data_dir):
        df = pd.read_parquet(data_dir)
        return df

    @staticmethod
    def _load_audio_from_bytes(audio_bytes):
        # from https://github.com/openai/whisper/blob/main/whisper/audio.py#L62
        audio = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        duration_ms = (len(audio) / 16000) * 1000
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return mel, int(duration_ms / 20) + 1

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _pad_token(tensor, max_length, pad_value=0):
        if tensor.size(0) > max_length:
            return tensor[:max_length]

        # If tensor is shorter than max_length, pad with pad_value
        padded_tensor = torch.full((max_length,), pad_value, dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:tensor.size(0)] = tensor

        return padded_tensor

    def _get_audio_embeddings(self, audio_info):
        mel, leng = load_audio_from_bytes(audio_info)
        audio_feature = get_whisper_embeddings(self.whisper_model, mel)
        return audio_feature, leng


    def __getitem__(self, item):
        data = self.data.iloc[item]

        input_ids = get_input_template(self.config["token_config"], self.max_seq_length - 3, self.model_layers)
        audio_feature, audio_length = self._get_audio_embeddings(data['question_audio']["bytes"])

        audio_feature = pad_to_max_length(audio_feature, self.max_seq_length)
        audio_feature = audio_feature.squeeze(0)

        features = dict()
        text_tokens, text_mask = get_target_text_token(data['question'], self.tokenizer, self.config["token_config"], self.max_seq_length)
        text_length = text_tokens.size(0)
        
        features['text_length'] = text_length
        features['text'] = text_tokens.to(torch.long)
        features['text_mask'] = text_mask
        features['audio_feature'] = audio_feature
        features['input_ids'] = input_ids
        features['audio_length'] = audio_length

        return features
