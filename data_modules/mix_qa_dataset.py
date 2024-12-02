

import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.data_utils import (
    load_audio_from_bytes, 
    get_whisper_embeddings, 
    pad_to_max_length, 
    get_audio_template,
    pad_text_tokens,
    pad_snac_tokens,
    construct_snac_tokens,
    get_target_text_token
)
from pathlib import Path
import typing as tp

from data_modules.base_dataset import MiniOmniBaseDataset


class MixQaDataset(MiniOmniBaseDataset):
    
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
        
        features = self._collate_common_features(data)
        
        answer_snac = data["answer_snac"]
        answer_snac_tokens, _ = construct_snac_tokens(answer_snac)
        answer_snac_tokens, answer_snac_padding_mask = pad_snac_tokens(self.config["token_config"], answer_snac_tokens, self.max_seq_length)
        
        question_tokens = self.tokenizer.encode(data["question"])
        question_tokens = pad_text_tokens(self.token_config, question_tokens, self.max_seq_length)
        
        audio_input_ids = get_audio_template(self.token_config, self.max_seq_length, self.model_layers)
        input_ids = audio_input_ids + [question_tokens]
        features["input_ids"] = input_ids

        features["answer_snac_tokens"] = torch.tensor(answer_snac_tokens, dtype=torch.long)
        features["answer_snac_padding_mask"] = torch.tensor(answer_snac_padding_mask, dtype=torch.bool)
        features["task"] = "A1A2|A1T2"

        return features
        
        
