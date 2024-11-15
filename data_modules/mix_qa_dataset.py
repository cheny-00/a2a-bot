

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
    construct_snac_tokens
)
from pathlib import Path
import typing as tp



class MixQaDataset(Dataset):
    
    def __init__(
        self,
        data_dir: tp.AnyStr,
        whisper_model,
        tokenizer,
        config: tp.Dict,
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
        
    def __getitem__(self, item):
        
        data = self.data.iloc[item]
        
        question_audio = data["question_audio"]["bytes"]
        
        answer_snac = data["answer_snac"]
        answer_snac_tokens, _ = construct_snac_tokens(answer_snac)
        answer_snac_tokens, answer_padding_mask = pad_snac_tokens(self.config, answer_snac_tokens, self.max_seq_length)
        
        
        
        question_tokens = self.tokenizer.encode(data["question"])
        answer_tokens = self.tokenizer.encode(data["answer"])
        answer_token_length = answer_tokens.size(0)
        
        question_audio_feature, question_audio_length = self._get_audio_embeddings(question_audio)
        question_audio_feature = pad_to_max_length(question_audio_feature, self.max_seq_length)
        question_audio_feature = question_audio_feature.squeeze(0)
        
        audio_input_ids = get_audio_template(self.config["token_config"], self.max_seq_length, self.model_layers)

        question_tokens = pad_text_tokens(self.config["token_config"], question_tokens, self.max_seq_length)
        answer_tokens = pad_text_tokens(self.config["token_config"], answer_tokens, self.max_seq_length)
        

        
        features = dict()
        
        features["answer_snac_tokens"] = answer_snac_tokens
        features["answer_padding_mask"] = answer_padding_mask
        
        features["question_audio_feature"] = question_audio_feature
        features["question_audio_length"] = question_audio_length   
        
        # features["question_tokens"] = question_tokens.to(torch.long)
        features["answer_tokens"] = answer_tokens.to(torch.long)
        features["answer_token_length"] = answer_token_length
        
        
        input_ids = audio_input_ids + [question_tokens]
        features["input_ids"] = input_ids

        return features
        
        
