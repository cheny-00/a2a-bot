# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

from typing import Dict, AnyStr
from utils.data_utils import get_input_template

from data_modules.base_dataset import MiniOmniBaseDataset

class AsrDataset(MiniOmniBaseDataset):
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
        self.target_text_name = "question"
        task = "A1T1"
        features = self._collate_common_features(data, task)
        input_ids = get_input_template(self.config["token_config"], self.max_seq_length - 3, self.model_layers, speical_token_name="pad_a")

        features['input_ids'] = input_ids
        features['task'] = task

        return features
