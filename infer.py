# -- coding: utf-8 --
# @Time    :   2025/01/07
# @Author  :   chy

import pytorch_lightning as pl

from data_modules import dataset_selector
from torch.utils.data import DataLoader

def infer_once(trainer: pl.Trainer, model: pl.LightningModule, whisper_model, tokenizer, config):
    user_text = "Test Test Test"
    user_audio = None 
    task = "T1T2"
    input_data = {
        "question": user_text,
        "question_audio": user_audio,
    }
    input_data = [input_data]
    
    predict_dataset = dataset_selector(task).from_input(input_data, whisper_model, tokenizer, config, task)
    predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
    
    
    trainer.predict(model, predict_dataloader)
    
    


# if __name__ == "__main__":
    
#     from params import get_args
#     args = get_args()
#     infer_once(args)
