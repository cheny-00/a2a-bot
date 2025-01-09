# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


from . import asr_dataset
from . import text_qa_dataset
from . import voice_qa_dataset


def dataset_selector(task: str):
    if task == "A1T1":
        return asr_dataset.AsrDataset
    elif task == "A1T2":
        return voice_qa_dataset.VoiceQaDataset
    elif task == "T1T2":
        return text_qa_dataset.TextQaDataset
    else:
        raise ValueError(f"Invalid task: {task}")
