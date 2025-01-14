# -- coding: utf-8 --
# @Time    :   2024/12/26
# @Author  :   chy


import torch
import pytorch_lightning as pl
import soundfile as sf
from mini_omni.litgpt.generate import base
from functools import partial
from mini_omni.snac_utils.snac_utils import reconscruct_snac, reconstruct_tensors
from utils.logging_utils import display_prediction
from mini_omni.snac_utils.snac_utils import layershift

def om_predict_step(self: pl.LightningModule, batch, batch_idx, dataloader_idx=0, display_result=True, precision=32):
    # device 可能存在问题
    # batch size => 1
    device = "cpu"
    batch = convert_batch_from_fp32_to_precision(batch, precision)
    if "infer_params" in self.config:
        device = self.config["infer_params"]["infer_device"]
    task = batch["task"][0]
    self.model.to(device)
    self.model.set_kv_cache(batch_size=1, device=device) 
    result_tokens = predict_func(self.model, batch, self.config, task)
    self.model.clear_kv_cache()
    self.model.to(self.device)
    result_text = convert_results(result_tokens, task, self.config, self.snac_model, self.tokenizer)
    if display_result:
        display_prediction(task, batch_idx, batch['question'], result_text)
    return result_text

def predict_func(model, batch, config, task):
    if task == "A1T1":
        generate_func = base.generate_ASR
    elif task == "A1T2":
        generate_func = base.generate_AT
    elif task == "T1T2":
        generate_func = base.generate_TT
    elif task == "A1A2":
        generate_func = base.generate_AA
    elif task == "T1A2":
        generate_func = base.generate_TA
    else:
        raise ValueError(f"task {task} is not supported")
    seq_length = batch["text_length"] if task[0] == "T" else batch["audio_length"]
    new_input_ids = convert_input_ids_for_prediction(batch["input_ids"], seq_length)
    result_tokens = generate_func(
        model,
        batch["audio_feature"], new_input_ids, batch["audio_length"], task,
        max_returned_tokens=config["max_seq_length"],
        temperature=config["infer_params"]["temperature"],
        top_k=config["infer_params"]["top_k"],
        top_p=config["infer_params"]["top_p"],
        eos_id_a=config["token_config"]["eot"],
        eos_id_t=config["token_config"]["eot"],
        pad_id_t=config["token_config"]["pad_t"],
        shift=config["token_config"]["padded_text_vocab_size"],
        include_prompt=True,
        generate_text=True,
        tqdm_disable=True,
    )
    
    return result_tokens

def convert_results(result_tokens, task, config, snac_model, tokenizer, step=0):
    
    if task[2] == "A":
        audio_list = reconscruct_snac(result_tokens)
        audio = reconstruct_tensors(audio_list)
        if config["infer_params"]["out_dir"] is None:
            out_dir = f"./output/default/{task}"
        else:
            out_dir = config["infer_params"]["out_dir"]
        with torch.inference_mode():
            audio_hat = snac_model.decode(audio)
        sf.write(
            f"{out_dir}/{step:02d}.wav",
            audio_hat.squeeze().cpu().numpy(),
            24000,
        )  
            
        token_list = result_tokens[-1]
        text = tokenizer.decode(torch.tensor(token_list)).strip()
    elif task[2] == "T":
        text = tokenizer.decode(torch.tensor(result_tokens)).strip()
    else:
        raise ValueError(f"task {task} is not supported")
    
    return text


def convert_input_ids_for_prediction(input_ids, seq_length):
    
    new_input_ids = []
    for _layer in input_ids:
        _layer = _layer.squeeze(0).tolist()
        _new_layer = [_layer[0]] + _layer[1:seq_length+1] + _layer[-2:]
        new_input_ids.append(torch.tensor(_new_layer).unsqueeze(0))
    return new_input_ids


def convert_batch_from_fp32_to_precision(batch, precision: int):
    if precision == 32:
        return batch
    if precision == 16:
        torch_precision = torch.float16
    elif precision == 8:
        torch_precision = torch.float8
    else:
        raise ValueError(f"precision {precision} is not supported")
    
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(torch_precision)
        elif isinstance(batch[key], list):
            batch[key] = [item.to(torch_precision) for item in batch[key]]
    return batch