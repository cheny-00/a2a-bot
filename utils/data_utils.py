# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import torch
import whisper
import ffmpeg
import numpy as np
import typing as tp
import torch.nn.functional as F
from pathlib import Path

from mini_omni.snac_utils.snac_utils import layershift


def _process_audio(audio: np.ndarray) -> tp.Tuple[np.ndarray, int]:
    """Helper function to process audio data into mel spectrogram.
    
    Args:
        audio (np.ndarray): Audio data as numpy array
    Returns:
        tuple: (mel spectrogram, number of frames)
    """
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1


def load_audio_from_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> tp.Tuple[np.ndarray, int]:
    """Load and process audio data from bytes using ffmpeg.
    
    Args:
        audio_bytes (bytes): Raw audio data in bytes
        sample_rate (int): Sample rate of the audio data
    Returns:
        tuple: (mel spectrogram, number of frames)
    """
    # Create ffmpeg process with same parameters as whisper
    process = (
        ffmpeg.input('pipe:', threads=0)
        .output(
            'pipe:',
            format='s16le',
            acodec='pcm_s16le',
            ac=1,
            ar=sample_rate,
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    
    # Feed audio bytes and get output
    output, _ = process.communicate(input=audio_bytes)
    
    # Convert to numpy array and normalize (same as whisper implementation)
    audio = np.frombuffer(output, np.int16).flatten().astype(np.float32) / 32768.0
    
    return _process_audio(audio)


def load_audio_from_path(audio_path: tp.Union[str, Path], sample_rate: int = 16000) -> tp.Tuple[np.ndarray, int]:
    audio = whisper.load_audio(audio_path, sr=sample_rate)
    return _process_audio(audio)



def get_whisper_embeddings(whisper_model, mel, leng):
    """

    :param whisper_model:
    :param mel:
    :return: shape will be [seq, emb_size]
    """
    device = whisper_model.device
    with torch.no_grad():
        audio_feature = whisper_model.embed_audio(mel.unsqueeze(0).to(device))[0][:leng]
    return audio_feature


def get_input_template(token_config, seq_length, model_layers=8, speical_token_name:str="answer_a"):

    input_ids = list()
    for i in range(model_layers - 1):
        audio_input_token = layershift(token_config["input_a"], i)
        audio_pad_token = layershift(token_config["pad_a"], i)
        audio_end_token = layershift(token_config["eoa"], i)
        audio_special_token = layershift(token_config[speical_token_name], i)
        input_tokens = [audio_input_token] + [audio_pad_token] * seq_length + [audio_end_token, audio_special_token]
        input_tokens = torch.tensor(input_tokens)
        input_ids.append(input_tokens)
    
    # text input tokens
    text_input_tokens = [token_config["pad_t"]] + [token_config["pad_t"]] * seq_length + [token_config["pad_t"], token_config["answer_t"]]
    text_input_tokens = torch.tensor(text_input_tokens)
    input_ids.append(text_input_tokens)
    input_ids = [inp.detach().contiguous() for inp in input_ids]

    return input_ids


def get_audio_template(token_config, max_seq_length=None, model_layers=8, last_token_name="pad_a"):
    input_ids = []
    # First model_layers-1 layers use audio padding tokens with layershift
    for i in range(model_layers - 1):
        audio_pad_token = layershift(token_config["pad_a"], i)
        pad_tokens = [audio_pad_token] * (max_seq_length - 1)
        _last_token = layershift(token_config[last_token_name], i)
        input_ids.append(torch.tensor(pad_tokens + [_last_token]))
    return input_ids

def pad_text_tokens(token_config, tokens, max_seq_length):
    
    text_tokens = torch.cat([
        torch.tensor([token_config["input_t"]], dtype=torch.long),
        tokens,
        torch.tensor([token_config["pad_t"]] * (max_seq_length - len(tokens) - 3), dtype=torch.long),
        torch.tensor([token_config["eot"], token_config["answer_t"]], dtype=torch.long),
    ])
    return text_tokens




def get_text_input(tokens: torch.Tensor, token_config, max_seq_length=None, model_layers=8):
    """
    Generate input IDs for text tokens across model layers.
    
    Args:
        tokens (torch.Tensor): Text tokens to process
        token_config (dict): Configuration dictionary containing token mappings
        max_seq_length (int): Maximum sequence length
        model_layers (int): Number of model layers (default=8)
    
    Returns:
        list: List of tensor inputs for each layer
    """
    input_ids = []
    seq_length = len(tokens) if max_seq_length is None else max_seq_length
    # First model_layers-1 layers use audio padding tokens with layershift
    for i in range(model_layers - 1):
        audio_pad_token = layershift(token_config["pad_a"], i)
        pad_tokens = [audio_pad_token] * (seq_length + 3)
        input_ids.append(torch.tensor(pad_tokens))
    
    # Last layer contains actual text tokens
    text_tokens = torch.cat([
        torch.tensor([token_config["input_t"]], dtype=torch.long),
        tokens,
        torch.tensor([token_config["pad_t"]] * (seq_length - len(tokens)), dtype=torch.long),
        torch.tensor([token_config["eot"], token_config["answer_t"]], dtype=torch.long)
    ])
    input_ids.append(text_tokens)
    return input_ids


def pad_to_max_length(input_tensor, max_seq_length, pad_value=0):
    """
    Pad or truncate a 2D or 3D tensor to a specified maximum sequence length using F.pad.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [seq, emb_size] or [n, seq, emb_size]
        max_seq_length (int): The desired sequence length after padding or truncation.
        pad_value (int or float): Value to use for padding shorter sequences.

    Returns:
        torch.Tensor: A tensor with padded or truncated sequences.
    """
    current_length = input_tensor.size(-2)  # Get the current sequence length (second to last dimension)
    if current_length < max_seq_length:
        # Calculate padding size
        padding_size = max_seq_length - current_length
        if input_tensor.dim() == 2:
            # Pad only the sequence dimension (second dimension of 2D tensor)
            pad_dims = (0, 0, 0, padding_size)  # (pad_left, pad_right, pad_top, pad_bottom)
        elif input_tensor.dim() == 3:
            # Pad the sequence dimension (second dimension of 3D tensor)
            pad_dims = (0, 0, 0, padding_size, 0, 0)  # Expand to fit the 3D case
        else:
            raise Exception(f"input_tensor's dim incorrect, should be 2 or 3, but get {input_tensor.dim()}")
        padded_tensor = F.pad(input_tensor, pad_dims, 'constant', pad_value)
    elif current_length > max_seq_length:
        # Truncate the tensor
        padded_tensor = input_tensor[..., :max_seq_length, :]
    else:
        # No padding or truncation needed
        padded_tensor = input_tensor

    return padded_tensor


def construct_snac_tokens(snac: tp.AnyStr, layers=7) -> tp.Tuple[tp.List[tp.List[tp.AnyStr]], int]:
    
    snac_layers = snac.split("#")
    snac_tokens = [[] for _ in range(layers)]
    n_layers = 0
    for _layer in snac_layers:
        if _layer == "":
            continue
        tokens = _layer.strip().split(" ")
        for i, token in enumerate(tokens):
            snac_tokens[i].append(int(token))
        n_layers += 1
    return snac_tokens, n_layers
        
def pad_snac_tokens(token_config: tp.Dict, snac_tokens: tp.List[tp.List[int]], max_seq_length: int):
    
    padded_snac_tokens = list()
    pad_a = token_config["pad_a"]
    eoa = token_config["eoa"]
    padding_mask = list()
    for i, tokens in enumerate(snac_tokens):
        available_space = max_seq_length - (i + 1) - 1 
        
        if len(tokens) > available_space:
            tokens = tokens[:available_space]
            
        padded_token = [pad_a] * (i + 1) + tokens + [eoa]
        _mask = [0] * (i + 1) + [1] * len(tokens) + [1]
        
        if len(padded_token) < max_seq_length:
            padded_token = padded_token + [pad_a] * (max_seq_length - len(padded_token))
            _mask = _mask + [0] * (max_seq_length - len(_mask))
            
        padded_snac_tokens.append(padded_token)
        padding_mask.append(_mask)
        assert len(padded_token) == max_seq_length == len(_mask), "padded_token and padding_mask should have the same length"
        
    return padded_snac_tokens, padding_mask
    

def get_target_text_token(text: tp.AnyStr, tokenizer, token_config, max_seq_length=None):
    tokens = tokenizer.encode(text)
    input_t = token_config["input_t"]
    pad_t = token_config["pad_t"]
    eot = token_config["eot"]
    # Convert to tensor and concatenate using proper tensor operations
    tokens = torch.cat([
        torch.tensor([input_t], dtype=torch.long),
        tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens),
        torch.tensor([eot], dtype=torch.long)
    ])
    token_mask = torch.ones(len(tokens), dtype=torch.bool)
    
    if max_seq_length is not None:
        padding = torch.full((max_seq_length - len(tokens),), pad_t, dtype=torch.long)
        tokens = torch.cat([tokens, padding])
        # Create mask using proper tensor operations
        mask_padding = torch.zeros(max_seq_length - len(token_mask), dtype=torch.bool)
        token_mask = torch.cat([token_mask, mask_padding])
    
    return tokens, token_mask


def auto_expand_mask(mask, target_shape):
    """
    Automatically expands a 1D mask to match the target shape.
    
    Args:
        mask (torch.Tensor): Input 1D mask of shape (B,).
        target_shape (tuple): Desired shape to expand to.
    
    Returns:
        torch.Tensor: Expanded mask with appropriate broadcasting.
    """
    # Dynamically unsqueeze the mask to match the dimensions
    while len(mask.shape) < len(target_shape):
        mask = mask.unsqueeze(-1)  # Add new dimensions to the end
    return mask
