# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import torch
import whisper
import numpy as np
import torch.nn.functional as F

from mini_omni.snac_utils.snac_utils import layershift


def load_audio_from_bytes(audio_bytes):
    # from https://github.com/openai/whisper/blob/main/whisper/audio.py#L62
    audio = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)

    return mel, int(duration_ms / 20) + 1


def get_whisper_embeddings(whisper_model, mel):
    """

    :param whisper_model:
    :param mel:
    :return: shape will be [seq, emb_size]
    """
    device = whisper_model.device
    with torch.no_grad():
        audio_feature = whisper_model.embed_audio(mel.unsqueeze(0).to(device))
    return audio_feature


def get_input_template(token_config, seq_length, model_layers=8):

    input_ids = list()
    for i in range(model_layers - 1):
        audio_input_token = layershift(token_config["input_a"], i)
        audio_pad_token = layershift(token_config["pad_a"], i)
        audio_end_token = layershift(token_config["eoa"], i)
        audio_special_token = layershift(token_config["answer_a"], i)
        input_tokens = [audio_input_token] + [audio_pad_token] * seq_length + [audio_end_token, audio_special_token]
        input_tokens = torch.tensor(input_tokens)
        input_ids.append(input_tokens)
    text_input_tokens = [token_config["input_t"]] + [token_config["pad_t"]] * seq_length + [token_config["eot"], token_config["answer_t"]]
    text_input_tokens = torch.tensor(text_input_tokens)
    input_ids.append(text_input_tokens)

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
