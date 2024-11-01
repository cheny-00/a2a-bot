# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy

import inspect

from lightning.pytorch.cli import instantiate_class


def get_mapped_kwargs(func, kwargs):
    sig = inspect.signature(func)
    class_args = [
        param.name for param in sig.parameters.values() if param.name != "self"
    ]
    mapped_args = {arg: kwargs[arg] for arg in class_args if arg in kwargs}

    return mapped_args


def instantiate_torch_optimizer(optimizer, model_parameters, **kwargs):
    # From: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/utils.py#L561
    # Special care taken where some optimizers do not have some parameters referenced in some of the code, for example "fused" in the pretrain.py script:
    #   bnb.optim.AdamW8bit
    #   grokadamw.GrokAdamW
    #   torch.optim.RMSprop

    if isinstance(optimizer, str):
        if "." in optimizer:
            class_module, class_name = optimizer.rsplit(".", 1)
        else:
            class_module, class_name = "torch.optim", optimizer

        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}
        optimizer = optimizer_cls(model_parameters, **kwargs)
    elif isinstance(optimizer, dict):
        optimizer = dict(optimizer)
        class_module, class_name = optimizer["class_path"].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        optimizer_cls = getattr(module, class_name)

        valid_params = set(inspect.signature(optimizer_cls).parameters)
        kwargs = {key: value for key, value in dict(kwargs).items() if key in valid_params}

        optimizer["init_args"].update(kwargs)
        optimizer = instantiate_class(model_parameters, optimizer)
    else:
        raise ValueError(f'Unrecognized "optimizer" value: {optimizer}')

    return optimizer
