# -- coding: utf-8 --
# @Time    :   2024/10/31
# @Author  :   chy


import pytorch_lightning as pl


def omni_stage_1_training(self: pl.LightningModule, batch, batch_idx):
    audio_feature = batch['audio_feature']
    input_ids = batch['input_ids']
    audio_length = batch['audio_length']
    target = batch['text']

    logit_a, logit_t = self(audio_feature, input_ids, whisper_lens=audio_length, task='asr')

    loss = self.loss_function["xnet"](logit_t.reshape(-1, logit_t.size(-1)), target.reshape(-1))

    self.log(
        "loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
    )

    return loss
