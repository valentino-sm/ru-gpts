#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from typing import Union, Iterable

import numpy as np
import torch
if os.environ.get("FORCE_CPU", "0").lower() in ["1", "true", "yes"]:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
from deepspeed import DeepSpeedConfig
from huggingface_hub import hf_hub_download
from torch.nn import CrossEntropyLoss
from transformers import GPT2Tokenizer, PreTrainedModel, PretrainedConfig

from src import mpu
from .fp16 import FP16_Module
from .model import GPT3Model
from .download_utils import WEIGHTS_NAME, DEEPSPEED_CONFIG_NAME
from transformers.utils import logging


logger = logging.get_logger(__name__)
NoneType = type(None)


def get_deepspeed_config(path):
    return DeepSpeedConfig(path)


def get_sparse_attention_config(path, num_heads):
    ds_config = get_deepspeed_config(path)
    if hasattr(ds_config, 'sparse_attention') and ds_config.sparse_attention:
        sa_config = ds_config.sparse_attention
        sa_mode = sa_config.get('mode')
        if sa_mode == 'dense':
            from deepspeed.ops.sparse_attention import DenseSparsityConfig as STConfig
        elif sa_mode == 'fixed':
            from deepspeed.ops.sparse_attention import FixedSparsityConfig as STConfig
        elif sa_mode == 'bigbird':
            from deepspeed.ops.sparse_attention import BigBirdSparsityConfig as STConfig
        elif sa_mode == 'bslongformer':
            from deepspeed.ops.sparse_attention import BSLongformerSparsityConfig as STConfig
        elif sa_mode == 'variable':
            from deepspeed.ops.sparse_attention import VariableSparsityConfig as STConfig
        else:
            raise NotImplementedError(
                f'Given sparsity mode, {sa_mode}, has not been implemented yet!'
            )
        del sa_config['mode']
        return STConfig(num_heads=num_heads, **sa_config)
    else:
        return None


def get_model(deepspeed_config_path):
    num_local_heads = 16
    sparse_mode = 'alternating'
    deepspeed_sparsity_config = get_sparse_attention_config(deepspeed_config_path, num_local_heads)
    if deepspeed_sparsity_config is not None:
        logger.info(f"Use sparse attention with mode {sparse_mode}")
    else:
        logger.info(f"Use dense attention")
    model = GPT3Model(num_layers=24,
                      vocab_size=50264,
                      hidden_size=2048,
                      num_attention_heads=num_local_heads,
                      embedding_dropout_prob=0.1, attention_dropout_prob=0.1, output_dropout_prob=0.1,
                      max_sequence_length=2048,
                      checkpoint_activations=False,
                      checkpoint_num_layers=1,
                      parallel_output=False,
                      deepspeed_sparsity_config=deepspeed_sparsity_config,
                      sparse_mode=sparse_mode)
    # GPU allocation.
    model.to(device)

    # Fp16 conversion only for non-CPU devices.
    if device.type != 'cpu':
        model = FP16_Module(model)

    return model


def setup_model(weights_path, deepspeed_config_path):
    model = get_model(deepspeed_config_path)
    print("Load checkpoint from " + weights_path)
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)['module']
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("Model Loaded")
    return model


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indices where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indices:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


class ModelOutput(object):
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss

    def __getitem__(self, key):
        if key == "logits":
            return self.logits
        raise StopIteration


class RuGPT3XL(PreTrainedModel):
    def __init__(self, model, tokenizer, model_path, seq_len=512):
        super().__init__(PretrainedConfig())
        self.model = model
        self.pad_token_id = tokenizer.encoder['<pad>']
        self.eos_token_id = tokenizer.encoder['<|endoftext|>']
        self.seq_len = seq_len
        self.model_path = model_path
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path=None, seq_len=512, weights_path=None, deepspeed_config_path=None):
        # Distributed initialization removed for single GPU inference.

        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.distributed.is_initialized():
            mpu.model_parallel_cuda_manual_seed(seed)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
        logger.info("Check cached model files...")
        if weights_path is None:
            weights_path = hf_hub_download(model_name_or_path, WEIGHTS_NAME)
        if deepspeed_config_path is None:
            deepspeed_config_path = hf_hub_download(model_name_or_path, DEEPSPEED_CONFIG_NAME)
        model = setup_model(weights_path, deepspeed_config_path)
        model.to(device)
        model = model.eval()
        return cls(model, tokenizer=tokenizer, seq_len=seq_len, model_path=model_name_or_path)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        kwargs.update({"input_ids": input_ids})
        return kwargs

    def generate(
            self, text: Union[str, NoneType] = None,
            input_ids: Union[torch.LongTensor, NoneType] = None,
            max_length: Union[int, None] = None,
            min_length: Union[int, NoneType] = None,
            do_sample: Union[bool, NoneType] = None,
            early_stopping: Union[bool, NoneType] = None,
            num_beams: Union[int, NoneType] = None,
            temperature: Union[float, NoneType] = None,
            top_k: Union[int, NoneType] = None,
            top_p: Union[float, NoneType] = None,
            repetition_penalty: Union[float, NoneType] = None,
            bad_words_ids: Union[Iterable[int], NoneType] = None,
            bos_token_id: Union[int, NoneType] = None,
            pad_token_id: Union[int, NoneType] = None,
            eos_token_id: Union[int, NoneType] = None,
            length_penalty: Union[float, NoneType] = None,
            no_repeat_ngram_size: Union[int, NoneType] = None,
            num_return_sequences: Union[int, NoneType] = None,
            decoder_start_token_id: Union[int, NoneType] = None,
            use_cache: Union[bool, NoneType] = None,
            **model_kwargs):
        if text is not None:
            input_ids = torch.tensor([self.tokenizer(text)['input_ids']], device=device, dtype=torch.long)
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        res = super().generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            **model_kwargs
        )
        return list(map(self.tokenizer.decode, res.tolist()))

    def __call__(self, text=None, input_ids=None, labels=None, **kwargs):
        if input_ids is None:
            if text is None:
                text = ""
            input_ids = torch.tensor([self.tokenizer(text)['input_ids']], device=device, dtype=torch.long)
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=device, dtype=torch.long)
        if isinstance(labels, list):
            labels = torch.tensor(labels, device=device, dtype=torch.long)
        res = []
        if labels is not None:
            lbls = labels
        else:
            lbls = [None] * len(input_ids)
        loss = None
        original_context_length = 0
        seq_len = self.seq_len
        for tokens, lbl in zip(input_ids, lbls):
            context_tokens = tokens.tolist()
            context_length = len(context_tokens)
            original_context_length = len(context_tokens)
            
            while context_length > seq_len:
                seq_len += 16
            if context_length < seq_len:
                context_tokens.extend([self.pad_token_id] * (seq_len - context_length))
                if labels is not None:
                    lbl = lbl.tolist()
                    lbl.extend([self.pad_token_id] * (seq_len - context_length))
                    lbl = torch.tensor(lbl, device=device, dtype=torch.long)
            if context_length > 2048:
                context_tokens = context_tokens[-2048:]
                if labels is not None:
                    lbl = lbl.tolist()[-2048:]
                    lbl = torch.tensor(lbl, device=device, dtype=torch.long)
            context_tokens_tensor = torch.tensor(context_tokens, device=device, dtype=torch.long)
            context_length_tensor = torch.tensor([context_length], device=device, dtype=torch.long)


            # context_length = context_length_tensor[0].item()

            tokens = context_tokens_tensor
            tokens = tokens.view(1, -1).contiguous()
            tokens = tokens.to(device)
            attention_mask, loss_mask, position_ids = get_masks_and_position_ids(tokens, self.pad_token_id, False,
                                                                                 False)
            lm_logits = self.model(tokens, position_ids, attention_mask)
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = lbl[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=self.pad_token_id)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            res.append((lm_logits, loss))
        logits = torch.cat([x[0] for x in res], dim=0)[:, : original_context_length, :]
        if loss is not None:
            loss = [x[1] for x in res]
        return ModelOutput(logits, loss)
