# coding=utf-8
# Copyright (c) 2020, Sber.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer."""

import math

import torch
from torch.nn import LayerNorm
from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear
from .utils import divide
from .utils import split_tensor_along_last_dim
from src.utils import DEEPSPEED_WRAP


class GPT3ParallelSelfAttention(torch.nn.Module):
    """Parallel self-attention layer for GPT3.

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, output_layer_init_method=None,
                 use_deepspeed_sparse=None):
        super(GPT3ParallelSelfAttention, self).__init__()
        self.use_deepspeed_sparse = use_deepspeed_sparse
        if DEEPSPEED_WRAP:
            deepspeed = DEEPSPEED_WRAP.deepspeed
            from deepspeed.ops.sparse_attention import SparseSelfAttention
        if self.use_deepspeed_sparse is not None:
            self.sparse_self_attention = SparseSelfAttention(self.use_deepspeed_sparse)
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size,
                                                     num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads,
                                                        world_size)
        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(hidden_size, 3 * hidden_size,
                                                    stride=3,
                                                    gather_output=False,
                                                    init_method=init_method)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(hidden_size,
                                       hidden_size,
                                       input_is_parallel=True,
                                       init_method=output_layer_init_method)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if DEEPSPEED_WRAP:
            if DEEPSPEED_WRAP.deepspeed.checkpointing.is_configured():
                global get_cuda_rng_tracker, checkpoint
                get_cuda_rng_tracker = DEEPSPEED_WRAP.deepspeed.checkpointing.get_cuda_rng_tracker
                checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ltor_mask):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if self.use_deepspeed_sparse:
            context_layer = self.sparse_self_attention(
                query_layer.half(),
                key_layer.half(),
                value_layer.half(),
                attn_mask=ltor_mask).half()
        else:
            # Raw attention scores. [b, np, s, s]
            attention_scores = torch.matmul(query_layer,
                                            key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.hidden_size_per_attention_head)
            # Apply the left to right attention mask.
            attention_scores = torch.mul(attention_scores, ltor_mask) - 10000.0 * (1.0 - ltor_mask)

            # Attention probabilities. [b, np, s, s]
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            if DEEPSPEED_WRAP and DEEPSPEED_WRAP.deepspeed.checkpointing.is_configured():
                with get_cuda_rng_tracker().fork():
                    attention_probs = self.attention_dropout(attention_probs)
            else:
                attention_probs = self.attention_dropout(attention_probs)

            # Context layer.
            # [b, np, s, hn]
            context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


# @torch.jit.script
# Remove torch.jit for colab
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


class GPT3ParallelMLP(torch.nn.Module):
    """MLP for GPT3.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.

    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method,
                 output_layer_init_method=None):
        super(GPT3ParallelMLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(hidden_size, 4 * hidden_size,
                                                  gather_output=False,
                                                  init_method=init_method)
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4 * hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPT3ParallelTransformerLayer(torch.nn.Module):
    """A single layer transformer for GPT3.

    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 output_layer_init_method=None,
                 use_deepspeed_sparse=None):
        super(GPT3ParallelTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = GPT3ParallelSelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            use_deepspeed_sparse=use_deepspeed_sparse)

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size,
                                                  eps=layernorm_epsilon)

        # MLP
        self.mlp = GPT3ParallelMLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GPT3ParallelTransformer(torch.nn.Module):
    """GPT-3 transformer.

    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 use_scaled_init_for_output_weights=True,
                 use_deepspeed_sparse=None,
                 sparse_mode='all'):
        super(GPT3ParallelTransformer, self).__init__()

        if DEEPSPEED_WRAP:
            from deepspeed.ops.sparse_attention import SparseSelfAttention

        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(init_method_std,
                                                          num_layers)
        if use_deepspeed_sparse and sparse_mode == 'alternating':
            print('Use alternating sparse & dense attention layers')

        def get_layer(layer_num, num_layers):
            sparsity_config = use_deepspeed_sparse
            if use_deepspeed_sparse:
                if sparse_mode == 'alternating' and layer_num % 2:  # even layers are dense
                    sparsity_config = None
                elif sparse_mode == 'top_bottom' and layer_num >= num_layers // 2:  # top levels are dense
                    sparsity_config = None
            return GPT3ParallelTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                use_deepspeed_sparse=sparsity_config)

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [get_layer(i, num_layers) for i in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        if DEEPSPEED_WRAP:
            if DEEPSPEED_WRAP.deepspeed.checkpointing.is_configured():
                global get_cuda_rng_tracker, checkpoint
                get_cuda_rng_tracker = DEEPSPEED_WRAP.deepspeed.checkpointing.get_cuda_rng_tracker
                checkpoint = DEEPSPEED_WRAP.deepspeed.checkpointing.checkpoint

    def forward(self, hidden_states, attention_mask):

        def custom(start, end):
            def custom_forward(*inputs):
                layers_ = self.layers[start:end]
                x_ = inputs[0]
                for layer in layers_:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.layers)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                hidden_states = checkpoint(custom(l, l + chunk_length),
                                           hidden_states, attention_mask)
                l += chunk_length
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm.
        output = self.final_layernorm(hidden_states)

        return output
