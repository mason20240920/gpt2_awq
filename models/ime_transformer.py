#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : ime_transformer.py
@Author  : Barry Allen
@Date    : 2025/3/16 18:12
@Desc    :
"""
from typing import Any

from torch import nn

from utils.ime_model_mapper import IMEModelMapper
import torch

class IMEEmbedding(torch.nn.Module):
    """
    讯飞输入法的Embedding
    """
    def __init__(self, wte: nn.Embedding, wpe: nn.Embedding,  config: Any):
        super(IMEEmbedding, self).__init__()
        self.hidden_size = config.hidden_size
        self.wte: nn.Embedding = wte
        self.wpe: nn.Embedding = wpe

    def forward(self, input_ids: torch.Tensor):
        input_shape = input_ids.size()  # [batch size, sequence length]
        input_ids = input_ids.view(-1, input_shape[-1])
        # 2. 确定输入的设备
        device: torch.device = input_ids.device if input_ids is not None else "cpu"
        # 4. 确定输入的位置编码
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long,
                                    device=device)  # [sequence length + 0]
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        return hidden_states


class IMEAttention(nn.Module):
    """
    讯飞输入法的Attention
    """
    layer_id: int # 当前层编号
    hidden_size: int # 隐藏层维度大小
    head_dim: int # 每个注意力头的维度
    num_heads: int # 查询(Q)注意力头数量

    def __init__(self,
                 attn: nn.Module,
                 layer_id: int,
                 config: Any):
        super(IMEAttention, self).__init__()
        self.layer_id = layer_id
        max_positions: int = config.max_length
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.split_size = self.hidden_size
        assert self.hidden_size % self.num_heads == 0

        # 不需要梯度，不会被优化器更新 bias
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 不需要梯度, 不会被优化器更新
        self.register_buffer("masked_bias", torch.tensor(0.0), persistent=False)

        # 将源注意力模块中的参数映射到当前模块
        IMEModelMapper.do_map(dst=self, src=attn, map_dict=config.model_map["attention"])

    def forward(self,
                hidden_states: torch.Tensor):
        """
        完全适配GPT-2的推理结构
        :param hidden_states:
        :return:
        """
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)  # [batch_size, seq_len, num_heads, head_dim]

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        attn_weights: torch.Tensor = torch.matmul(query_states, key_states.transpose(-1, -2))

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query_states.size(-2), key_states.size(-2)
        causal_mask: torch.Tensor = self.bias[:, :, key_length - query_length: key_length, :key_length].to(attn_weights.device)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)

        return attn_output

class IMEDecoder(nn.Module):
    """
    讯飞输入法的Decoder
    """
    def __init__(self, decoder: nn.Module,
                 layer_id: int,
                 config: Any):
        super(IMEDecoder, self).__init__()
        IMEModelMapper.do_map(self, decoder, config.model_map["decoder"])
        self.self_attn = IMEAttention(self.self_attn, layer_id=layer_id, config=config)
        self.hidden_size = config.hidden_size
        self.alpha: float = 1.0

    def forward(self, hidden_states: torch.Tensor):
        """
        推理
        :param hidden_states:
        :return:
        """
        residual: torch.Tensor = hidden_states
        hidden_states: torch.Tensor = self.input_rmsnorm(hidden_states)
        # Self Attention
        attn_output = self.self_attn(hidden_states=hidden_states)
        # Fully Connected
        hidden_states = residual + attn_output
        # general
        residual = hidden_states
        hidden_states = self.post_attn_rmsnorm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states=hidden_states)
        # 归一化
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states



