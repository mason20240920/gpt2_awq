#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : ime_model_mapper.py
@Author  : Barry Allen
@Date    : 2025/3/16 11:16
@Desc    : 从原始模型到量化模型的映射器
"""
from typing import List, Dict, Tuple
from transformers import GPT2Config

class IMEModelMapper(object):
    config_key: str # 配置的关键字

    model_key: str # 模型关键字

    decoder_key: str # 解码器的关键字

    attention_key: str # 注意力的关键字

    default_config: Dict # 默认的配置

    default_model: Dict # 默认的模型

    default_decoder: Dict # 默认的解码器

    default_attention: Dict # 默认注意力机制

    default_map: Dict # 默认模型结构

    def __init__(self):
        self.attrs: List = []
        self.mapper = dict()
        self.register_model()

    def get_map(self, config: GPT2Config) -> Tuple[str, Dict]:
        model_type:str = config.model_type
        return model_type, self.default_map

    def register_model(self):
        """
        注册model
        :return:
        """
        self.register_default()


    def register_default(self):
        """
        注册默认函数
        :return:
        """
        self.config_key = 'config'
        self.model_key = 'model'
        self.decoder_key = 'decoder'
        self.attention_key = 'attention'
        self.default_config = {
            'hidden_size': 'n_embd',
            'num_attention_heads': 'n_head',
            'num_hidden_layers': 'n_layer',
        }
        self.default_model = {
            'lm_': 'lm_head',
            'wte': 'transformer.wte',
            'wpe': 'transformer.wpe',
            'blocks_': 'transformer.h',
            'final_layer_norm_': 'transformer.ln_f',
        }
        self.default_decoder = {
            "self_attn": "attn",
            "mlp": "mlp",
            "input_rmsnorm": "ln_1",
            "post_attn_rmsnorm": "ln_2",
        }
        self.default_attention = {
            "c_attn": "c_attn",
            "c_proj": "c_proj",
        }
        self.default_map = {
            'config': self.default_config,
            'model': self.default_model,
            'decoder': self.default_decoder,
            'attention': self.default_attention
        }

    @staticmethod
    def do_map(dst, src, map_dict):
        for dst_attr, src_attr in map_dict.items():
            attributes = src_attr.split('.')
            obj = src
            for attr in attributes:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    obj = None
                    break
            setattr(dst, dst_attr, obj)


