#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : inter_mid_gpt2_model.py
@Author  : Barry Allen
@Date    : 2025/3/17 11:39
@Desc    : 中间转换的GPT2Model
"""
from typing import Dict, List

import torch
from transformers import GPT2Config

from awq_core.awq_quantizer import AwqQuantizer
from models.ime_args_model import IMEArgsModel
from models.ime_gpt2_lm_head_model import IMEGPT2LMHeadModel
from torch import nn

from models.ime_transformer import IMEDecoder, IMEEmbedding
from utils.ime_model_mapper import IMEModelMapper
from utils.other_utils import spinner_run, visit_module, export_awq_gpt2_to_onnx


class InterMidGPT2Model(nn.Module):
    model: IMEGPT2LMHeadModel  # 预训练的GPT2模型

    config: GPT2Config # GPT2的配置文件

    model_type: str # 模型类型

    model_map: Dict # 模型映射

    head_dim: int # 每个头的dim维度

    hidden_size: int # hidden维度

    num_attention_heads: int # 注意力头的数量

    blocks: nn.ModuleList = []

    max_length: int # 最大长度

    args: IMEArgsModel

    embed: IMEEmbedding  # 张量映射

    def __init__(self, config: GPT2Config, model: IMEGPT2LMHeadModel, args: IMEArgsModel):
        """
        初始化中间层的model
        :param config: GPT2的配置文件
        :param model: 输入法优化后的GPT2模型
        """
        super(InterMidGPT2Model, self).__init__()
        self.init_from_config(args=args)
        self.load_model(pretrained_model=model, config=config)

    def init_from_config(self, args: IMEArgsModel):
        self.args = args
        self.max_length = 16
        # self.dst_name = "llm"
        # # load config from args
        # self.onnx_path = os.path.join(self.args.dst_path(), "onnx")
        # # init export dst dir
        # if not os.path.exists(self.args.dst_path()):
        #     os.makedirs(self.args.dst_path())
        # if not os.path.exists(self.onnx_path):
        #     os.makedirs(self.onnx_path)

    def load_pretrained_model(self,
                              pretrained_model: IMEGPT2LMHeadModel,
                              config: GPT2Config):
        """
        加载预训练模型
        :param pretrained_model: 预训练的模型
        :param config: GPT2配置文件
        :return:
        """
        self.model = pretrained_model
        self.config = config



    @spinner_run(f"load pretrained ", True)
    def load_model(self,
                   pretrained_model: IMEGPT2LMHeadModel,
                   config: GPT2Config):
        """
        加载模型
        :param pretrained_model: 预训练模型
        :param config: 配置文件
        :return:
        """
        self.load_pretrained_model(pretrained_model=pretrained_model,
                                   config=config)
        model_mapper: IMEModelMapper = IMEModelMapper()
        self.model_type, self.model_map = model_mapper.get_map(config=config)

        # 进行AWQ量化的话，必须全部为float
        self.model.float()
        visit_module(module=self.model) # 全部权重参数设置为float
        # 匹配配置文件到中间变量的模型上
        IMEModelMapper.do_map(dst=self,
                              src=self.config,
                              map_dict=self.model_map["config"])
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 匹配模型的参数到中间变量的模型上
        IMEModelMapper.do_map(dst=self, src=self.model, map_dict=self.model_map["model"])
        self.embed = IMEEmbedding(wte=self.wte, wpe=self.wpe, config=self)
        # 替换具体的block
        for block in self.blocks_:
            layer_id = len(self.blocks)
            self.blocks.append(IMEDecoder(decoder=block, layer_id=layer_id, config=self))

        return self.model_type

    @spinner_run(f"awq successfully ", True)
    def awq_quant(self):
        self.awq_quantizer = AwqQuantizer(inter_mid_model=self)
        self.awq_quantizer.quantize()
        return self.model_type

    def forward(self, hidden_states: torch.Tensor):
        for i in range(self.num_hidden_layers):
            hidden_states = self.blocks[i](hidden_states)
        return hidden_states

    def embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        对输入的input ids进行Embedding
        :param input_ids:
        :return:
        """
        input_embeds: torch.Tensor = self.embed(input_ids)
        return input_embeds

    @spinner_run("export onnx successfully ", True)
    def export(self, onnx_path: str="ime_awq_gpt.onnx"):
        if self.args.awq():
            self.awq_quant()
        self.to(torch.device("cpu"))
        export_awq_gpt2_to_onnx(model=self, save_path=onnx_path)
        return self.model_type




