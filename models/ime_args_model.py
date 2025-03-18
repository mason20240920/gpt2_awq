#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : ime_args_model.py
@Author  : Barry Allen
@Date    : 2025/3/17 17:42
@Desc    :
"""

from models.singleton_meta import SingletonMeta

class IMEArgsModel(metaclass=SingletonMeta):
    """
        输入拼音的参数
        """
    _model_name: str = ""  # 模型名称

    _dst_path: str = ""  # 目标文件夹

    _quant_block: int = 128  # quant block, default is 0 mean channle-wise

    _quant_bit: int = 8  # 4 or 8, default is 8.

    _sym: bool = False  # 是否使用对称量化

    _awq: bool = True  # 是否使用AWQ激活权重量化

    def __init__(self,
                 model_name: str,
                 dst_path: str,
                 quant_block: int = 128,
                 quant_bit: int = 8,
                 sym: bool = False,
                 awq: bool = True):
        """
        初始化Args对象
        :param model_name:
        :param dst_path:
        :param quant_block:
        :param quant_bit:
        :param sym:
        :param awq:
        """
        self._model_name = model_name
        self._dst_path = dst_path
        self._quant_block = quant_block
        self._quant_bit = quant_bit
        self._sym = sym
        self._awq = awq

    def dst_path(self) -> str:
        """
        目标路径
        :return:
        """
        return self._dst_path

    def quant_block(self) -> int:
        return self._quant_block

    def quant_bit(self) -> int:
        return self._quant_bit

    def sym(self) -> bool:
        return self._sym

    def awq(self) -> bool:
        return self._awq

    def model_name(self) -> str:
        return self._model_name