#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : awq_qwen
@File    : singleton_meta.py
@Author  : Barry Allen
@Date    : 2025/3/6 09:19
@Desc    : 单例基类
"""

class SingletonMeta(type):
    """
    单例模式, 保证只初始化一次
    """
    _instance = {}

    def __call__(cls, *args, **kwargs):
        """
        如果该类还未创建实例，则调用父类创建一个新的实例
        :param args:
        :param kwargs:
        :return:
        """
        if cls not in cls._instance:
            instance = super().__call__(*args, **kwargs)
            cls._instance[cls] = instance
        # 否则直接返回已经创建实例
        return cls._instance[cls]

