#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : other_utils.py
@Author  : Barry Allen
@Date    : 2025/3/17 11:40
@Desc    :
"""
import functools
import gc
import inspect
import time
import traceback
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn

from yaspin import yaspin

from constant import CASES_FILE_NAME

RESET = "\033[0m"
GREEN = "\033[32;1m"
YELLOW = "\033[33;4m"

def spinner_run(text='Processing...', hide=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with yaspin(text=text, color="cyan") as spinner:
                start = time.time()
                try:
                    if hide: spinner.hide()
                    result = func(*args, **kwargs)
                    if hide: spinner.show()
                except Exception as e:
                    spinner.fail("💥 Failed")
                    traceback.print_exc()
                    exit(1)
                end = time.time()
                during = f'[{end-start:05.2f} s]'.replace('[0', '[ ')
                padding = ' ' * (64 - len(spinner.text) - len(result))
                spinner.text = f'{spinner.text}{YELLOW}{result}{RESET}{padding}{GREEN}{during}{RESET}'
                spinner.ok("✅ Done")
                return result
        return wrapper
    return decorator

def visit_module(module: nn.Module):
    """
    将module里面参数遍历为float
    :param module:
    :return:
    """
    if not isinstance(module, nn.Linear) and hasattr(module, 'weight'):
        module.float()
    for name, child in module.named_children():
        visit_module(child)

def read_testcases_to_awq(file_path: str = CASES_FILE_NAME,
                          n_samples: int = 20,
                          max_seq_len: int = 16) -> List[torch.Tensor]:
    """
    读取测试集来获取验证集合
    :param file_path: 返回文件路径
    :param n_samples: 最大样本数量
    :param max_seq_len: 最大字符串长度
    :return: 返回样本的Tensor
    """
    ret: List[torch.Tensor] = []
    with open(file=file_path, mode='rt', encoding='utf-8') as f:
        while True:
            line:str = f.readline().strip()
            if not line: break
            simple_int: List[int] = list(map(lambda x: int(x), line.split(" ")))
            if len(simple_int) > max_seq_len:
                continue
            if len(ret) > n_samples:
                break
            ret.append(torch.tensor(simple_int))
    return ret

def get_best_device() -> torch.device:
    """
    获取最佳的设备
    :return:
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def clear_memory(weight=None):
    """
    清理权重内存
    :param weight:
    :return:
    """
    if weight is not None:
        del weight
    gc.collect()
    # 根据设备类型执行特定的内存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()
    elif torch.backends.mps.is_available():
        pass

def get_named_linears(module: nn.Module) -> Dict[str, nn.Linear]:
    """
    获取线性函数名称
    :param module:
    :return:
    """
    return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}

def exclude_layers_to_not_quantize(linear_layers: Dict[str, nn.Linear],
                                   modules_to_not_convert: Optional[List[str]] = None) -> Dict[str, nn.Linear]:
    """
    排除不需要量化的层。AWQ论文中提到，并非所有权重都同等重要，保护关键权重对模型性能至关重要。
    这个函数允许用户指定某些层保持原始精度，而不是使用混合精度量化（这在论文中被认为是硬件效率低下的）。
    :param linear_layers:  所有线性层的字典，键为层名称
    :param modules_to_not_convert: 包含不需要量化的模块名称关键字列表
    :return:  过滤后需要进行量化的线性层字典
    """
    # 如果没有指定不量化的模块，返回所有层
    if modules_to_not_convert is None:
        return linear_layers

    filtered_layers = {}
    for name, linear_layer in linear_layers.items():
        if not any(key in name for key in modules_to_not_convert):
            filtered_layers[name] = linear_layer
    return filtered_layers

def sanitize_kwargs(inputs_kwargs: Dict, module: nn.Module):
    """
    清理并过滤输入参数，移除模块前向传播中不支持的参数。
    这个方法的主要目的是确保在不同版本的transformers库之间保持兼容性

    工作原理：
    1. 获取模块forward方法的参数签名
    2. 仅保留在签名中存在的参数
    3. 返回过滤后的参数字典
    :param inputs_kwargs:  输入参数字典，包含要传递给模型层的所有参数
    :param module:  目标量化模块，通常是transformer中的某个子模块
    :return:  dict: 经过清理的参数字典，只包含模块forward方法支持的参数
    """
    module_signature = inspect.signature(module.forward).parameters  # 获取模块forward方法的参数签名
    sanitized_kwargs = {}  # 创建新字典存储过滤后的参数
    for k, v in inputs_kwargs.items():  # 遍历输入参数，只保留在模块签名中存在的参数
        if k in module_signature:
            sanitized_kwargs[k] = v
    return sanitized_kwargs


def get_op_name(module: nn.Module, op: nn.Module):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

def get_op_by_name(module: nn.Module, op_name: str):
    """
    根据算子名称获取model
    :param module: 模型
    :param op_name:  算子名称
    :return:
    """
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

def apply_scale(module, scales_list, input_feat_dict=None):
    """
    执行最优的scale
    :param module:
    :param scales_list:
    :param input_feat_dict:
    :return:
    """
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        best_device = get_best_device()
        prev_op.to(best_device)
        for layer in layers:
            layer.to(best_device)
        scales.to(best_device)
        if (
                isinstance(prev_op, torch.nn.Linear)
                and type(layers) == list
                and isinstance(layers[0], torch.nn.Linear)
        ):
            if len(layers) == 1:
                scale_fc_fc(prev_op, layers[0], scales)
            else:
                scale_fc_fcs(prev_op, layers, scales)
        elif (
                is_allowed_norms(prev_op)
                or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)

        elif is_allowed_act_fns(prev_op):
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()

@torch.no_grad()
def scale_fc_fc(fc1: torch.nn.Linear, fc2: torch.nn.Linear, scales: torch.Tensor):
    """
    对两个相邻的全连接层应用AWQ缩放策略
    :param fc1: 第一个全连接层
    :param fc2: 第二个全连接层
    :param scales: 缩放因子张量
    :return:
    """
    # 确保输入的层是nn.Linear类型
    assert isinstance(fc1, torch.nn.Linear)
    assert isinstance(fc2, torch.nn.Linear)

    # 将scales移到与fc1权重相同的设备上
    scales = scales.to(fc1.weight.device)

    # 对fc1最后几行权重进行逆向缩放(除以scales)
    # view(-1, 1)将scales转为列向量，确保广播机制正确应用到每行
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    # 如果fc1有偏置项，也需要相应缩放
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # 对fc2的权重进行正向缩放(乘以scales)
    # view(1, -1)将scales转为行向量，确保广播机制正确应用到每列
    fc2.weight.mul_(scales.view(1, -1))

    # 检查是否存在数值不稳定(NaN)的情况
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def scale_fc_fcs(fc1: torch.nn.Linear, fcs: List[torch.nn.Linear], scales: torch.Tensor):
    """
    对两个相邻的全连接层应用AWQ缩放策略
    :param fc1: 第一个全连接层
    :param fcs: 第二个全连接层列表
    :param scales:
    :return:
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    # 对每个后续层的权重进行正向缩放(乘以scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

def is_allowed_norms(op):
    """
    是否允许Norms
    :param op:
    :return:
    """
    if isinstance(op, torch.nn.LayerNorm):
        return True
    if any(t in str(type(op)) for t in ['LlamaRMSNorm', 'GemmaRMSNorm', 'CohereLayerNorm']):
        return True
    return False

@torch.no_grad()
def scale_ln_fcs(ln: torch.nn.Linear,
                 fcs: List[torch.nn.Linear],
                 scales: torch.Tensor):
    """
    含有RMSNorm下完成的缩放
    :param ln: 
    :param fcs: 
    :param scales: 
    :return: 
    """
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # GemmaRMSNorm is different from Llama's in that it multiplies
    # (1 + weight) to the output, instead of just weight.
    if 'GemmaRMSNorm' in str(type(ln)):
        ln.weight += 1
        ln.weight.div_(scales)
        ln.weight -= 1
    else:
        ln.weight.div_(scales)

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0

def is_allowed_act_fns(op):
    from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation
    allowed_act_fns = [
        torch.nn.GELU,
        NewGELUActivation,
        PytorchGELUTanh,
        GELUActivation,
    ]
    return op in allowed_act_fns

@torch.no_grad()
def scale_gelu_fc(gelu, fc: torch.nn.Linear, scales: torch.Tensor):
    """
    GELU
    :param gelu:
    :param fc:
    :param scales:
    :return:
    """
    assert is_allowed_act_fns(gelu)
    assert isinstance(fc, torch.nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def apply_clip(module, clip_list: Tuple[str, torch.Tensor]):
    for name, max_val in clip_list:
        layer: torch.nn.Linear = get_op_by_name(module, name)
        layer.to(get_best_device())
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()

def export_awq_gpt2_to_onnx(model: nn.Module,
                            save_path: str,
                            batch_size: int = 1,
                            sequence_length: int = 5):
    """
    导出GPT-2的onnx模型
    :param model:
    :param save_path:
    :param batch_size:
    :param sequence_length:
    :return:
    """
    # 1. 设置为评估模式
    model.eval()

    # 2. 准备示例输入
    dummy_input_ids: torch.Tensor = torch.randint(0, 6003, (batch_size, sequence_length))

    hidden_states: torch.Tensor = model.embedding(input_ids=dummy_input_ids)

    # 3. 定义输入输出名称
    input_names: List[str] = ["input_ids"]
    output_names: List[str] = ["output"]

    # 4. 导出模型
    torch.onnx.export(
        model,
        (hidden_states),  # 模型输入
        save_path,  # 保存路径
        input_names=input_names,  # 输入名称
        output_names=output_names,  # 输出名称
        do_constant_folding=True,  # 进行常量折叠优化
        opset_version=20,  # ONNX算子版本
        verbose=False,
        export_params=True,  # 导出模型参数
    )


