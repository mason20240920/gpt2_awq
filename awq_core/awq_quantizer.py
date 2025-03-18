#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : awq_quantizer.py
@Author  : Barry Allen
@Date    : 2025/3/17 17:18
@Desc    : AWQ量化核心
"""
import functools
import logging
from collections import defaultdict
from typing import List, Optional, Dict

from torch import nn, Tensor
import torch
from tqdm import tqdm

# from models.inter_mid_gpt2_model import InterMidGPT2Model
from utils.other_utils import read_testcases_to_awq, get_best_device, clear_memory, get_named_linears, \
    exclude_layers_to_not_quantize, sanitize_kwargs, get_op_name, apply_scale, apply_clip


class AwqQuantizer:
    def __init__(self,
                 inter_mid_model: nn.Module,
                 modules_to_not_convert: Optional[List[str]] = None,
                 apply_clip: bool = True,
                 n_parallel_ime_samples: Optional[int] = None,
                 max_ime_samples: int = 20,
                 max_ime_seq_len: int = 16,
                 max_chunk_memory: int = 1024 * 1024 * 1024):
        """
        初始化AWQ量化器
        根据论文，AWQ通过激活感知的方式来保护重要的权重通道，实现高效的低比特量化
        :param inter_mid_model: InterMidGPT2Model
        :param modules_to_not_convert: 不需要转换的模块列表
        :param apply_clip: 是否应用裁剪
        :param n_parallel_ime_samples: 并行校准样本数
        :param max_ime_samples:
        :param max_ime_seq_len:
        :param max_chunk_memory:
        """
        self.awq_model: nn.Module = inter_mid_model
        self.model: nn.Module = inter_mid_model
        self.w_bit: int = inter_mid_model.args.quant_bit()
        self.group_size: int = inter_mid_model.args.quant_block()  # 量化分组大小，默认128
        self.zeropoint:bool = not inter_mid_model.args.sym()  # 是否使用零点量化
        self.duo_scaling: bool = True  # 是否使用双重缩放
        self.apply_clip: bool = apply_clip  # 是否应用裁剪
        self.n_parallel_ime_samples: Optional[int] = n_parallel_ime_samples
        self.max_ime_samples: int = max_ime_samples  # 最大校准样本数
        self.max_ime_seq_len: int = max_ime_seq_len  # 最大序列长度
        self.max_chunk_memory: int = max_chunk_memory  # 最大分块内存
        self.modules_to_not_convert: List[str] = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        # 初始化量化数据
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_ime_samples, max_seq_len=self.max_ime_seq_len
        )

    def init_quant(self, n_samples=20, max_seq_len=16):
        """
        初始化AWQ量化过程
        :param n_samples: 校准数据集的样本数量，默认20
        :param max_seq_len: 最大序列长度，默认16
        :return:
        tuple: (modules, layer_kwargs, inps)
            - modules: 模型的各层模块
            - layer_kwargs: 层的额外参数(位置编码、注意力掩码等)
            - inps: 预处理后的输入嵌入
        """
        # 获取模型的所有层模块
        modules: nn.ModuleList = self.awq_model.blocks

        # 获取校验集
        samples:list[Tensor] = read_testcases_to_awq(n_samples=n_samples,
                                                     max_seq_len=max_seq_len)
        concat_sample: Tensor = torch.cat(samples, dim=0)  # 合并校验集

        self.model.seq_len = len(samples) * self.max_ime_seq_len  # 最大数据长度

        # 初始化输入列表和层参数字典
        inps: List= []
        layer_kwargs: Dict = {}
        # 创建inps
        # 设置模型序列相关参数
        # TODO: 一些参数设置

        # 获取最佳计算设备(CPU/GPU)
        best_device: torch.device = get_best_device()

        inps: torch.Tensor = self.model.embedding(input_ids=concat_sample).to(best_device)

        # 清理不需要的数据，优化内存使用
        # 这对于在边缘设备上运行很重要
        del samples
        clear_memory()
        return modules, layer_kwargs, inps

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        """
        伪量化张量
        论文中提出的权重量化方法，支持零点量化和对称量化两种模式
        :param w:
        :return:
        """
        org_w_shape = w.shape
        # 如果启用分组，将权重重组为分组形式
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        # 零点量化模式
        if self.zeropoint:
            max_val = w.amax(dim=1, keepdim=True)  # 获取每组最大值
            min_val = w.amin(dim=1, keepdim=True)  # 获取每组最小值
            offset = 1 << (self.w_bit - 1)  # 计算偏移量
            clip_max = offset - 1  # 裁剪上限
            clip_min = -offset  # 裁剪下限
            # 计算量化比例
            scales = (max_val - min_val) / (clip_max - clip_min)
            # 计算零点
            zeros = - torch.round(min_val / scales) + clip_min
            # 量化过程
            qw = torch.round(w / scales) + zeros
            qw = torch.clamp(qw, clip_min, clip_max)
            # 反量化
            w = (qw - zeros) * scales
            zeros = min_val.view(org_w_shape[0], -1)
        else:
            # 对称量化模式
            abs_max = w.abs().amax(dim=1, keepdim=True)  # 获取绝对值最大值
            offset = 1 << (self.w_bit - 1)  # 计算偏移量
            clip_max = offset - 1  # 裁剪上限
            clip_min = -clip_max  # 裁剪下限
            scales = abs_max / clip_max  # 计算量化比例
            # 量化并裁剪
            w = torch.clamp(torch.round(w / scales), clip_min, clip_max) * scales
            zeros = None

        # 确保没有NaN值
        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # 调整形状
        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros

    def _get_input_feat(self,
                        layer: nn.Module,
                        named_linears: Dict[str, nn.Linear]) -> dict:
        """
        获取所有线性层的输入特征
        :param layer:  需要处理的网络层
        :param named_linears:  命名的线性层字典 {name: linear_layer}
        :return:  每个线性层的输入特征字典 {name: input_features}
        """

        # 首先，获取所有线性层的输入特征
        # 定义钩子函数，用于捕获前向传播时的输入特征
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]  # 获取输入元组的第一个元素（实际输入）
            x = x.detach().cpu()  # 分离计算图并移至CPU，避免内存占用
            feat_dict[name].append(x)  # 将输入特征存储到对应层的列表中

        # 初始化存储输入特征的默认字典
        input_feat = defaultdict(list)
        handles = []  # 存储钩子句柄，用于后续移除

        # 为每个线性层注册前向钩子
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        # 将输入移动到当前层所在的设备（处理多GPU情况）
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        # 处理模块的关键字参数，确保兼容性（用于trust_remote_code模型）
        module_kwargs = sanitize_kwargs(self.module_kwargs, layer)

        # 执行前向传播，触发钩子收集输入特征
        self.inps = self._module_forward(self.inps, layer, module_kwargs)

        # 移除所有钩子，避免内存泄漏
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        # 将每个层收集到的输入特征列表拼接成单个张量
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    @torch.no_grad()
    def _module_forward(self,
                        x: torch.Tensor,
                        module: torch.nn.Module,
                        module_kwargs: Dict) -> torch.Tensor:
        """
        模块前向传播函数，用于AWQ校准阶段

        论文中提到，AWQ需要分析模型中间层的激活值分布来确定重要的权重通道
        为了高效处理大规模校准数据，实现了两种模式：
        1. 一次性处理所有样本（当内存充足时）
        2. 分批处理模式（当需要处理大量校准数据时）
        :param x: 输入张量，包含校准数据
        :param module:  需要进行前向传播的模块
        :param module_kwargs: 模块的额外参数
        :return:
        """
        if self.n_parallel_ime_samples is None:
            # 模式1：一次性处理所有校准样本
            # 适用于数据量较小或显存充足的情况
            # print(module, x, module_kwargs); exit(0)
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                # 某些模块（如Transformer）可能返回多个输出
                # 我们只需要主要输出（通常是第一个元素）
                module_output = module_output[0]
        else:
            # 模式2：分批处理校准样本
            # 论文中提到的内存优化策略，避免OOM（显存溢出）
            # but only n_parallel_calib_samples at a time
            module_output = []
            # 将输入数据划分为大小为n_parallel_calib_samples的批次
            partitioned_inputs = torch.split(x, self.n_parallel_ime_samples)

            # 逐批处理数据
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                # 将结果移至CPU内存，进一步降低GPU内存占用
                module_output.append(partial_output.cpu())

            # 合并所有批次的结果
            module_output = torch.cat(module_output, dim=0)

        return module_output

    def quantize(self):
        """
        量化推理
        :return:
        """
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device: torch.device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                best_device = get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            self.inps = self.inps.to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears:Dict[str, nn.Linear] = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config = [dict(
                prev_op=self.modules[i].input_rmsnorm,
                layers=[
                    self.modules[i].self_attn.c_attn,
                ],
                inp=input_feat["self_attn.c_attn"],
                module2inspect=self.modules[i].self_attn,
                kwargs=self.module_kwargs,
            )]
            # q, k, v proj
            # c_proj
            if self.modules[i].self_attn.c_attn.weight.shape == self.modules[i].self_attn.c_proj.weight.shape:
                module_config.append(
                    dict(
                        prev_op=self.modules[i].self_attn.c_attn,
                        layers=[self.modules[i].self_attn.c_proj],
                        inp=input_feat["self_attn.c_proj"],
                    )
                )
            # mlp fc
            module_config.append(
                dict(
                    prev_op=self.modules[i].post_attn_rmsnorm,
                    layers=[self.modules[i].mlp.c_fc],
                    inp=input_feat["mlp.c_fc"],
                    module2inspect=self.modules[i].mlp,
                )
            )
            # mlp proj
            module_config.append(
                dict(
                    prev_op=self.modules[i].mlp.c_fc,
                    layers=[self.modules[i].mlp.c_proj],
                    inp=input_feat["mlp.c_proj"],
                )
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]

            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)


            clear_memory()

    @torch.no_grad()
    def _search_best_scale(self,
                           module,
                           prev_op,
                           layers: List[torch.nn.Linear],
                           inp: torch.Tensor,
                           module2inspect=None,
                           kwargs={}):
        """
        搜索最优缩放因子的函数

        这是AWQ论文中提出的激活感知量化(Activation-aware Weight Quantization)的核心实现。
        该方法通过分析权重和激活值的分布特征，为每个通道确定最优的量化缩放因子。

        :param module:  要量化的模块
        :param prev_op:  前一个算子
        :param layers:  需要量化的线性层列表
        :param inp:  输入张量（校准数据）
        :param module2inspect:  需要检查的特定模块
        :param kwargs:  额外参数
        :return:
        """
        # 确保在只有一个层时正确设置module2inspect
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device 确保输入数据在正确的设备上
        # parameters(): 返回模块的参数(权重和偏置值)的生成器
        # next作用: 获取生成器中的第一个参数(不需要遍历所有参数，只取第一个即可)
        # to作用: 转移参数的设备的名称(mps, cpu, cuda)
        inp = inp.to(next(module2inspect.parameters()).device)

        # [步骤 1]: 计算归一化权重的逐通道平均值
        # 论文3.2节中描述的权重重要性评估
        # All layer weights are concatted together
        # eg:  GPT2一个投影层的权重形状
        # c_attn.weight: [2304, 768]
        #
        # # 使用 dim=0 合并后
        # combined_weight: [2304, 768]  # 2304 = 2304 * 1
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape  # 获取合并权重的形状(192, 256)
        # 将权重重组为量化组的形式，对应论文中的group-wise量化策略, 如果为0就走channel-wise
        if self.group_size > 0:
            weight = weight.view(-1, self.group_size)
        # 计算每个量化组内权重的相对重要性
        # 这对应论文中识别重要通道的步骤
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        # eg:
        #   |每组的weight绝对值| / (|每组weight绝对值的最大值| + 1e-6 >> 防止值的变化)
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        #  返回原始的权重的的形状(192, 256)
        w_scale = w_scale.view(org_shape)
        # 计算每个输出通道的平均重要性
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [步骤 2]: 分块计算输入激活值的逐通道平均值
        # 这实现了论文中的内存效率优化策略
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)  # 行数
        num_channels = inp_flat.size(1)  # 列数
        element_size_bytes = inp_flat.element_size() * 2  # multiplied by 2 for FP32 # FP32的内存占用

        # 基于最大内存限制动态计算分块大小
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)  # 找到最小内存占用

        # 使用FP32进行累加以提高精度
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)

        # 分块处理以避免内存溢出
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)  # 计算平均值
        clear_memory(x_sum)

        # [步骤 3]: 计算模块的输出
        # 获取FP16精度下的参考输出
        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = sanitize_kwargs(kwargs, module2inspect)  # 清理并过滤输入参数，移除模块前向传播中不支持的参数
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)  # 获取fp16的输出值

        # [步骤 4]: 计算损失并确定最优缩放因子
        # 这对应论文中的缩放因子优化过程
        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        # 返回操作名称和最优缩放因子
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
            self,
            x: torch.Tensor,
            w_mean: torch.Tensor,
            x_mean: torch.Tensor,
            module2inspect: torch.nn.Module,
            linears2scale: List[torch.nn.Linear],
            fp16_output: torch.Tensor,
            kwargs=None):
        """
        计算最优缩放因子以最小化量化误差

        论文中的目标函数:
        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        其中:
        - Q: 权重量化函数
        - X: 校准数据集的输入
        - W: FP16格式的原始权重
        - s: 每个通道的缩放因子
        :param x:  校准数据集的输入
        :param w_mean: 权重均值
        :param x_mean: 激活均值
        :param module2inspect: 需要检查的模块
        :param linears2scale: 需要缩放的线性层列表
        :param fp16_output: FP16下的原始输出
        :param kwargs:
        :return:
        """
        # 网格搜索的粒度，论文中提到使用20个点来搜索最优α
        # We used a grid size of 20 to search for the
        # optimal α in Equation 5
        if kwargs is None:
            kwargs = {}
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        # 将数据移到正确的设备上并重塑
        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        # 保存原始权重以便恢复
        ord_weights = []
        for fc in linears2scale:
            ord_weights.append(fc.weight.data.clone())

        # 网格搜索最优α值
        for ratio in range(n_grid):
            # α在[0,1]范围内变化（这段代码实际上是在实现网格搜索，将[0,1)区间均匀分成20份）
            # α值的搜索空间：[0.00, 0.05, 0.10, ..., 0.95]
            ratio = ratio / n_grid

            # 计算缩放因子 s
            # 论文中提到 s = sX^α，其中sX是激活的平均幅度
            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                # 同时考虑激活和权重分布
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                # 只考虑激活分布
                # clamp(min=1e-4): 小于1e-4的值都会被设置为1e-4
                # view(-1): 重塑为1维
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)

            # 归一化缩放因子
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            # 处理数值问题
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            # 执行 Q(W * s) 操作
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                        self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X 计算量化后的输出
            int_w_output = self._module_forward(x, module2inspect, kwargs)

            # compute mean squared error (L2 norm) 计算L2范数损失
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()

            for fc, ord_weight in zip(linears2scale, ord_weights):
                fc.weight.data = ord_weight.clone()

        del ord_weights

        # 错误检查
        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
            self,
            fp16_output: torch.Tensor,
            int_w_output: torch.Tensor,
            device: torch.device):
        """
        计算量化误差，对应论文公式(4)中的L2损失:
         L(s) = ||Q(W·diag(s))(diag(s)^(-1)·X) - WX||
        :param fp16_output:  原始FP16权重输出 W·X
        :param int_w_output: 量化后的输出 Q(W·s)(X/s)
        :param device:
        :return:
        """
        # 将输出展平为一维向量便于计算
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        # 动态计算分块大小，控制内存使用
        # 内存限制为max_chunk_memory (通常1GB)
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        # 将计算分块处理，避免OOM
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # 分块计算L2损失
        # 对应论文中的误差计算：
        # Err(Q(w·s)(x/s)) = Δ'·RoundErr(ws/Δ')·x/s
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            # 计算每块的均方误差
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # 归一化损失，得到平均每个元素的误差
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv", "c_"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(self,
                           w: torch.Tensor,
                           input_feat: torch.Tensor,
                           n_grid=20,
                           max_shrink=0.5,
                           n_sample_token=512):
        """
        计算最佳裁剪值，实现AWQ中的激活感知缩放。
        论文中提到，通过缩放重要通道可以减少量化误差，
        该函数通过网格搜索找到最优缩放因子。
        :param w:  输入权重
        :param input_feat:  输入特征/激活值
        :param n_grid:  网格搜索的粒度，用于寻找最优α
        :param max_shrink:  最大收缩比例，防止过度缩放
        :param n_sample_token:  采样token数，用于减少计算量
        :return:
        """
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        # 将权重和输入特征重组为分组结构
        # w: [输出通道, 输入通道] -> [输出通道, 1, 组数, 组大小]
        # input_feat: [token数, 输入通道] -> [1, token数, 组数, 组大小]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        # 对输入特征进行下采样，减少计算量
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]

        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # 设置批处理大小，防止OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        # 按批次处理输出通道
        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

            # 计算原始权重的最大绝对值
            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            # 初始化最佳裁剪值和最小误差
            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)


