#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project : gpt2_awq
@File    : main.py
@Author  : Barry Allen
@Date    : 2025/3/12 19:33
@Desc    : 基于GPT-2的AWQ量化模型训练
"""
import argparse
import os
import warnings

import torch

from models.ime_args_model import IMEArgsModel

warnings.filterwarnings('ignore', category=UserWarning)
from transformers import GPT2Config

from models.ime_gpt2_lm_head_model import IMEGPT2LMHeadModel
from models.inter_mid_gpt2_model import InterMidGPT2Model

def loaded_trained_model(arguments: argparse.Namespace):
    """
    加载训练好的模型(epoch 150)的模型
    :return:
    """
    cur_model_path = os.path.join(arguments.output_dir, "model_901")
    model: IMEGPT2LMHeadModel = IMEGPT2LMHeadModel.from_pretrained(cur_model_path)

    # 4. 将模型设置为评估模式（如果只是推理）
    model.eval()

    input_ids:torch.LongTensor = torch.tensor([0, 3, 48, 44], dtype=torch.long)
    # 5. 进行推理
    output = model(input_ids)
    values, indices = torch.max(output, dim=1)
    print(indices)

def load_transformer_model(arguments: argparse.Namespace):
    cur_model_path = os.path.join(args.output_dir, "model_901")
    args_model: IMEArgsModel = IMEArgsModel(model_name="Qwen2.5/0.5B", dst_path="awq", awq=True, quant_block=0)
    model: IMEGPT2LMHeadModel = IMEGPT2LMHeadModel.from_pretrained(cur_model_path)
    ime_mid_gpt2_model: InterMidGPT2Model = InterMidGPT2Model(config=model_config, model=model, args=args_model)
    # 4. 将模型设置为评估模式（如果只是推理）
    ime_mid_gpt2_model.eval()
    ime_mid_gpt2_model.export()
    ime_mid_gpt2_model.to(device="cpu")
    # export_awq_gpt2_to_onnx(model=ime_mid_gpt2_model, save_path="ime_awq_gpt.onnx")


    # ime_mid_gpt2_model.to(device="cpu")
    # input_ids: torch.LongTensor = torch.tensor([0, 3, 48, 44], dtype=torch.long)
    # # 5. 进行推理
    #
    # # 6. 位置编码的embedding
    # # 4. 确定输入的位置编码
    # hidden_states: torch.Tensor = ime_mid_gpt2_model.embedding(input_ids=input_ids)
    # output = ime_mid_gpt2_model(hidden_states)
    # output = output.view(-1, 768)
    # output = ime_mid_gpt2_model.lm_(output)
    # values, indices = torch.max(output, dim=1)
    # print(indices)
    # values, indices = torch.max(output, dim=1)
    # print(indices)

if __name__ == '__main__':
    # 1. 命令行读取
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0, 1, 2, 3', required=False, help='设置使用哪些显卡')
    parser.add_argument("--model_config", default="/Users/mason/Desktop/Desktop/PythonProjects/ime_gpt_2/my_model_config/config.json", type=str, required=False,
                        help="选择模型参数")
    parser.add_argument("--epochs", default=500, type=int, required=False, help="训练循环")
    parser.add_argument("--batch_size", default=5, type=int, required=False, help="训练的 batch size")
    parser.add_argument("--lr", default=1.5e-4, type=float, required=False, help="学习率")
    parser.add_argument("--warmup_steps", default=2000, type=int, required=False, help="warmup 步数")
    parser.add_argument("--log_step", default=1000, type=int, required=False,
                        help="多少次汇报一次loss, 设置为gradient accumulation的整数倍")
    parser.add_argument("--stride", default=768, type=int, required=False, help="训练时取训练数据的窗口步长")
    parser.add_argument("--gradient_accumulation", default=1, type=int, required=False, help="梯度积累")
    parser.add_argument("--fp16", action='store_true', required=False, help="混合精度")
    parser.add_argument("--fp16_opt_level", default='O1', required=False)
    parser.add_argument("--max_grad_norm", default=0.5, type=float, required=False)
    parser.add_argument("--num_pieces", default=100, type=int, required=False, help="将训练语料分成多少份")
    parser.add_argument("--min_length", default=0, type=int, required=False, help="最短收录文章长度")
    parser.add_argument("--output_dir", default="/Users/mason/Desktop/Desktop/PythonProjects/ime_gpt_2/output/", type=str, required=False, help="模型输出路径")
    parser.add_argument("--pretrained_model", default="", type=str, required=False, help="模型训练起点路径")
    parser.add_argument("--writer_dir", default="tensorboard_summary", type=str, required=False, help="Tensorboard路径")
    parser.add_argument("--segment", action="store_true", help="中文以词为单位")
    parser.add_argument("--bpe_token", action="store_true", help="subword")
    parser.add_argument("--encoder_json", default="tokenization/encoder.json", type=str, help="encoder.json")
    parser.add_argument("--vocab_bpe", default="tokenization/vocab.bpe", type=str, help="vocab.bpe")
    args = parser.parse_args()
    # print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config: GPT2Config = GPT2Config.from_json_file(args.model_config)
    # print('config:\n' + model_config.to_json_string())

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("using device:", device)
    # loaded_trained_model(arguments=args)
    load_transformer_model(arguments=args)
    # train(arguments=args, config=model_config, tensor_device=device)
    # loaded_trained_model(arguments=args)
    # # 1. 获取GPT-2的配置
    # gpt2_config: GPT2Config = GPT2Config(
    #     n_head=12,
    #     vocab_size=6003,
    #     n_embd=768,
    #     use_cache=False,
    #     n_layer=1,
    #     n_inner=3072,
    # )
    # model: IMEGPT2LMHeadModel = IMEGPT2LMHeadModel(config=gpt2_config)
    # model.eval()
    #










