#!/bin/bash

# 1. 强制进入项目根目录 (WSL 路径)
# 只要你在项目根目录下运行这个脚本，下面的逻辑就很稳
cd "$(dirname "$0")/.."

# 2. 设置环境变量
export WANDB_PROJECT="assessing_safety"
# 设置 PYTHONPATH 为当前目录，确保 python 能找到 main.py
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3. 运行训练脚本
# 使用 python3 确保调用的是 Linux 环境的 python
python3 main.py train config/SFT.yaml

# 4. 设置模型路径
# 请确保 D:\NLSR-master\NLSR-master\pretrained_model 下确实有 Llama 模型
model_path="./pretrained_model/Meta-Llama-3-8B"
save_path="./saves/lora/sft/checkpoint-125"

# 5. 合并模型权重
python3 export_merged.py \
    --org_model_path "$model_path" \
    --lora_path "$save_path" \
    --save_path "$save_path-merged"