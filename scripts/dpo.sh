#!/bin/bash

# 1. 定位到脚本所在目录的上一级（即项目根目录）
# 假设你的脚本在 scripts/ 文件夹下
cd "$(dirname "$0")/.."

# 2. 设置环境变量
export WANDB_PROJECT="assessing_safety"
# 设置 PYTHONPATH 为当前工作目录，确保 Python 能识别项目内的模块
export PYTHONPATH=$PWD
# 如果你只有 1 张显卡，建议改成 export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 3. 运行 DPO 训练
# 使用 python3 明确调用 WSL 环境中的 Python
python3 main.py train config/DPO.yaml