#!/bin/bash

# 1. 精准定位项目根目录
# 假设脚本在 scripts/某子文件夹/ 下，退两层回到根目录
# 直接进入你当前终端所在的目录
cd /mnt/d/NLSR-master/NLSR-master

# 2. 设置环境变量
export PYTHONPATH=$PWD
# 权重合并通常在 CPU 上即可完成，若显存充足也可指定 GPU
export CUDA_VISIBLE_DEVICES="" 

# 3. 实验参数
source_type="sft"
target_type="dpo"
alpha_all=(0.9) # 可添加更多值如 (0.1 0.3 0.5 0.7 0.9)

# 4. 循环执行融合逻辑
for alpha in "${alpha_all[@]}"; do
    echo "-----> Processing with alpha: ${alpha}..."

    # 运行融合脚本
    python3 ./weak_to_strong/expo-lora.py \
        --weak_model_path "./saves/lora/${source_type}/checkpoint-125-merged" \
        --moderate_model_path "./saves/lora/${target_type}" \
        --alpha "${alpha}" \
        --save_path "./saves/lora/expo_${target_type}_lora/${source_type}_to_${target_type}-alpha_${alpha}"
done