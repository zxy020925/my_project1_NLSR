#!/bin/bash

# 1. 定位到项目根目录 (假设脚本在 scripts/safe_lora/ 下)
# 直接进入你当前终端所在的目录
cd /mnt/d/NLSR-master/NLSR-master

# 2. 设置环境变量
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=1

# 3. 实验参数
dataset_name="sst2"
dataset_selected_all=("n1000_p0.01")
region_method="low_rank" 

sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2
fusion_effect="sft_to_dpo-alpha_0.9"
alignment_name="expo_dpo_lora"

# 4. 循环执行重构逻辑
for dataset_selected in "${dataset_selected_all[@]}"; do
    echo "-----> Running with dataset_selected=$dataset_selected"

    for prune_rate in "${prune_rates[@]}"; do
        echo "-----> Running with prune_rate=$prune_rate"
        
        # 使用 python3 运行重构脚本
        python3 ./safe_lora/identify_realign.py \
             --model_path "./saves/lora/sft/checkpoint-125-merged" \
             --lora_path "./saves/lora/baselines/poison_ratio/aligned-finetune-${dataset_selected}" \
             --aligned_path "./saves/lora/${alignment_name}/${fusion_effect}" \
             --mask_path "./saves/lora/prune_regions/${alignment_name}-${region_method}-${sparsity_ratio}/mask_bottom_${sparsity_ratio}.pt" \
             --sparsity_ratio "${sparsity_ratio}" \
             --prune_rate "${prune_rate}" \
             --epsilon "${epsilon}" \
             --realign_type "adaptive_mask_replace" \
             --output_path "./saves/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${dataset_selected}-${region_method}"
    done
done