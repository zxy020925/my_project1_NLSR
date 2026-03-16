#!/bin/bash

# 1. 精准定位项目根目录
# 假设脚本在 scripts/ours/ 下，退两层回到项目根目录
# 直接进入你当前终端所在的目录
cd /mnt/d/NLSR-master/NLSR-master

# 2. 设置环境变量
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=0

# 3. 实验参数设置
dataset_name=sst2
alignment_name=expo_dpo_lora
region_method=low_rank
data_selected_all=("n1000_p0.01" "n1000_p0.05" "n1000_p0.1" "n1000_p0.2" "n1000_p0.3")
model_path="./saves/lora/realign/expo-adaptive_mask_replace-safe_lora"

sparsity_ratio=0.8
prune_rates=(0.5)
epsilon=0.2

# 4. 循环执行安全评估
for data_selected in "${data_selected_all[@]}"; do
    echo "-----> Running with data_selected=$data_selected"

    for prune_rate in "${prune_rates[@]}"; do
        echo "------> Running with prune_rate=$prune_rate"
        
        # 第一步：生成预测结果 (Prediction)
        python3 ./evaluation/poison/pred.py \
            --model_folder "./saves/lora/sft/checkpoint-125-merged" \
            --lora_folder "${model_path}/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}_prune_rate_${prune_rate}_epsilon_${epsilon}" \
            --instruction_path "BeaverTails" \
            --start 0 \
            --end 1000 \
            --output_path "./results/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}_prune_rate_${prune_rate}_epsilon_${epsilon}-safety.json"

        # 第二步：使用安全评估器进行评分 (Safety Scoring)
        # 注意：请确保 ./pretrained_model/beaver-dam-7b 路径下已下载好评估模型
        python3 ./evaluation/poison/eval_safety.py \
              --safety_evaluator_path "./pretrained_model/beaver-dam-7b" \
              --input_path "./results/lora/realign/expo-adaptive_mask_replace-safe_lora/${dataset_name}-${alignment_name}-${data_selected}-${region_method}/sparsity_ratio_${sparsity_ratio}_prune_rate_${prune_rate}_epsilon_${epsilon}-safety.json" \
              --add
    done
done