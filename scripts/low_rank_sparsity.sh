#!/bin/bash

# 1. 精准定位项目根目录 (假设脚本在 scripts/某子目录下)
# 直接进入你当前终端所在的目录
cd /mnt/d/NLSR-master/NLSR-master

# 2. 设置环境变量
export PYTHONPATH=$PWD
# 建议先设置一个总的可见设备
export CUDA_VISIBLE_DEVICES=0,1

# 3. 实验参数
prune_type="low_rank"
sparsity_ratios=(0.8)
alignment_type=("expo_dpo_lora")
fusion_effects=("sft_to_dpo-alpha_0.9")

# 4. 执行识别逻辑
for sparsity_ratio in "${sparsity_ratios[@]}"
do
    for i in "${!alignment_type[@]}"
    do
        echo "-----> sparsity_ratio: ${sparsity_ratio}..."
        echo "---> Alignment type: ${alignment_type[$i]}..."

        alignment_name="${alignment_type[$i]}"
        
        # 初始赋值，防止非 expo 模式下变量为空
        modified_alignment_name="${alignment_name}"
        
        if [[ "$alignment_name" == *"expo"* ]]; then
            modified_alignment_name="${alignment_name}/${fusion_effects[$i]}"
        fi

        # 运行识别脚本
        # 显式指定 python3
        python3 ./prune_regions/identify_neurons_or_ranks.py \
             --model_path "./saves/lora/sft/checkpoint-125-merged" \
             --lora_path "./saves/lora/${modified_alignment_name}" \
             --sparsity_ratio "${sparsity_ratio}" \
             --prune_method "${prune_type}" \
             --data_path "./LLaMA_Factory/data/safety/prune_regions/${alignment_name}-safety_regions-filtered.json" \
             --output_dir "./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio}" \
             --save_mask \
             --nsamples 2000
    done
done