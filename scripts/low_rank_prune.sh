#!/bin/bash

# 1. 精准定位项目根目录
# 假设脚本在 scripts/某子文件夹/ 下，退两层回到项目根目录
# 直接进入你当前终端所在的目录
cd /mnt/d/NLSR-master/NLSR-master

# 2. 设置环境变量
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=1

# 3. 实验参数
prune_type="low_rank"
sparsity_ratios=(0.8)
alignment_types=("expo_dpo_lora") 
fusion_effects=("sft_to_dpo-alpha_0.9")

# 4. 执行稀疏化分析逻辑
# 修正了原脚本中变量 $i 未定义的问题
for sparsity_ratio in "${sparsity_ratios[@]}"
do
    # 使用索引来遍历，确保 alignment_types 和 fusion_effects 能对应上
    for (( i=0; i<${#alignment_types[@]}; i++ ))
    do
        alignment_name="${alignment_types[$i]}"
        echo "-----> sparsity_ratio: ${sparsity_ratio}..."
        echo "---> Alignment type: ${alignment_name}..."

        # 判断是否包含 expo，决定子路径
        if [[ "$alignment_name" == *"expo"* ]]; then
            modified_alignment_name="${alignment_name}/${fusion_effects[$i]}"
        else
            modified_alignment_name="${alignment_name}"
        fi

        # 运行 Python 脚本生成 Mask
        python3 ./prune_regions/sparsity_ratio_low.py \
            --rank_path "./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio}/rank_bottom_${sparsity_ratio}" \
            --output_dir "./saves/lora/prune_regions/${alignment_name}-${prune_type}-${sparsity_ratio}" \
            --sparsity_ratio "${sparsity_ratio}"
    done
done