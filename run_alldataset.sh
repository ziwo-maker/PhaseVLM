#!/bin/bash

# 设置环境变量
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置工作目录（根据你的实际路径修改）
cd /home/code/LLMTSCS/


# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_open_LLM_with_vllm.py     --llm_model gemma-3-27B     --llm_path /home/data/model/gemma-3-27B/     --dataset jinan     --traffic_file anon_3_4_jinan_real.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_open_LLM_with_vllm.py     --llm_model gemma-3-27B     --llm_path /home/data/model/gemma-3-27B/     --dataset newyork_28x7     --traffic_file anon_28_7_newyork_real_triple.json

# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_random.py        --dataset jinan     --traffic_file anon_3_4_jinan_synthetic_24h_6000.json


# CUDA_VISIBLE_DEVICES=0,1,2,3 python run_random.py        --dataset jinan     --traffic_file anon_3_4_jinan_real.json

# 定义通用参数
LLM_MODEL="gemma-3-27B"
LLM_PATH="/home/data/model/gemma-3-27B/"
PROJ_NAME="Gemini-27B-alldata"

echo "开始运行实验..."
echo "================================"

declare -A SEEN_RUNS

run_dataset() {
    local dataset="$1"
    local traffic_file="$2"
    local key="${dataset}::${traffic_file}"

    if [[ -n "${SEEN_RUNS[$key]}" ]]; then
        echo "跳过重复运行: ${dataset} / ${traffic_file}"
        return 0
    fi
    SEEN_RUNS[$key]=1

    echo "运行 ${dataset} 数据集 (${traffic_file})..."
    python run_open_LLM_with_vllm.py \
        --llm_model "$LLM_MODEL" \
        --llm_path "$LLM_PATH" \
        --dataset "$dataset" \
        --traffic_file "$traffic_file" \
        --proj_name "$PROJ_NAME"

    if [ $? -eq 0 ]; then
        echo "${dataset} 数据集 (${traffic_file}) 完成 ✓"
    else
        echo "${dataset} 数据集 (${traffic_file}) 失败 ✗"
    fi
}

# 运行 jinan 数据集
run_dataset "jinan" "anon_3_4_jinan_real_2000.json"
run_dataset "jinan" "anon_3_4_jinan_real_2500.json"
run_dataset "jinan" "anon_3_4_jinan_real.json"
run_dataset "jinan" "anon_3_4_jinan_synthetic_24h_6000.json"
run_dataset "jinan" "anon_3_4_jinan_synthetic_24000_60min.json"

# 运行 hangzhou 数据集
run_dataset "hangzhou" "anon_4_4_hangzhou_real_5816.json"
run_dataset "hangzhou" "anon_4_4_hangzhou_real.json"
run_dataset "hangzhou" "anon_4_4_hangzhou_synthetic_24000_60min.json"

# 运行 newyork_28x7 数据集
run_dataset "newyork_28x7" "anon_28_7_newyork_real_double.json"
run_dataset "newyork_28x7" "anon_28_7_newyork_real_triple.json"
