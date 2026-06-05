#!/bin/bash
cd "$(dirname "$0")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PYTHONPATH="$PWD" python ./scripts/infer_deepseek.py \
    --model /mnt/g/Models/DeepSeek-v2-lite-chat \
    --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --batch-size 1 \
    --beam-width 1 \
    --cpu-offload 0 \
    --warmup 1 \
    --input-token-num 128 \
    --output-token-num 100 \
    --expert-schedule-log logs/schedule_${TIMESTAMP}.jsonl \
    --record-expert-schedule


    # --debug-runtime-state \
    # --profile-expert-executor \
    ## 开启后保留预热阶段的占位专家缓存
    #--preserve-warmup-cache \ 
