TIMESTAMP=$(date +%Y%m%d_%H%M%S)

INPUT_TOKEN_NUM=(256)
CPU_OFFLOAD=(0 1 2)

for cpu_offload in "${CPU_OFFLOAD[@]}"; do
    for input_token_num in "${INPUT_TOKEN_NUM[@]}"; do
         PYTHONPATH="$PWD" python ./scripts/infer_deepseek.py \
            --model /home/lzx/Models/DeepSeek-v2-lite-chat \
            --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
            --batch-size 1 \
            --beam-width 1 \
            --cpu-offload $cpu_offload \
            --warmup 0 \
            --input-token-num $input_token_num \
            --output-token-num 128 \
            --expert-schedule-log logs/schedule_${TIMESTAMP}_cpu_offload_${cpu_offload}_input_token_num_${input_token_num}.jsonl \
            --record-expert-schedule \
            --hit-source-log logs/hit_source_${TIMESTAMP}_cpu_offload_${cpu_offload}_input_token_num_${input_token_num}.jsonl \
            > logs/infer_${TIMESTAMP}_cpu_offload_${cpu_offload}_input_token_num_${input_token_num}.log 2>&1
    done
done

# PYTHONPATH="$PWD" python ./scripts/infer_deepseek.py \
#     --model /mnt/g/Models/DeepSeek-v2-lite-chat \
#     --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
#     --batch-size 1 \
#     --beam-width 1 \
#     --cpu-offload 0 \
#     --warmup 0 \
#     --input-token-num 2048 \
#     --output-token-num 128 \
#     --expert-schedule-log logs/schedule_${TIMESTAMP}.jsonl \
#     --record-expert-schedule \
#     --hit-source-log logs/hit_source_${TIMESTAMP}.jsonl > logs/infer_${TIMESTAMP}.log 2>&1
#     # --debug-runtime-state \
#     # --profile-expert-executor \
#     ## 开启后保留预热阶段的占位专家缓存
#     #--preserve-warmup-cache \ 
