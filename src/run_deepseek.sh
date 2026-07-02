TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PYTHONPATH="$PWD" python ./scripts/infer_deepseek.py \
    --model /home/lzx/Models/DeepSeek-v2-lite-chat \
    --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
    --batch-size 1 \
    --beam-width 1 \
    --cpu-offload 0 \
    --sync-timing \
    --warmup 0 \
    --input-token-num 1024 \
    --output-token-num 32
    # --hit-source-log ./logs/hit_source_${TIMESTAMP}.jsonl \
    # --expert-schedule-log ./logs/scheduling_${TIMESTAMP}.jsonl \
    # --record-expert-schedule
