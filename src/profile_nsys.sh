#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

OUTPUT_DIR="$(realpath "${1:-./nsys_reports}")"
mkdir -p "$OUTPUT_DIR"
TRACE_NAME="deepseek_$(date +%Y%m%d_%H%M%S)"
TRACE_PATH="$OUTPUT_DIR/$TRACE_NAME"

PYTHONPATH="$PWD" nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=cpu \
    --cuda-memory-usage=true \
    --output="$TRACE_PATH" \
    --force-overwrite=true \
    -- python ./scripts/infer_deepseek.py \
        --model /mnt/g/Models/DeepSeek-v2-lite-chat \
        --dataset /home/lzx/program/moe_code/datasets/sharegpt_v3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
        --batch-size 1 \
        --beam-width 1 \
        --cpu-offload 1 \
        --warmup 0 \
        --input-token-num 128 \
        --output-token-num 10

if [[ -f "$TRACE_PATH.nsys-rep" ]]; then
    echo "Report saved to: $TRACE_PATH.nsys-rep"
    echo "Open with: nsys-ui $TRACE_PATH.nsys-rep"
else
    echo "ERROR: Nsight Systems did not generate $TRACE_PATH.nsys-rep" >&2
    echo "Files in output dir:" >&2
    ls -lah "$OUTPUT_DIR" >&2
    exit 1
fi
