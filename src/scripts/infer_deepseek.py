import argparse
import json
import os
import random

import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity

from model.deepseek import mDeepSeek


def load_prompts_from_dataset(dataset_path, batch_size, seed=42):
    rng = random.Random(seed)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_prompts = []
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "human":
                all_prompts.append(turn["value"])
                break
    return rng.sample(all_prompts, min(batch_size, len(all_prompts)))


def print_runtime_state(model, label):
    placement = model.placeholder_manager.snapshot()
    executor = model.expert_executor
    print(
        f"[{label}] "
        f"placeholder_resident={len(placement.placeholder_resident)} "
        f"free_placeholders={placement.free_placeholders} "
        f"loading={len(placement.loading)} "
        f"preload_request={executor.preload_request_count} "
        f"preload_success={executor.preload_success_count} "
        f"preload_skip={executor.preload_skip_count} "
        f"preload_hit={executor.preload_hit_count} "
        f"eviction={model.placeholder_manager.eviction_count}"
    )


def print_expert_timing(model):
    lines = model.expert_executor.format_timing_summary()
    if not lines:
        print("[expert-timing] no expert executor timing recorded")
        return
    print("[expert-timing] wall_share uses layer execute wall time; shares can sum above 100% when tasks overlap")
    for line in lines:
        print(line)

def print_placeholder_hit_rate(model):
    print(f"Placeholder resident hits: {model.placeholder_manager.placeholder_resident_hit_num}")
    print(f"Static GPU resident hits: {model.placeholder_manager.static_gpu_resdient_hit_num}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to DeepSeek model.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0: execute at GPU (baseline), 1: Scope strategy, 2: Fiddler strategy.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="Please tell me a joke.",
        help="Input text to generate (ignored if --dataset is set).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to ShareGPT JSON dataset. Overrides --input with random samples.",
    )
    parser.add_argument(
        "--output-token-num",
        type=int,
        default=20,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--input-token-num",
        type=int,
        default=None,
        help="Maximum number of input tokens to keep before generation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup generate runs before the measured run.",
    )
    parser.add_argument(
        "--preserve-warmup-cache",
        action="store_true",
        help="Keep dynamic placeholder cache populated by warmup runs.",
    )
    parser.add_argument(
        "--debug-runtime-state",
        action="store_true",
        help="Print placeholder and preload counters around warmup and measurement.",
    )
    parser.add_argument(
        "--profile-expert-executor",
        action="store_true",
        help="Print per-layer GPU/CPU/preload timing. Adds CUDA synchronization overhead.",
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam search width.")
    parser.add_argument(
        "--profile-torch",
        action="store_true",
        help="Enable torch.profiler and export Chrome trace.",
    )
    parser.add_argument(
        "--profile-torch-dir",
        type=str,
        default="./logs",
        help="Directory to save torch.profiler output.",
    )
    parser.add_argument(
        "--profile-decode-steps",
        type=int,
        default=1,
        help="Number of decode steps to profile after prefill (default: 1).",
    )

    args = parser.parse_args()
    model = mDeepSeek(args)

    if args.dataset:
        input_text = load_prompts_from_dataset(args.dataset, args.batch_size)
    else:
        input_text = args.input

    if args.debug_runtime_state:
        print_runtime_state(model, "before-warmup")

    for i in range(args.warmup):
        print(f"warmup {i + 1}/{args.warmup}")
        model.generate(
            input_text,
            output_token=args.output_token_num,
            input_token=args.input_token_num,
        )
        if args.debug_runtime_state:
            print_runtime_state(model, f"after-warmup-{i + 1}")

    if args.warmup > 0 and not args.preserve_warmup_cache:
        model.reset_runtime_state(clear_placeholders=True)
        if args.debug_runtime_state:
            print_runtime_state(model, "after-warmup-clean")

    profiler_ctx = None
    if args.profile_torch:
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
        )
        print(f"[profiler] torch.profiler created, output dir: {args.profile_torch_dir}")

    nvtx.range_push("inference")
    prefill_time, decode_time, hit_rate = model.generate(
        input_text,
        output_token=args.output_token_num,
        input_token=args.input_token_num,
        profiler=profiler_ctx,
        profile_decode_steps=args.profile_decode_steps if args.profile_torch else 0,
    )
    nvtx.range_pop()

    if profiler_ctx is not None:
        os.makedirs(args.profile_torch_dir, exist_ok=True)
        trace_path = os.path.join(args.profile_torch_dir, "trace.json")
        profiler_ctx.export_chrome_trace(trace_path)
        print(profiler_ctx.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        print(f"[profiler] Chrome trace saved to: {trace_path}")

    # Print runtime state and expert executor timing if requested
    if args.debug_runtime_state:
        print_runtime_state(model, "after-measure")
    if args.profile_expert_executor:
        print_expert_timing(model)
        print_placeholder_hit_rate(model)
        
    print(
        f"prefill_time: {prefill_time:.4f}, decode_time: {decode_time:.4f}, hit_rate: {hit_rate:.4f}"
    )
    if args.input_token_num is not None:
        print("tokens per second (prefill):", args.input_token_num / prefill_time)
    print("tokens per second (decode):", args.output_token_num / decode_time)
