import argparse
import datetime
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


def print_executor_hit_sources(model, hit_source_log=None):
    ex = model.expert_executor
    total_gpu_hits = ex.static_gpu_hit_count + ex.placeholder_hit_count + ex.ondemand_load_count
    total_gpu_tokens = ex.static_gpu_hit_tokens + ex.placeholder_hit_tokens + ex.ondemand_load_tokens
    placeholder_hit_rate = ex.placeholder_hit_count / max(total_gpu_hits, 1)
    preload_of_placeholder_rate = ex.preload_hit_count / max(ex.placeholder_hit_count, 1)
    preload_of_success_rate = ex.preload_hit_count / max(ex.preload_success_count, 1)
    preload_of_request_rate = ex.preload_hit_count / max(ex.preload_request_count, 1)
    print(f"[hit-source] static_gpu_hit={ex.static_gpu_hit_count} (tokens={ex.static_gpu_hit_tokens})")
    print(f"[hit-source] placeholder_hit={ex.placeholder_hit_count} (tokens={ex.placeholder_hit_tokens})")
    print(f"[hit-source] ondemand_load={ex.ondemand_load_count} (tokens={ex.ondemand_load_tokens})")
    print(f"[hit-source] preload_hit={ex.preload_hit_count} (of placeholder hits)")
    print(f"[hit-source] total_gpu_experts={total_gpu_hits} (tokens={total_gpu_tokens})")
    print(f"[hit-source] placeholder_hit_rate={placeholder_hit_rate:.4f} (placeholder / total_gpu)")
    print(f"[hit-source] preload_hit/placeholder={preload_of_placeholder_rate:.4f}")
    print(f"[hit-source] preload_hit/preload_success={preload_of_success_rate:.4f}")
    print(f"[hit-source] preload_hit/preload_request={preload_of_request_rate:.4f}")

    if hit_source_log:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "static_gpu_hit_count": ex.static_gpu_hit_count,
            "static_gpu_hit_tokens": ex.static_gpu_hit_tokens,
            "placeholder_hit_count": ex.placeholder_hit_count,
            "placeholder_hit_tokens": ex.placeholder_hit_tokens,
            "ondemand_load_count": ex.ondemand_load_count,
            "ondemand_load_tokens": ex.ondemand_load_tokens,
            "preload_hit_count": ex.preload_hit_count,
            "preload_request_count": ex.preload_request_count,
            "preload_success_count": ex.preload_success_count,
            "preload_skip_count": ex.preload_skip_count,
            "total_gpu_hits": total_gpu_hits,
            "total_gpu_tokens": total_gpu_tokens,
            "placeholder_hit_rate": placeholder_hit_rate,
            "preload_of_placeholder_rate": preload_of_placeholder_rate,
            "preload_of_success_rate": preload_of_success_rate,
            "preload_of_request_rate": preload_of_request_rate,
        }
        os.makedirs(os.path.dirname(hit_source_log) or ".", exist_ok=True)
        with open(hit_source_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")
        print(f"[hit-source] Saved to {hit_source_log}")


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
    parser.add_argument(
        "--sync-timing",
        action="store_true",
        help="Synchronize CUDA around measured prefill/decode timing sections.",
    )
    parser.add_argument(
        "--record-hot-experts",
        action="store_true",
        help="Record MoE expert selections for offline hot expert analysis.",
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
        default=2,
        help="Number of decode steps to profile after prefill (default: 1).",
    )
    parser.add_argument(
        "--record-expert-schedule",
        action="store_true",
        help="Record per-layer scheduling decisions (CPU/GPU/preload experts, strategy, placement, latency).",
    )
    parser.add_argument(
        "--expert-schedule-log",
        type=str,
        default=None,
        help="Path to write scheduling decision JSONL. If omitted, records are kept in memory only.",
    )
    parser.add_argument(
        "--hit-source-log",
        type=str,
        default=None,
        help="Path to write hit-source stats JSONL (e.g. logs/hit_source_${TIMESTAMP}.jsonl).",
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
        trace_path = os.path.join(args.profile_torch_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-trace.json")
        profiler_ctx.export_chrome_trace(trace_path)
        print(profiler_ctx.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        print(f"[profiler] Chrome trace saved to: {trace_path}")

    # Print runtime state and expert executor timing if requested
    if args.debug_runtime_state:
        print_runtime_state(model, "after-measure")
    if args.profile_expert_executor:
        print_expert_timing(model)
        print_placeholder_hit_rate(model)

    print_executor_hit_sources(model, hit_source_log=args.hit_source_log)

    print(
        f"prefill_time: {prefill_time:.4f}, decode_time: {decode_time:.4f}, hit_rate: {hit_rate:.4f}"
    )
    if args.input_token_num is not None:
        print("tokens per second (prefill):", args.input_token_num / prefill_time)
    print("tokens per second (decode):", args.output_token_num / decode_time)

    if args.record_expert_schedule and hasattr(model, "schedule_stats_recorder") and model.schedule_stats_recorder is not None:
        summary = model.schedule_stats_recorder.summary()
        print(f"[schedule-stats] Total scheduling calls: {summary['total_calls']}")
        for row in summary.get("by_layer", []):
            print(
                f"[schedule-stats] strategy={row['strategy']} phase={row['phase']} "
                f"layer={row['layer']} calls={row['calls']} reasons={row['reasons']} "
                f"gpu_total={row['gpu_total']} cpu_total={row['cpu_total']} "
                f"preload_total={row['preload_total']}"
            )
        if not args.expert_schedule_log:
            print(f"[schedule-stats] Records kept in memory ({len(model.schedule_stats_recorder.records)} records). "
                  "Use --expert-schedule-log PATH to write JSONL.")
        model.schedule_stats_recorder.close()
