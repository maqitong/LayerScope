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


def _safe_rate(numerator, denominator):
    return numerator / max(denominator, 1)


def _write_jsonl(path, record, label):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")
    print(f"[{label}] Saved to {path}")


def collect_runtime_state(model):
    placement = model.placeholder_manager.snapshot()
    ex = model.expert_executor
    return {
        "placeholder_resident": len(placement.placeholder_resident),
        "free_placeholders": placement.free_placeholders,
        "loading": len(placement.loading),
        "planned_preload": ex.planned_preload_count,
        "actual_preload_request": ex.actual_preload_request_count,
        "actual_preload_success": ex.actual_preload_success_count,
        "actual_preload_skip": ex.actual_preload_skip_count,
        "actual_preload_hit": ex.actual_preload_hit_count,
        "eviction": model.placeholder_manager.eviction_count,
    }


def print_runtime_state(model, label):
    stats = collect_runtime_state(model)
    fields = " ".join(f"{key}={value}" for key, value in stats.items())
    print(f"[{label}] {fields}")


def collect_execution_stats(model):
    ex = model.expert_executor
    total_planned_gpu_count = (
        ex.planned_gpu_static_count
        + ex.planned_gpu_placeholder_count
        + ex.planned_gpu_ondemand_count
    )
    total_planned_gpu_tokens = (
        ex.planned_gpu_static_tokens
        + ex.planned_gpu_placeholder_tokens
        + ex.planned_gpu_ondemand_tokens
    )

    return {
        "planned_gpu_static_count": ex.planned_gpu_static_count,
        "planned_gpu_static_tokens": ex.planned_gpu_static_tokens,
        "planned_gpu_placeholder_count": ex.planned_gpu_placeholder_count,
        "planned_gpu_placeholder_tokens": ex.planned_gpu_placeholder_tokens,
        "planned_gpu_ondemand_count": ex.planned_gpu_ondemand_count,
        "planned_gpu_ondemand_tokens": ex.planned_gpu_ondemand_tokens,
        "planned_cpu_count": ex.planned_cpu_count,
        "planned_cpu_tokens": ex.planned_cpu_tokens,
        "planned_preload_count": ex.planned_preload_count,
        "actual_preload_hit_count": ex.actual_preload_hit_count,
        "actual_preload_request_count": ex.actual_preload_request_count,
        "actual_preload_success_count": ex.actual_preload_success_count,
        "actual_preload_skip_count": ex.actual_preload_skip_count,
        "actual_preload_skip_already_gpu_count": ex.actual_preload_skip_already_gpu_count,
        "actual_preload_skip_loading_count": ex.actual_preload_skip_loading_count,
        "actual_preload_skip_no_slot_count": ex.actual_preload_skip_no_slot_count,
        "total_planned_gpu_count": total_planned_gpu_count,
        "total_planned_gpu_tokens": total_planned_gpu_tokens,
        "total_planned_cpu_count": ex.planned_cpu_count,
        "total_planned_cpu_tokens": ex.planned_cpu_tokens,
        "planned_placeholder_rate": _safe_rate(
            ex.planned_gpu_placeholder_count,
            total_planned_gpu_count,
        ),
        "actual_preload_of_placeholder_rate": _safe_rate(
            ex.actual_preload_hit_count,
            ex.planned_gpu_placeholder_count,
        ),
        "actual_preload_of_success_rate": _safe_rate(
            ex.actual_preload_hit_count,
            ex.actual_preload_success_count,
        ),
        "actual_preload_of_request_rate": _safe_rate(
            ex.actual_preload_hit_count,
            ex.actual_preload_request_count,
        ),
    }


def print_executor_hit_sources(model, hit_source_log=None):
    stats = collect_execution_stats(model)
    print(
        "[hit-source] "
        f"planned_gpu_static={stats['planned_gpu_static_count']} "
        f"(tokens={stats['planned_gpu_static_tokens']})"
    )
    print(
        "[hit-source] "
        f"planned_gpu_placeholder={stats['planned_gpu_placeholder_count']} "
        f"(tokens={stats['planned_gpu_placeholder_tokens']})"
    )
    print(
        "[hit-source] "
        f"planned_gpu_ondemand={stats['planned_gpu_ondemand_count']} "
        f"(tokens={stats['planned_gpu_ondemand_tokens']})"
    )
    print(
        "[hit-source] "
        f"actual_preload_hit={stats['actual_preload_hit_count']} "
        "(of planned placeholder hits)"
    )
    print(
        "[hit-source] "
        f"total_planned_gpu_count={stats['total_planned_gpu_count']} "
        f"(tokens={stats['total_planned_gpu_tokens']})"
    )
    print(
        "[hit-source] "
        f"total_planned_cpu_count={stats['total_planned_cpu_count']} "
        f"(tokens={stats['total_planned_cpu_tokens']})"
    )
    print(
        "[hit-source] "
        f"planned_placeholder_rate={stats['planned_placeholder_rate']:.4f} "
        "(placeholder / planned_gpu)"
    )
    print(
        "[hit-source] actual_preload_hit/planned_placeholder="
        f"{stats['actual_preload_of_placeholder_rate']:.4f}"
    )
    print(
        "[hit-source] actual_preload_hit/actual_preload_success="
        f"{stats['actual_preload_of_success_rate']:.4f}"
    )
    print(
        "[hit-source] actual_preload_hit/actual_preload_request="
        f"{stats['actual_preload_of_request_rate']:.4f}"
    )
    print(
        "[hit-source] "
        f"actual_preload_skip={stats['actual_preload_skip_count']} "
        f"(loading={stats['actual_preload_skip_loading_count']} "
        f"no_slot={stats['actual_preload_skip_no_slot_count']} "
        f"already_gpu={stats['actual_preload_skip_already_gpu_count']})"
    )

    if hit_source_log:
        record = {"timestamp": datetime.datetime.now().isoformat(), **stats}
        _write_jsonl(hit_source_log, record, "hit-source")


def print_expert_timing(model):
    lines = model.expert_executor.format_timing_summary()
    if not lines:
        print("[expert-timing] no expert executor timing recorded")
        return
    print(
        "[expert-timing] wall_share uses layer execute wall time; "
        "shares can sum above 100% when tasks overlap"
    )
    for line in lines:
        print(line)




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
        choices=[0, 1, 2, 3],
        help="0: execute at GPU (baseline), 1: Scope strategy, 2: Fiddler strategy, 3: Pregated strategy.",
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

    input_text = [input_text[0]*1024]

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

    prefill_time, decode_time = model.generate(
        input_text,
        output_token=args.output_token_num,
        input_token=args.input_token_num,
        profiler=profiler_ctx,
        profile_decode_steps=args.profile_decode_steps if args.profile_torch else 0,
    )

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

    print_executor_hit_sources(model, hit_source_log=args.hit_source_log)

    print(
        f"prefill_time: {prefill_time:.4f}, decode_time: {decode_time:.4f}"
    )
    if args.input_token_num is not None:
        print("tokens per second (prefill):", args.input_token_num / prefill_time)
    print("tokens per second (decode):", args.output_token_num / decode_time)

    if args.record_expert_schedule and hasattr(model, "schedule_stats_recorder") and model.schedule_stats_recorder is not None:
        summary = model.schedule_stats_recorder.summary()
        print(f"[schedule-stats] Total scheduling calls: {summary['total_calls']}")
        if not args.expert_schedule_log:
            print(f"[schedule-stats] Records kept in memory ({len(model.schedule_stats_recorder.records)} records). "
                  "Use --expert-schedule-log PATH to write JSONL.")
        model.schedule_stats_recorder.close()
