"""Runtime microbenchmark for DeepSeek-V2-Lite using real inference operators.

Drives mDeepSeek.generate() through the full runtime path:
  gate routing -> expert_strategy -> ExpertExecutionManager -> placeholder/preload
and collects timing from RuntimeMonitor + ExpertExecutionManager.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.deepseek import mDeepSeek


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


def bench_runtime_generate(model, prompts, input_token, output_token,
                           warmup=0, preserve_warmup_cache=False):
    results = []

    for i in range(warmup):
        print(f"warmup {i + 1}/{warmup}")
        model.generate(
            prompts,
            output_token=output_token,
            input_token=input_token,
        )

    if warmup > 0 and not preserve_warmup_cache:
        model.reset_runtime_state(clear_placeholders=True)

    for rep in range(1):
        prefill_time, decode_time, hit_rate = model.generate(
            prompts,
            output_token=output_token,
            input_token=input_token,
        )

        input_tok = input_token if input_token is not None else 0
        prefill_throughput = (input_tok / prefill_time) if prefill_time > 0 else 0.0
        decode_throughput = (output_token / decode_time) if decode_time > 0 else 0.0

        expert_timing = model.expert_executor.timing_summary()

        runtime_summary = model.monitor.runtime_summary(
            model.expert_executor, model.placeholder_manager
        )

        results.append({
            "prefill_time_s": round(prefill_time, 6),
            "decode_time_s": round(decode_time, 6),
            "hit_rate": round(hit_rate, 6),
            "input_token_num": input_tok,
            "output_token_num": output_token,
            "prefill_throughput_tokens_per_s": round(prefill_throughput, 4),
            "decode_throughput_tokens_per_s": round(decode_throughput, 4),
        })

    return results, expert_timing, runtime_summary


def estimate_latency_fields(expert_timing, model):
    total_gpu = 0.0
    total_cpu = 0.0
    total_preload = 0.0
    total_calls = 0.0
    for row in expert_timing:
        total_gpu += row.get("gpu", 0.0)
        total_cpu += row.get("cpu", 0.0)
        total_preload += row.get("preload", 0.0)
        total_calls += row.get("calls", 0.0)

    if total_calls > 0:
        gpu_avg_ms = (total_gpu / total_calls) * 1000
        cpu_avg_ms = (total_cpu / total_calls) * 1000
        preload_calls = model.expert_executor.preload_success_count
        copy_avg_ms = (total_preload / max(preload_calls, 1)) * 1000 if total_preload > 0 else model.latency_copy
    else:
        gpu_avg_ms = model.latency_gpu
        cpu_avg_ms = model.latency_cpu
        copy_avg_ms = model.latency_copy

    fallback_notes = []
    if total_calls == 0:
        fallback_notes.append("no expert executor calls; using model default latency")
    if total_preload == 0:
        fallback_notes.append("no preload timing; copy estimate may be imprecise")

    return {
        "gpu_avg_ms": round(gpu_avg_ms, 4),
        "cpu_avg_ms": round(cpu_avg_ms, 4),
        "copy_avg_ms": round(copy_avg_ms, 4),
        "fallback_notes": fallback_notes,
    }


def build_benchmark_json(args, generation_results, expert_timing, runtime_summary,
                         latency_estimates):
    expert_cpu_list = [{"token_count": 1, "avg_time_ms": latency_estimates["cpu_avg_ms"]}]
    expert_gpu_list = [{"token_count": 1, "avg_time_ms": latency_estimates["gpu_avg_ms"]}]
    copy_ms = latency_estimates["copy_avg_ms"]

    data = {
        "benchmark_type": "runtime_generate",
        "model_name": os.path.basename(args.model.rstrip("/")).lower(),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "batch_size": args.batch_size,
            "input_token_num": args.input_token_num,
            "output_token_num": args.output_token_num,
            "cpu_offload": args.cpu_offload,
            "beam_width": args.beam_width,
            "warmup": args.warmup,
            "profile_expert_executor": True,
            "sync_timing": True,
        },
        "generation": generation_results,
        "expert_executor_timing": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
            for row in expert_timing
        ],
        "runtime_summary": runtime_summary,
        "runtime_latency_estimates": {
            "expert_cpu_avg_ms": latency_estimates["cpu_avg_ms"],
            "expert_gpu_avg_ms": latency_estimates["gpu_avg_ms"],
            "expert_copy_avg_ms": latency_estimates["copy_avg_ms"],
        },
        "expert_cpu": expert_cpu_list,
        "expert_gpu": expert_gpu_list,
        "expert_weight_copy": {
            "trials_ms": [round(copy_ms, 4)],
            "avg_ms": round(copy_ms, 4),
            "median_ms": round(copy_ms, 4),
            "p95_ms": round(copy_ms, 4),
        },
    }

    if latency_estimates["fallback_notes"]:
        data["notes"] = "; ".join(latency_estimates["fallback_notes"])

    return data


def save_benchmark_results(data, output_dir=None):
    model_name = data["model_name"]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    filename = f"micro_{model_name}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark results saved to: {filepath}")
    return filepath


def print_benchmark_report(generation_results, expert_timing, runtime_summary,
                           latency_estimates):
    print("=" * 60)
    print("Runtime Generation Benchmark")
    print("=" * 60)

    for i, gen in enumerate(generation_results):
        print(f"\n  Repeat {i + 1}:")
        print(f"    prefill_time   = {gen['prefill_time_s']:.4f}s")
        print(f"    decode_time    = {gen['decode_time_s']:.4f}s")
        print(f"    hit_rate        = {gen['hit_rate']:.4f}")
        print(f"    prefill_throughput = {gen['prefill_throughput_tokens_per_s']:.2f} tokens/s")
        print(f"    decode_throughput  = {gen['decode_throughput_tokens_per_s']:.2f} tokens/s")

    print()
    print("-" * 60)
    print("Expert Executor Timing Summary")
    print("-" * 60)
    if not expert_timing:
        print("  (no timing data)")
    else:
        print(
            f"  {'layer':>5} {'calls':>6} {'wall':>10} {'gpu':>10} "
            f"{'cpu':>10} {'preload':>10}"
        )
        for row in expert_timing:
            print(
                f"  {int(row['layer']):>5} {int(row['calls']):>6} "
                f"{row['wall']:>10.6f} {row['gpu']:>10.6f} "
                f"{row['cpu']:>10.6f} {row['preload']:>10.6f}"
            )

    print()
    print("-" * 60)
    print("Hit Source Summary")
    print("-" * 60)
    hs = runtime_summary.get("hit_source", {})
    print(f"  static_gpu_hit   = {hs.get('static_gpu_hit_count', 0)} (tokens={hs.get('static_gpu_hit_tokens', 0)})")
    print(f"  placeholder_hit  = {hs.get('placeholder_hit_count', 0)} (tokens={hs.get('placeholder_hit_tokens', 0)})")
    print(f"  ondemand_load    = {hs.get('ondemand_load_count', 0)} (tokens={hs.get('ondemand_load_tokens', 0)})")
    print(f"  preload_hit      = {hs.get('preload_hit_count', 0)}")
    print(f"  total_gpu_hits   = {hs.get('total_gpu_hits', 0)}")
    print(f"  placeholder_hit_rate = {hs.get('placeholder_hit_rate', 0):.4f}")

    print()
    print("-" * 60)
    print("Placeholder Summary")
    print("-" * 60)
    ps = runtime_summary.get("placeholder", {})
    print(f"  resident       = {ps.get('placeholder_resident', 0)}")
    print(f"  free           = {ps.get('free_placeholders', 0)}")
    print(f"  loading        = {ps.get('loading', 0)}")
    print(f"  eviction_count = {ps.get('eviction_count', 0)}")

    sched = runtime_summary.get("schedule", None)
    if sched is not None:
        print()
        print("-" * 60)
        print("Schedule Summary")
        print("-" * 60)
        print(f"  total_calls = {sched.get('total_calls', 0)}")

    print()
    print("-" * 60)
    print("Latency Estimates (for _load_latency_tables compatibility)")
    print("-" * 60)
    print(f"  expert_gpu_avg_ms  = {latency_estimates['gpu_avg_ms']:.4f}")
    print(f"  expert_cpu_avg_ms  = {latency_estimates['cpu_avg_ms']:.4f}")
    print(f"  expert_copy_avg_ms = {latency_estimates['copy_avg_ms']:.4f}")
    if latency_estimates["fallback_notes"]:
        print(f"  notes: {'; '.join(latency_estimates['fallback_notes'])}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(
        description="Runtime microbenchmark using real mDeepSeek inference operators."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to DeepSeek model.")
    parser.add_argument("--cpu-offload", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--input-token-num", type=int, default=64,
                        help="Max input tokens for tokenization.")
    parser.add_argument("--output-token-num", type=int, default=20,
                        help="Number of tokens to generate.")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup generate runs before measurement.")
    parser.add_argument("--input", type=str, default="Please tell me a joke.",
                        help="Input prompt text.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to ShareGPT JSON. Overrides --input.")

    args = parser.parse_args()

    args.profile_expert_executor = True
    args.sync_timing = True
    args.record_expert_schedule = True
    args.expert_schedule_log = None
    args.record_hot_experts = False

    model = mDeepSeek(args)

    if args.dataset:
        prompts = load_prompts_from_dataset(args.dataset, args.batch_size)
    elif args.batch_size > 1:
        prompts = [args.input] * args.batch_size
    else:
        prompts = args.input

    generation_results, expert_timing, runtime_summary = bench_runtime_generate(
        model=model,
        prompts=prompts,
        input_token=args.input_token_num,
        output_token=args.output_token_num,
        warmup=args.warmup,
    )

    latency_estimates = estimate_latency_fields(expert_timing, model)

    print_benchmark_report(generation_results, expert_timing, runtime_summary,
                           latency_estimates)

    data = build_benchmark_json(args, generation_results, expert_timing,
                                runtime_summary, latency_estimates)
    save_benchmark_results(data)
