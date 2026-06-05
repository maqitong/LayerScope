#!/usr/bin/env python3
"""分析 schedule.jsonl 中每条调度记录的逐步决策过程和最优性。

用法:
    python src/scripts/analyze_schedule.py src/logs/schedule.jsonl
    python src/scripts/analyze_schedule.py src/logs/schedule.jsonl --call-index 81
    python src/scripts/analyze_schedule.py src/logs/schedule.jsonl --layer 4 --reason decode-mode-c
    python src/scripts/analyze_schedule.py src/logs/schedule.jsonl --summary
    python src/scripts/analyze_schedule.py src/logs/schedule.jsonl --suboptimal

核心概念:
    - wall time (执行墙钟时间): max(GPU专家数×t_g, CPU专家数×t_c)
      GPU 和 CPU 并行执行，总耗时取决于较慢的一侧。
    - n_g_rho (理想GPU专家数): 使 wall time 最小的 GPU 专家数量。
    - suboptimal (次优): 实际 wall time > 最优 wall time 的调度决策。
    - slack (时间浪费): actual_wall - optimal_wall，衡量偏离最优的程度。

汇总表各列:
    - reason:    调度模式名称（decode-mode-a/b/c、decode-fallback-prefill、prefill）
    - count:     该模式被触发的总次数
    - subopt:    其中未达到理论最优的调用次数
    - avg_slack: 次优调用的平均时间浪费（ms）
    - gpu:       该模式分配到 GPU 执行的专家总次数
    - cpu:       该模式分配到 CPU 执行的专家总次数
    - preload:   该模式预加载的专家总次数
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_records(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def replay_decode_step(record: Dict) -> Dict:
    """从 JSONL 记录回放 schedule_decode 的逐步计算。

    计算步骤:
    1. t_c (单token CPU延迟): 从 latency_cpu_table 取 token_count=1 的值
    2. t_g (单token GPU计算延迟): 从 latency_gpu_table 取 token_count=1 的值
    3. n_g_rho (理想GPU专家数): 遍历 0..k，找使 max(n_g*t_g, (k-n_g)*t_c) 最小的 n_g
    4. cur_below: 当前层 GPU 驻留专家数 < n_g_rho?
    5. next_below: 下一层 GPU 驻留专家数 < n_g_rho?
    6. 根据 (cur_below, next_below) 选择四种模式之一
    7. 比较 actual_wall vs optimal_wall 判断是否最优
    8. 计算 t_gap (CPU-GPU时间差) 和 preload_capacity (气泡内可预加载数)

    返回:
        包含所有中间计算值的字典
    """
    latency = record.get("latency", {})
    strategy_params = record.get("strategy_params", {})
    placement = record.get("placement", {})
    current_demands = record.get("current_demands", [])
    future_demands = record.get("future_demands", [])

    t_c = float(list(latency.get("latency_cpu_table", {}).values())[0]) if latency.get("latency_cpu_table") else 0.142
    t_g = float(list(latency.get("latency_gpu_table", {}).values())[0]) if latency.get("latency_gpu_table") else 0.093
    t_io = latency.get("t_io", 0.8)
    alpha = strategy_params.get("alpha", 0.1)
    t_attn = strategy_params.get("t_attn", 0.6)
    r_hit = strategy_params.get("r_hit", 0.8)

    k = len(current_demands)
    current_resident_ids = placement.get("current_resident", [])
    future_resident_ids = placement.get("future_resident", [])

    n_g_rho_values = []
    for n_g in range(k + 1):
        wall = max(n_g * t_g, (k - n_g) * t_c)
        n_g_rho_values.append({"n_g": n_g, "gpu_time": round(n_g * t_g, 6), "cpu_time": round((k - n_g) * t_c, 6), "wall": round(wall, 6)})
    n_g_rho = min(range(k + 1), key=lambda n_g: max(n_g * t_g, (k - n_g) * t_c))

    cur_resident_count = len(current_resident_ids)
    next_resident_count = len(future_resident_ids)
    cur_below = cur_resident_count < n_g_rho
    next_below = next_resident_count < n_g_rho

    if cur_below and next_below:
        mode = "decode-fallback-prefill"
    elif cur_below and not next_below:
        mode = "decode-mode-a"
    elif not cur_below and next_below:
        mode = "decode-mode-b"
    else:
        mode = "decode-mode-c"

    actual_gpu_count = len(record.get("gpu_experts", []))
    actual_cpu_count = len(record.get("cpu_experts", []))
    cpu_ids = [e["expert_id"] for e in record.get("cpu_experts", [])]
    gpu_ids = [e["expert_id"] for e in record.get("gpu_experts", [])]
    preload_ids = [e["expert_id"] for e in record.get("preload_experts", [])]
    current_non_resident_ids = [e["expert_id"] for e in current_demands if e["expert_id"] not in current_resident_ids]

    n_resident_gpu = sum(1 for gid in gpu_ids if gid in set(current_resident_ids))
    n_transfer_gpu = actual_gpu_count - n_resident_gpu

    is_prefill_logic = (mode == "decode-fallback-prefill")

    if is_prefill_logic:
        gpu_compute_time = actual_gpu_count * t_g
        gpu_io_time = alpha + n_transfer_gpu * t_io
        actual_wall = max(gpu_compute_time, gpu_io_time) + t_g if actual_gpu_count > 0 else 0.0
        actual_wall = max(actual_wall, actual_cpu_count * t_c)

        best_wall = float("inf")
        best_n_g = 0
        for n_g in range(k + 1):
            n_xfer = max(0, n_g - cur_resident_count)
            n_cpu = k - n_g
            gpu_comp = n_g * t_g
            gpu_io_val = alpha + n_xfer * t_io
            wall_g = max(gpu_comp, gpu_io_val) + t_g if n_g > 0 else 0.0
            wall_cpu = n_cpu * t_c
            wall = max(wall_g, wall_cpu)
            if wall < best_wall:
                best_wall = wall
                best_n_g = n_g
        optimal_wall = best_wall
    else:
        actual_wall = max(actual_gpu_count * t_g, actual_cpu_count * t_c)
        optimal_wall = max(n_g_rho * t_g, (k - n_g_rho) * t_c)

    suboptimal = actual_wall > optimal_wall + 1e-9
    slack = round(actual_wall - optimal_wall, 6)
    slack_pct = round(slack / optimal_wall * 100, 2) if optimal_wall > 0 else 0.0

    future_non_resident_ids = [e["expert_id"] for e in future_demands if e["expert_id"] not in future_resident_ids]

    t_gap = max(0.0, actual_cpu_count * t_c - actual_gpu_count * t_g)
    preload_capacity = math.floor((t_gap + t_attn) / max(t_io, 1e-9)) if t_io > 0 else 0
    xi = (2 * r_hit - 1) * t_io

    steps = {
        "k": k,
        "t_c": t_c,
        "t_g": t_g,
        "t_io": t_io,
        "alpha": alpha,
        "t_attn": t_attn,
        "r_hit": r_hit,
        "n_g_rho_scan": n_g_rho_values,
        "n_g_rho": n_g_rho,
        "current_resident_count": cur_resident_count,
        "next_resident_count": next_resident_count,
        "cur_below": cur_below,
        "next_below": next_below,
        "mode_selected": mode,
        "actual_gpu_count": actual_gpu_count,
        "actual_cpu_count": actual_cpu_count,
        "n_resident_gpu": n_resident_gpu,
        "n_transfer_gpu": n_transfer_gpu,
        "actual_wall_time": round(actual_wall, 6),
        "optimal_wall_time": round(optimal_wall, 6),
        "suboptimal": suboptimal,
        "slack": slack,
        "slack_pct": slack_pct,
        "cpu_ids": cpu_ids,
        "gpu_ids": gpu_ids,
        "preload_ids": preload_ids,
        "current_resident_ids": current_resident_ids,
        "current_non_resident_ids": current_non_resident_ids,
        "future_resident_ids": future_resident_ids,
        "preload_capacity": preload_capacity,
        "t_gap": round(t_gap, 6),
        "xi": round(xi, 6),
        "is_prefill_logic": is_prefill_logic,
    }

    if is_prefill_logic:
        best_n_xfer = max(0, best_n_g - cur_resident_count)
        steps["prefill_logic_note"] = (
            f"fallback-prefill 使用含I/O的成本模型: "
            f"实际 GPU={actual_gpu_count}(驻留={n_resident_gpu}, 需加载={n_transfer_gpu}) "
            f"I/O开销={n_transfer_gpu}×{t_io}={n_transfer_gpu*t_io:.2f}ms。"
            f"最优基准含I/O: optimal_n_g={best_n_g} "
            f"(驻留={cur_resident_count}, 需加载={best_n_xfer}, wall={optimal_wall:.4f})"
        )

    if mode == "decode-mode-c" and cur_resident_count > n_g_rho:
        excess = cur_resident_count - n_g_rho
        steps["mode_c_issue"] = f"GPU 保留了 {cur_resident_count} 个驻留专家，但 n_g_rho={n_g_rho}，多保留了 {excess} 个。"
        steps["mode_c_suggestion"] = f"若将 GPU 限制为 {n_g_rho} 个驻留专家，wall time 可从 {actual_wall:.4f} 降至 {optimal_wall:.4f}（-{slack_pct}%）。"

    if mode == "decode-mode-b":
        need_next = max(0, n_g_rho - next_resident_count)
        steps["need_next"] = need_next
        steps["gpu_trimmed_to_n_g_rho"] = n_g_rho
        if actual_gpu_count != n_g_rho and cur_resident_count > n_g_rho:
            steps["mode_b_issue"] = f"mode-b 将 GPU 限制为 n_g_rho={n_g_rho}，但实际 gpu_count={actual_gpu_count}。"

    return steps


def replay_prefill_step(record: Dict) -> Dict:
    """回放 schedule_prefill 的关键中间值。

    prefill 阶段使用三步调度法:
    1. 分离当前驻留/非驻留专家和未来非驻留专家
    2. 合并排序后构造全局候选队列（_select_global_queue）
    3. 从全局队列中选出当前层值得按需加载到 GPU 的专家（_select_current_ondemand）
    4. 根据 I/O 气泡窗口选择未来专家预加载（_select_preload）

    返回:
        包含关键中间值的字典
    """
    latency = record.get("latency", {})
    strategy_params = record.get("strategy_params", {})
    placement = record.get("placement", {})
    current_demands = record.get("current_demands", [])
    future_demands = record.get("future_demands", [])

    t_c = float(list(latency.get("latency_cpu_table", {}).values())[0]) if latency.get("latency_cpu_table") else 0.142
    t_g = float(list(latency.get("latency_gpu_table", {}).values())[0]) if latency.get("latency_gpu_table") else 0.093
    t_io = latency.get("t_io", 1.4)
    alpha = strategy_params.get("alpha", 0.1)
    t_attn = strategy_params.get("t_attn", 0.6)

    current_resident_ids = placement.get("current_resident", [])
    cpu_ids = [e["expert_id"] for e in record.get("cpu_experts", [])]
    gpu_ids = [e["expert_id"] for e in record.get("gpu_experts", [])]
    preload_ids = [e["expert_id"] for e in record.get("preload_experts", [])]

    actual_gpu_count = len(gpu_ids)
    actual_cpu_count = len(cpu_ids)
    actual_wall = max(actual_gpu_count * t_g, actual_cpu_count * t_c)

    return {
        "current_demands_count": len(current_demands),
        "future_demands_count": len(future_demands),
        "current_resident": current_resident_ids,
        "gpu_ids": gpu_ids,
        "cpu_ids": cpu_ids,
        "preload_ids": preload_ids,
        "actual_wall_time": round(actual_wall, 6),
        "free_placeholders": placement.get("free_placeholders", 0),
        "gpu_resident_count": placement.get("gpu_resident_count", 0),
        "placeholder_resident_count": placement.get("placeholder_resident_count", 0),
    }


def format_steps(record: Dict, steps: Dict) -> str:
    """将单条记录的计算步骤格式化为可读的多行文本。

    decode 阶段输出内容:
    - 延迟参数 (t_c, t_g, t_io)
    - n_g_rho 扫描表 (每个 n_g 对应的 GPU时间/CPU时间/wall time)
    - 当前层/下一层驻留状态
    - cur_below/next_below 判断
    - 最终决策 (GPU/CPU/preload 专家列表)
    - 最优性分析 (actual_wall vs optimal_wall, slack)
    - 预加载上下文 (t_gap, preload_capacity)

    prefill 阶段输出内容:
    - 需求数量和驻留状态
    - 最终决策
    - GPU/placeholder 状态
    """
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"call_index={record.get('call_index')}  layer={record.get('layer')}  phase={record.get('phase')}")
    lines.append(f"strategy={record.get('strategy')}  reason={record.get('reason')}")
    lines.append(f"{'─'*80}")
    if record.get("phase") == "decode":
        lines.append(f"  k={steps['k']}  t_c={steps['t_c']}  t_g={steps['t_g']}  t_io={steps['t_io']}")
        lines.append(f"  n_g_rho scan:")
        for entry in steps["n_g_rho_scan"]:
            marker = " ← optimal" if entry["n_g"] == steps["n_g_rho"] else ""
            lines.append(f"    n_g={entry['n_g']}: gpu_time={entry['gpu_time']:.4f}  cpu_time={entry['cpu_time']:.4f}  wall={entry['wall']:.4f}{marker}")
        lines.append(f"  n_g_rho={steps['n_g_rho']}")
        lines.append(f"  current_resident={steps['current_resident_ids']} (count={steps['current_resident_count']})")
        lines.append(f"  current_non_resident={steps['current_non_resident_ids']}")
        lines.append(f"  next_resident_count={steps['next_resident_count']}  (future_resident={steps['future_resident_ids']})")
        lines.append(f"  cur_below={steps['cur_below']}  next_below={steps['next_below']}")
        lines.append(f"  → mode={steps['mode_selected']}")
        lines.append(f"  ── Decision ──")
        lines.append(f"  gpu_experts={steps['gpu_ids']} (count={steps['actual_gpu_count']}, resident={steps['n_resident_gpu']}, need_load={steps['n_transfer_gpu']})")
        lines.append(f"  cpu_experts={steps['cpu_ids']} (count={steps['actual_cpu_count']})")
        lines.append(f"  preload_experts={steps['preload_ids']}")
        if steps.get("is_prefill_logic"):
            lines.append(f"  ── Optimality (含I/O成本模型) ──")
        else:
            lines.append(f"  ── Optimality ──")
        lines.append(f"  actual_wall={steps['actual_wall_time']:.4f}  optimal_wall={steps['optimal_wall_time']:.4f}")
        if steps["suboptimal"]:
            lines.append(f"  ⚠ SUBOPTIMAL: slack={steps['slack']:.4f} ({steps['slack_pct']}%)")
        else:
            lines.append(f"  ✓ optimal")
        if steps.get("mode_c_issue"):
            lines.append(f"  ⚠ {steps['mode_c_issue']}")
            lines.append(f"    → {steps['mode_c_suggestion']}")
        if steps.get("prefill_logic_note"):
            lines.append(f"  ℹ {steps['prefill_logic_note']}")
        if steps.get("mode_b_issue"):
            lines.append(f"  ⚠ {steps['mode_b_issue']}")
        lines.append(f"  ── Preload Context ──")
        lines.append(f"  t_gap={steps['t_gap']}  preload_capacity={steps['preload_capacity']}  xi={steps['xi']}")
    else:
        lines.append(f"  current_demands={steps['current_demands_count']}  future_demands={steps['future_demands_count']}")
        lines.append(f"  current_resident={steps['current_resident']}")
        lines.append(f"  gpu_experts={steps['gpu_ids']}  cpu_experts={steps['cpu_ids']}  preload={steps['preload_ids']}")
        lines.append(f"  actual_wall={steps['actual_wall_time']:.4f}")
        lines.append(f"  gpu_resident={steps['gpu_resident_count']}  placeholder_resident={steps['placeholder_resident_count']}  free={steps['free_placeholders']}")

    counters = record.get("counters", {})
    if counters:
        hit = counters.get("expert_hit", 0)
        all_ = counters.get("expert_all", 1)
        lines.append(f"  ── Counters ──")
        lines.append(f"  hit={hit}  all={all_}  rate={hit/max(all_,1):.4f}")

    return "\n".join(lines)


def print_summary(records: List[Dict]) -> None:
    """打印汇总统计，包含两个表:

    表1 - 按调度模式汇总:
    - 各模式被触发次数、次优次数、平均时间浪费
    - GPU/CPU/preload 专家分配总次数

    表2 - 按层汇总次优调用:
    - 每层 decode 调用中次优决策的比例
    - 每层次优调用的平均 slack
    """
    by_mode = defaultdict(lambda: {"count": 0, "layers": defaultdict(int), "suboptimal": 0, "slack_total": 0.0, "gpu_total": 0, "cpu_total": 0, "preload_total": 0})
    for rec in records:
        reason = rec.get("reason", "unknown")
        bucket = by_mode[reason]
        bucket["count"] += 1
        bucket["layers"][rec.get("layer", -1)] += 1
        bucket["gpu_total"] += len(rec.get("gpu_experts", []))
        bucket["cpu_total"] += len(rec.get("cpu_experts", []))
        bucket["preload_total"] += len(rec.get("preload_experts", []))
        if rec.get("phase") == "decode":
            steps = replay_decode_step(rec)
            if steps["suboptimal"]:
                bucket["suboptimal"] += 1
                bucket["slack_total"] += steps["slack"]

    print(f"\n{'='*80}")
    print(f"Schedule Summary  ({len(records)} records)")
    print(f"{'='*80}")
    print(f"{'reason':<30} {'count':>6} {'subopt':>7} {'avg_slack':>10} {'gpu':>6} {'cpu':>6} {'preload':>8}")
    print(f"{'─'*80}")
    for reason in sorted(by_mode.keys()):
        b = by_mode[reason]
        avg_slack = b["slack_total"] / b["suboptimal"] if b["suboptimal"] > 0 else 0
        print(f"{reason:<30} {b['count']:>6} {b['suboptimal']:>7} {avg_slack:>10.4f} {b['gpu_total']:>6} {b['cpu_total']:>6} {b['preload_total']:>8}")

    print(f"\n{'='*80}")
    print("Suboptimal decode calls by layer:")
    print(f"{'='*80}")
    layer_stats = defaultdict(lambda: {"total": 0, "suboptimal": 0, "slack_total": 0.0})
    for rec in records:
        if rec.get("phase") != "decode":
            continue
        layer = rec.get("layer", -1)
        steps = replay_decode_step(rec)
        ls = layer_stats[layer]
        ls["total"] += 1
        if steps["suboptimal"]:
            ls["suboptimal"] += 1
            ls["slack_total"] += steps["slack"]
    print(f"{'layer':>6} {'total':>6} {'subopt':>7} {'%subopt':>8} {'avg_slack':>10}")
    print(f"{'─'*45}")
    for layer in sorted(layer_stats.keys()):
        ls = layer_stats[layer]
        pct = ls["suboptimal"] / ls["total"] * 100 if ls["total"] > 0 else 0
        avg = ls["slack_total"] / ls["suboptimal"] if ls["suboptimal"] > 0 else 0
        print(f"{layer:>6} {ls['total']:>6} {ls['suboptimal']:>7} {pct:>7.1f}% {avg:>10.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="分析 schedule.jsonl 中每条调度记录的逐步决策过程和最优性。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("jsonl_path", help="schedule.jsonl 文件路径")
    parser.add_argument(
        "--call-index", type=int,
        help="显示指定 call_index 的详细决策过程追踪",
    )
    parser.add_argument(
        "--layer", type=int,
        help="按层号过滤（如 --layer 4 只看第4层）",
    )
    parser.add_argument(
        "--reason", type=str,
        help="按调度模式过滤: decode-mode-a, decode-mode-b, decode-mode-c, decode-fallback-prefill, prefill",
    )
    parser.add_argument(
        "--phase", type=str,
        help="按阶段过滤: prefill 或 decode",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="打印汇总统计表（按模式+按层的次优分析）",
    )
    parser.add_argument(
        "--suboptimal", action="store_true",
        help="只显示次优决策（actual_wall > optimal_wall 的记录）",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最多显示多少条记录（0=不限制）",
    )
    args = parser.parse_args()

    records = load_records(args.jsonl_path)
    print(f"Loaded {len(records)} records from {args.jsonl_path}")

    if args.summary:
        print_summary(records)
        return

    filtered = records
    if args.call_index is not None:
        filtered = [r for r in filtered if r.get("call_index") == args.call_index]
    if args.layer is not None:
        filtered = [r for r in filtered if r.get("layer") == args.layer]
    if args.reason:
        filtered = [r for r in filtered if r.get("reason") == args.reason]
    if args.phase:
        filtered = [r for r in filtered if r.get("phase") == args.phase]

    shown = 0
    for rec in filtered:
        if rec.get("phase") == "decode":
            steps = replay_decode_step(rec)
            if args.suboptimal and not steps["suboptimal"]:
                continue
            print(format_steps(rec, steps))
        else:
            steps = replay_prefill_step(rec)
            print(format_steps(rec, steps))
        shown += 1
        if args.limit > 0 and shown >= args.limit:
            break

    if not shown:
        print("No matching records found.")
    else:
        print(f"\n(showing {shown} records, {len(filtered)} matched filters)")


if __name__ == "__main__":
    main()
