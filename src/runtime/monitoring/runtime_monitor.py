import datetime
import json
import os
from collections import Counter
from typing import Dict, Optional

import torch

from strategies.scheduling.expert_scheduling import ExpertSchedulingStatsRecorder


class RuntimeMonitor:
    """Aggregates runtime counters and reporting data without owning execution."""

    def __init__(
        self,
        runtime_meta: Optional[Dict] = None,
        schedule_log: Optional[str] = None,
        record_schedule: bool = False,
    ):
        self.runtime_meta = runtime_meta or {}
        self.schedule_log = schedule_log
        self.hot_expert_counts = Counter()
        self.generation_timing = []
        self.schedule_stats_recorder = None
        if record_schedule:
            self.schedule_stats_recorder = ExpertSchedulingStatsRecorder(
                output_path=schedule_log,
                runtime_meta=self.runtime_meta,
            )

    def reset(self) -> None:
        self.generation_timing.clear()
        if self.schedule_stats_recorder is not None:
            self.schedule_stats_recorder.reset()

    def close(self) -> None:
        if self.schedule_stats_recorder is not None:
            self.schedule_stats_recorder.close()

    def attach_schedule_recorder(self, strategy) -> None:
        if self.schedule_stats_recorder is not None:
            strategy.set_stats_recorder(self.schedule_stats_recorder)

    def record_generation_timing(self, prefill_time: float, decode_time: float) -> None:
        self.generation_timing.append(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "prefill_time": prefill_time,
                "decode_time": decode_time,
            }
        )

    def record_hot_experts(self, layer: int, selected_experts: torch.Tensor) -> None:
        selected = selected_experts.detach().to("cpu").reshape(-1).tolist()
        self.hot_expert_counts.update((int(layer), int(expert_id)) for expert_id in selected)

    def reset_hot_expert_stats(self) -> None:
        self.hot_expert_counts.clear()

    def sorted_hot_expert_stats(self):
        return sorted(
            self.hot_expert_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )

    def export_hot_expert_stats(self):
        return [
            {"layer": layer, "expert": expert, "count": count}
            for (layer, expert), count in self.sorted_hot_expert_stats()
        ]

    def schedule_summary(self) -> Dict:
        if self.schedule_stats_recorder is None:
            return {"total_calls": 0}
        return self.schedule_stats_recorder.summary()

    def hit_source_summary(self, executor) -> Dict:
        total_gpu_hits = (
            executor.static_gpu_hit_count
            + executor.placeholder_hit_count
            + executor.ondemand_load_count
        )
        total_gpu_tokens = (
            executor.static_gpu_hit_tokens
            + executor.placeholder_hit_tokens
            + executor.ondemand_load_tokens
        )
        return {
            "static_gpu_hit_count": executor.static_gpu_hit_count,
            "static_gpu_hit_tokens": executor.static_gpu_hit_tokens,
            "placeholder_hit_count": executor.placeholder_hit_count,
            "placeholder_hit_tokens": executor.placeholder_hit_tokens,
            "ondemand_load_count": executor.ondemand_load_count,
            "ondemand_load_tokens": executor.ondemand_load_tokens,
            "preload_hit_count": executor.preload_hit_count,
            "preload_request_count": executor.preload_request_count,
            "preload_success_count": executor.preload_success_count,
            "preload_skip_count": executor.preload_skip_count,
            "total_gpu_hits": total_gpu_hits,
            "total_gpu_tokens": total_gpu_tokens,
            "placeholder_hit_rate": executor.placeholder_hit_count / max(total_gpu_hits, 1),
            "preload_of_placeholder_rate": executor.preload_hit_count / max(executor.placeholder_hit_count, 1),
            "preload_of_success_rate": executor.preload_hit_count / max(executor.preload_success_count, 1),
            "preload_of_request_rate": executor.preload_hit_count / max(executor.preload_request_count, 1),
        }

    def placeholder_summary(self, placeholder_manager) -> Dict:
        snapshot = placeholder_manager.snapshot()
        return {
            "placeholder_resident": len(snapshot.placeholder_resident),
            "free_placeholders": snapshot.free_placeholders,
            "loading": len(snapshot.loading),
            "eviction_count": placeholder_manager.eviction_count,
            "placeholder_resident_hit_num": placeholder_manager.placeholder_resident_hit_num,
            "static_gpu_resdient_hit_num": placeholder_manager.static_gpu_resdient_hit_num,
        }

    def runtime_summary(self, executor=None, placeholder_manager=None) -> Dict:
        summary = {"generation_timing": list(self.generation_timing)}
        if executor is not None:
            summary["hit_source"] = self.hit_source_summary(executor)
        if placeholder_manager is not None:
            summary["placeholder"] = self.placeholder_summary(placeholder_manager)
        if self.schedule_stats_recorder is not None:
            summary["schedule"] = self.schedule_summary()
        return summary

    def write_hit_source_log(self, path: str, executor) -> None:
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            **self.hit_source_summary(executor),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n")
