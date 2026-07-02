import json
import os
from typing import Dict, List, Optional, Protocol

from model.expert_types import ExpertSchedule


class ExpertSchedulingMonitor(Protocol):
    def record_gpu_only(self, **kwargs) -> None:
        ...

    def record_fiddler(self, **kwargs) -> None:
        ...

    def record_prefetch_hybrid(self, **kwargs) -> None:
        ...


class ExpertSchedulingStatsRecorder:
    """Opt-in monitor for structured expert scheduling decisions.

    Scheduling strategies pass only lightweight decision data to the semantic
    record_* methods. Each record keeps strategy, layer, reason, and selected
    expert ids for active/cpu/gpu/preload groups.
    """

    def __init__(self, output_path: Optional[str] = None):
        self.enabled = True
        self.output_path = output_path
        self._records: List[Dict] = []
        self._file_handle = None
        if self.output_path is not None:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            self._file_handle = open(self.output_path, "a", encoding="utf-8")

    def record(self, record: Dict) -> None:
        if not self.enabled:
            return
        self._records.append(record)
        if self._file_handle is not None:
            self._file_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file_handle.flush()

    def record_gpu_only(
        self,
        *,
        strategy_name: str,
        layer: int,
        active_experts: List[int],
    ) -> None:
        self.record({
            "strategy": strategy_name,
            "reason": "gpu-only",
            "layer": layer,
            "active_experts": active_experts,
            "cpu_experts": [],
            "gpu_experts": active_experts,
            "preload_experts": [],
        })

    def record_fiddler(
        self,
        *,
        strategy_name: str,
        layer: int,
        active_experts: List[int],
        cpu_experts: List[int],
        gpu_experts: List[int],
    ) -> None:
        self.record({
            "strategy": strategy_name,
            "reason": "fiddler-cost-opt",
            "layer": layer,
            "active_experts": active_experts,
            "cpu_experts": cpu_experts,
            "gpu_experts": gpu_experts,
            "preload_experts": [],
        })

    def record_prefetch_hybrid(
        self,
        *,
        strategy_name: str,
        layer: int,
        active_experts: List[int],
        schedule: ExpertSchedule,
    ) -> None:
        self.record({
            "strategy": strategy_name,
            "reason": schedule.reason,
            "layer": layer,
            "active_experts": active_experts,
            "cpu_experts": schedule.cpu_expert_ids,
            "gpu_experts": schedule.gpu_expert_ids,
            "preload_experts": schedule.preload_expert_ids,
        })

    @property
    def records(self) -> List[Dict]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def summary(self) -> Dict:
        if not self._records:
            return {"total_calls": 0}
        by_key: Dict[tuple, Dict] = {}
        for rec in self._records:
            key = (rec.get("strategy", ""), rec.get("layer", -1))
            bucket = by_key.setdefault(
                key,
                {
                    "count": 0,
                    "reasons": {},
                    "gpu_total": 0,
                    "cpu_total": 0,
                    "preload_total": 0,
                    "unique_gpu": set(),
                    "unique_cpu": set(),
                    "unique_preload": set(),
                },
            )
            bucket["count"] += 1
            reason = rec.get("reason", "")
            bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
            bucket["gpu_total"] += len(rec.get("gpu_experts", []))
            bucket["cpu_total"] += len(rec.get("cpu_experts", []))
            bucket["preload_total"] += len(rec.get("preload_experts", []))
            for expert in rec.get("gpu_experts", []):
                bucket["unique_gpu"].add(expert)
            for expert in rec.get("cpu_experts", []):
                bucket["unique_cpu"].add(expert)
            for expert in rec.get("preload_experts", []):
                bucket["unique_preload"].add(expert)
        rows = []
        for (strategy, layer), bucket in sorted(by_key.items()):
            rows.append({
                "strategy": strategy,
                "layer": layer,
                "calls": bucket["count"],
                "reasons": bucket["reasons"],
                "gpu_total": bucket["gpu_total"],
                "cpu_total": bucket["cpu_total"],
                "preload_total": bucket["preload_total"],
                "unique_gpu": sorted(bucket["unique_gpu"]),
                "unique_cpu": sorted(bucket["unique_cpu"]),
                "unique_preload": sorted(bucket["unique_preload"]),
            })
        return {"total_calls": len(self._records), "by_layer": rows}
