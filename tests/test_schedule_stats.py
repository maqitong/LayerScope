import json
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "src", "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from expert_latency import ExpertLatencyModel
from expert_scheduling import (
    ExpertSchedulingStatsRecorder,
    GPUOnlyStrategy,
    FiddlerStrategy,
    PrefetchHybridStrategy,
    PDScopeScheduler,
    _demand_to_dict,
    _placement_summary,
    _latency_summary,
    _token_counts_by_expert,
)
from expert_types import (
    ExpertDemand,
    ExpertKey,
    ExpertLayerRequest,
    ExpertSchedule,
    PlacementSnapshot,
    build_current_demands,
)


def _make_strategy_args():
    dev = torch.device("cpu")

    def is_expert_in_gpu(layer, expert_id):
        return expert_id < 2

    return dev, is_expert_in_gpu


def _make_tensors():
    selected = torch.tensor([[[3, 1], [3, 5]]])
    weights = torch.tensor([[[0.6, 0.4], [0.7, 0.3]]])
    return selected, weights


def test_recorder_records_and_resets():
    recorder = ExpertSchedulingStatsRecorder()
    recorder.record({"strategy": "test", "layer": 0})
    recorder.record({"strategy": "test", "layer": 1})
    assert len(recorder.records) == 2
    assert recorder.records[0]["call_index"] == 0
    assert recorder.records[1]["call_index"] == 1
    recorder.reset()
    assert len(recorder.records) == 0


def test_recorder_writes_jsonl(tmp_path):
    path = str(tmp_path / "schedule.jsonl")
    recorder = ExpertSchedulingStatsRecorder(output_path=path)
    recorder.record({"strategy": "test", "layer": 0, "cpu_experts": [], "gpu_experts": [1]})
    recorder.close()
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["strategy"] == "test"
    assert data["gpu_experts"] == [1]
    assert "timestamp" in data
    assert "call_index" in data


def test_recorder_runtime_meta(tmp_path):
    path = str(tmp_path / "schedule.jsonl")
    recorder = ExpertSchedulingStatsRecorder(
        output_path=path,
        runtime_meta={"n_layer": 28, "n_expert": 64},
    )
    recorder.record({"strategy": "test"})
    recorder.close()
    data = json.loads(open(path, encoding="utf-8").readline())
    assert data["runtime"]["n_layer"] == 28
    assert data["runtime"]["n_expert"] == 64


def test_recorder_summary():
    recorder = ExpertSchedulingStatsRecorder()
    recorder.record({"strategy": "A", "phase": "prefill", "layer": 1, "reason": "gpu-only", "gpu_experts": [1, 2], "cpu_experts": [], "preload_experts": []})
    recorder.record({"strategy": "A", "phase": "decode", "layer": 1, "reason": "decode-mode-a", "gpu_experts": [1], "cpu_experts": [3], "preload_experts": [5]})
    recorder.record({"strategy": "A", "phase": "decode", "layer": 1, "reason": "decode-mode-a", "gpu_experts": [2], "cpu_experts": [3], "preload_experts": []})
    summary = recorder.summary()
    assert summary["total_calls"] == 3
    rows = summary["by_layer"]
    assert len(rows) == 2
    prefill_row = [r for r in rows if r["phase"] == "prefill"][0]
    assert prefill_row["calls"] == 1
    assert prefill_row["gpu_total"] == 2
    decode_row = [r for r in rows if r["phase"] == "decode"][0]
    assert decode_row["calls"] == 2
    assert decode_row["cpu_total"] == 2
    assert decode_row["unique_gpu"] == [1, 2]


def test_gpu_only_strategy_records():
    dev, is_gpu = _make_strategy_args()
    strategy = GPUOnlyStrategy(dev, is_gpu)
    recorder = ExpertSchedulingStatsRecorder()
    strategy.set_stats_recorder(recorder)
    selected, weights = _make_tensors()
    cpu, gpu, assignments = strategy.decide_and_prepare(
        3, None, selected, weights, 8, placement=PlacementSnapshot(gpu_resident={(3, 1)})
    )
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert rec["strategy"] == "GPUOnlyStrategy"
    assert rec["layer"] == 3
    assert rec["gpu_experts"] == [1, 3, 5]
    assert rec["cpu_experts"] == []
    assert rec["preload_experts"] == []
    assert rec["placement"]["gpu_resident_count"] == 1
    assert rec["placement"]["current_resident"] == [1]


def test_fiddler_strategy_records():
    dev, is_gpu = _make_strategy_args()
    strategy = FiddlerStrategy(dev, is_gpu, latency_cpu=0.2, latency_gpu=0.1, latency_io=1.0)
    recorder = ExpertSchedulingStatsRecorder()
    strategy.set_stats_recorder(recorder)
    selected, weights = _make_tensors()
    cpu, gpu, assignments = strategy.decide_and_prepare(
        3, None, selected, weights, 8, placement=PlacementSnapshot()
    )
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert rec["strategy"] == "FiddlerStrategy"
    assert rec["layer"] == 3
    assert "per_expert_details" in rec
    assert len(rec["per_expert_details"]) == 3
    detail_1 = [d for d in rec["per_expert_details"] if d["expert_id"] == 1][0]
    assert detail_1["resident"] is True
    assert detail_1["decision"] == "gpu"
    assert "strategy_params" in rec
    assert rec["strategy_params"]["latency_cpu"] == 0.2
    assert rec["strategy_params"]["latency_io"] == 1.0
    assert "counters" in rec


def test_prefetch_hybrid_strategy_records():
    dev, is_gpu = _make_strategy_args()
    strategy = PrefetchHybridStrategy(
        dev, is_gpu,
        t_io=1.0,
        latency_cpu_table={1: 0.5},
        latency_gpu_table={1: 0.1},
    )
    recorder = ExpertSchedulingStatsRecorder()
    strategy.set_stats_recorder(recorder)
    selected, weights = _make_tensors()
    cpu, gpu, preload, assignments = strategy.decide_and_prepare(
        3, None, selected, weights, 8,
        is_prefill=True,
        placement=PlacementSnapshot(gpu_resident={(3, 1)}),
    )
    assert len(recorder.records) == 1
    rec = recorder.records[0]
    assert rec["strategy"] == "PrefetchHybridStrategy"
    assert rec["scheduler"] == "PDScopeScheduler"
    assert rec["layer"] == 3
    assert rec["phase"] == "prefill"
    assert "reason" in rec
    assert isinstance(rec["cpu_experts"], list)
    assert isinstance(rec["gpu_experts"], list)
    assert isinstance(rec["preload_experts"], list)
    assert isinstance(rec["current_demands"], list)
    assert isinstance(rec["future_demands"], list)
    assert "placement" in rec
    assert rec["placement"]["current_resident"] == [1]
    assert "latency" in rec
    assert rec["latency"]["t_io"] == 1.0
    assert "strategy_params" in rec
    assert rec["strategy_params"]["alpha"] == 0.1
    assert rec["strategy_params"]["t_attn"] == 0.6
    assert rec["strategy_params"]["r_hit"] == 0.8
    assert "counters" in rec


def test_no_recorder_means_no_overhead():
    dev, is_gpu = _make_strategy_args()
    strategy = GPUOnlyStrategy(dev, is_gpu)
    assert strategy.stats_recorder is None
    selected, weights = _make_tensors()
    cpu, gpu, assignments = strategy.decide_and_prepare(3, None, selected, weights, 8)
    assert cpu == []
    assert set(gpu) == {1, 3, 5}


def test_demand_to_dict():
    d = ExpertDemand(key=ExpertKey(layer=3, expert_id=7), token_count=42, score=0.95, source="current")
    result = _demand_to_dict(d)
    assert result == {"layer": 3, "expert_id": 7, "token_count": 42, "score": 0.95, "source": "current"}


def test_placement_summary():
    p = PlacementSnapshot(
        gpu_resident={(3, 1), (3, 2)},
        placeholder_resident={(3, 5)},
        loading={(4, 7)},
        free_placeholders=1,
    )
    demands = [ExpertDemand(ExpertKey(3, 1), 5), ExpertDemand(ExpertKey(3, 3), 3)]
    result = _placement_summary(p, demands)
    assert result["gpu_resident_count"] == 2
    assert result["placeholder_resident_count"] == 1
    assert result["loading_count"] == 1
    assert result["free_placeholders"] == 1
    assert result["current_resident"] == [1]


def test_latency_summary():
    m = ExpertLatencyModel(t_io=1.5, latency_cpu_table={1: 0.2, 4: 0.8}, latency_gpu_table={1: 0.1})
    result = _latency_summary(m)
    assert result["t_io"] == 1.5
    assert result["latency_cpu_table"]["1"] == 0.2
    assert result["latency_gpu_table"]["1"] == 0.1


def test_token_counts_by_expert():
    data = {3: torch.tensor([0, 1, 2]), 7: torch.tensor([0])}
    result = _token_counts_by_expert(data)
    assert result == {3: 3, 7: 1}


def test_prefill_phase_label_in_gpu_only():
    dev, is_gpu = _make_strategy_args()
    strategy = GPUOnlyStrategy(dev, is_gpu)
    recorder = ExpertSchedulingStatsRecorder()
    strategy.set_stats_recorder(recorder)
    selected, weights = _make_tensors()
    strategy.decide_and_prepare(3, None, selected, weights, 8, is_prefill=True)
    assert recorder.records[0]["phase"] == "prefill"
    strategy.decide_and_prepare(3, None, selected, weights, 8, is_prefill=False)
    assert recorder.records[1]["phase"] == "decode"


def test_recorder_disabled():
    recorder = ExpertSchedulingStatsRecorder()
    recorder.enabled = False
    recorder.record({"strategy": "test"})
    assert len(recorder.records) == 0
