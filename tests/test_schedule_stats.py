import json
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
MODEL_DIR = os.path.join(ROOT, "src", "model")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
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
    assert result["current_static_resident"] == [1]
    assert result["current_placeholder_resident"] == []
    assert result["current_resident"] == [1]
    assert result["future_static_resident"] == []
    assert result["future_placeholder_resident"] == []


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


class TestHitSourceCounters:
    def _make_executor(self):
        from expert_executor import ExpertExecutionManager

        class FakePlaceholderManager:
            def is_static_gpu_resident(self, layer, expert_id):
                return expert_id == 0
            def get_placeholder_for_expert(self, layer, expert_id):
                if expert_id == 1:
                    return lambda x: x
                return None
            def acquire_placeholder(self, layer, expert_id):
                return lambda x: x
            def protect_expert(self, layer, expert_id):
                pass
            def unprotect_expert(self, layer, expert_id):
                pass
            def release_by_layer(self, layer):
                pass
            def load_weights(self, placeholder, expert):
                pass

        cpu_experts = {(1, eid): torch.nn.Linear(4, 4) for eid in range(64)}
        executor = ExpertExecutionManager(
            device="cpu",
            placeholder_manager=FakePlaceholderManager(),
            model=None,
            is_expert_in_gpu=lambda l, e: e == 0,
            cpu_experts=cpu_experts,
        )
        original_get = executor.get_cpu_expert
        def safe_get_cpu_expert(layer, expert_id):
            key = (layer, expert_id)
            if key in executor.cpu_experts:
                return executor.cpu_experts[key]
            return original_get(layer, expert_id)
        executor.get_cpu_expert = safe_get_cpu_expert
        return executor

    def _make_context(self, expert_ids, n_tokens=2):
        from expert_types import ExpertLayerContext, ExpertAssignment
        inps = torch.randn(n_tokens, 4)
        fake_experts = {eid: torch.nn.Linear(4, 4) for eid in expert_ids}
        assignments = {}
        for eid in expert_ids:
            assignments[eid] = ExpertAssignment(
                expert_id=eid,
                token_indices=torch.arange(n_tokens),
                routing_weights=torch.ones(n_tokens, 1),
            )
        return ExpertLayerContext(
            layer=1,
            experts=fake_experts,
            inps_flat=inps,
            hidden_dim=4,
            assignments=assignments,
        )

    def test_planned_static_gpu_counter(self):
        executor = self._make_executor()
        schedule = ExpertSchedule(gpu=[ExpertDemand(ExpertKey(1, 0), 2)])
        placement = PlacementSnapshot(gpu_resident={(1, 0)})
        executor.record_planned_execution_stats(schedule, placement)
        assert executor.planned_gpu_static_count == 1
        assert executor.planned_gpu_static_tokens == 2
        assert executor.planned_gpu_placeholder_count == 0
        assert executor.planned_gpu_ondemand_count == 0

    def test_planned_placeholder_counter(self):
        executor = self._make_executor()
        schedule = ExpertSchedule(gpu=[ExpertDemand(ExpertKey(1, 1), 2)])
        placement = PlacementSnapshot(placeholder_resident={(1, 1)})
        executor.record_planned_execution_stats(schedule, placement)
        assert executor.planned_gpu_placeholder_count == 1
        assert executor.planned_gpu_placeholder_tokens == 2
        assert executor.planned_gpu_static_count == 0
        assert executor.planned_gpu_ondemand_count == 0

    def test_planned_ondemand_counter(self):
        executor = self._make_executor()
        schedule = ExpertSchedule(gpu=[ExpertDemand(ExpertKey(1, 5), 2)])
        placement = PlacementSnapshot()
        executor.record_planned_execution_stats(schedule, placement)
        assert executor.planned_gpu_ondemand_count == 1
        assert executor.planned_gpu_ondemand_tokens == 2
        assert executor.planned_gpu_static_count == 0
        assert executor.planned_gpu_placeholder_count == 0

    def test_mixed_planned_counters(self):
        executor = self._make_executor()
        schedule = ExpertSchedule(
            cpu=[ExpertDemand(ExpertKey(1, 9), 4)],
            gpu=[
                ExpertDemand(ExpertKey(1, 0), 3),
                ExpertDemand(ExpertKey(1, 1), 3),
                ExpertDemand(ExpertKey(1, 5), 3),
            ],
            preload=[ExpertDemand(ExpertKey(2, 7), 1, source="predicted")],
        )
        placement = PlacementSnapshot(gpu_resident={(1, 0)}, placeholder_resident={(1, 1)})
        executor.record_planned_execution_stats(schedule, placement)
        assert executor.planned_gpu_static_count == 1
        assert executor.planned_gpu_placeholder_count == 1
        assert executor.planned_gpu_ondemand_count == 1
        assert executor.planned_gpu_static_tokens == 3
        assert executor.planned_gpu_placeholder_tokens == 3
        assert executor.planned_gpu_ondemand_tokens == 3
        assert executor.planned_cpu_count == 1
        assert executor.planned_cpu_tokens == 4
        assert executor.planned_preload_count == 1

    def test_wait_for_preload_returns_false_on_cpu(self):
        executor = self._make_executor()
        assert executor._wait_for_preload(1, 1) is False


class TestPreloadOverlap:
    def _make_records(self):
        return [
            {
                "call_index": 0,
                "layer": 3,
                "phase": "decode",
                "reason": "decode-mode-b",
                "preload_experts": [
                    {"expert_id": 1, "layer": 4, "token_count": 1, "score": 1.0, "source": "predicted"},
                    {"expert_id": 2, "layer": 4, "token_count": 1, "score": 0.9, "source": "predicted"},
                ],
                "gpu_experts": [],
                "cpu_experts": [],
                "current_demands": [],
            },
            {
                "call_index": 1,
                "layer": 4,
                "phase": "decode",
                "reason": "decode-mode-c",
                "preload_experts": [],
                "gpu_experts": [{"expert_id": 2}],
                "cpu_experts": [{"expert_id": 3}],
                "current_demands": [
                    {"expert_id": 2, "layer": 4, "token_count": 1, "score": 1.0, "source": "current"},
                    {"expert_id": 3, "layer": 4, "token_count": 1, "score": 0.8, "source": "current"},
                ],
            },
        ]

    def test_basic_overlap(self):
        SCRIPTS_DIR = os.path.join(ROOT, "src", "scripts")
        if SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, SCRIPTS_DIR)
        from analyze_schedule import compute_preload_overlap

        records = self._make_records()
        result = compute_preload_overlap(records)
        assert result["preload_calls"] == 1
        assert result["total_preloaded"] == 2
        assert result["matched_next_calls"] == 1
        assert result["overlap_count"] == 1
        assert result["total_next_demands"] == 2
        assert result["precision"] == 0.5
        assert result["next_coverage"] == 0.5
        assert result["wasted_preload"] == 1

    def test_no_preload_records(self):
        SCRIPTS_DIR = os.path.join(ROOT, "src", "scripts")
        if SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, SCRIPTS_DIR)
        from analyze_schedule import compute_preload_overlap

        result = compute_preload_overlap([{"layer": 1, "phase": "decode", "preload_experts": []}])
        assert result["preload_calls"] == 0
        assert result["precision"] == 0.0

    def test_unmatched_next_layer(self):
        SCRIPTS_DIR = os.path.join(ROOT, "src", "scripts")
        if SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, SCRIPTS_DIR)
        from analyze_schedule import compute_preload_overlap

        records = [
            {
                "call_index": 0,
                "layer": 3,
                "phase": "decode",
                "reason": "decode-mode-b",
                "preload_experts": [{"expert_id": 1, "layer": 4, "token_count": 1, "score": 1.0, "source": "predicted"}],
                "gpu_experts": [],
                "cpu_experts": [],
                "current_demands": [],
            },
        ]
        result = compute_preload_overlap(records)
        assert result["preload_calls"] == 1
        assert result["total_preloaded"] == 1
        assert result["matched_next_calls"] == 0
        assert result["overlap_count"] == 0
        assert result["wasted_preload"] == 1

    def test_by_reason_breakdown(self):
        SCRIPTS_DIR = os.path.join(ROOT, "src", "scripts")
        if SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, SCRIPTS_DIR)
        from analyze_schedule import compute_preload_overlap

        records = [
            {
                "call_index": 0,
                "layer": 3,
                "phase": "decode",
                "reason": "decode-mode-b",
                "preload_experts": [{"expert_id": 1, "layer": 4, "token_count": 1, "score": 1.0, "source": "predicted"}],
                "gpu_experts": [],
                "cpu_experts": [],
                "current_demands": [],
            },
            {
                "call_index": 1,
                "layer": 4,
                "phase": "decode",
                "reason": "decode-mode-c",
                "preload_experts": [],
                "gpu_experts": [{"expert_id": 1}],
                "cpu_experts": [],
                "current_demands": [{"expert_id": 1, "layer": 4, "token_count": 1, "score": 1.0, "source": "current"}],
            },
        ]
        result = compute_preload_overlap(records)
        assert "decode-mode-b" in result["by_reason"]
        assert result["by_reason"]["decode-mode-b"]["calls"] == 1
        assert result["by_reason"]["decode-mode-b"]["overlap"] == 1

    def test_examples_with_limit(self):
        SCRIPTS_DIR = os.path.join(ROOT, "src", "scripts")
        if SCRIPTS_DIR not in sys.path:
            sys.path.insert(0, SCRIPTS_DIR)
        from analyze_schedule import compute_preload_overlap

        records = self._make_records()
        result = compute_preload_overlap(records, limit=5)
        assert len(result["examples"]) == 1
        ex = result["examples"][0]
        assert ex["overlap"] == [2]
        assert ex["precision"] == 0.5
