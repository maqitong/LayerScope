import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, "src", "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from expert_latency import ExpertLatencyModel
from expert_scheduling import PDScopeScheduler
from expert_types import ExpertDemand, ExpertKey, ExpertLayerRequest, PlacementSnapshot, build_future_demands
from placeholder_manager import ExpertPlaceholderManager


def test_build_future_demands_aggregates_tensor_weights():
    predicted_experts = torch.tensor([[[2, 1], [2, 3]]])
    predicted_weights = torch.tensor([[[0.25, 0.5], [0.75, 0.125]]])

    demands = build_future_demands(4, predicted_experts, predicted_weights)

    by_expert = {d.key.expert_id: d for d in demands}
    assert set(by_expert) == {1, 2, 3}
    assert by_expert[1].token_count == 1
    assert by_expert[1].score == 0.5
    assert by_expert[2].token_count == 2
    assert by_expert[2].score == 1.0
    assert by_expert[3].token_count == 1
    assert by_expert[3].score == 0.125
    assert all(d.key.layer == 4 and d.source == "predicted" for d in demands)


def test_scheduler_returns_disjoint_current_experts():
    latency = ExpertLatencyModel(
        t_io=1.0,
        latency_cpu_table={1: 0.5, 4: 3.0},
        latency_gpu_table={1: 0.1, 4: 0.2},
    )
    request = ExpertLayerRequest(
        layer=1,
        phase="decode",
        current=[
            ExpertDemand(ExpertKey(1, 0), 1, source="current"),
            ExpertDemand(ExpertKey(1, 1), 1, source="current"),
        ],
        future=[ExpertDemand(ExpertKey(2, 3), 1, source="predicted")],
        assignments={},
    )
    placement = PlacementSnapshot(gpu_resident={(1, 0)}, free_placeholders=1)
    schedule = PDScopeScheduler().schedule(request, placement, latency)
    cpu = {(d.key.layer, d.key.expert_id) for d in schedule.cpu}
    gpu = {(d.key.layer, d.key.expert_id) for d in schedule.gpu}
    assert cpu.isdisjoint(gpu)
    assert cpu | gpu == {(1, 0), (1, 1)}


def test_placeholder_snapshot_treats_placeholder_as_gpu():
    class TinyExpert:
        def to(self, device):
            return self

    manager = ExpertPlaceholderManager(TinyExpert(), device="cpu", num_placeholders=1)
    placeholder = manager.acquire_placeholder(1, 7)
    assert placeholder is not None
    snapshot = manager.snapshot()
    assert snapshot.is_on_gpu(1, 7)
    assert manager.is_on_gpu(1, 7)
