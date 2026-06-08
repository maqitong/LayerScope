from abc import ABC, abstractmethod
from typing import List

import torch
from common.types import ExpertDemand, build_future_demands


class ExpertPredictor(ABC):
    """专家预测器抽象基类"""

    @abstractmethod
    def predict(
        self,
        hidden_states: torch.Tensor,
        expert_provider,
        current_layer: int,
        lookahead: int = 1,
    ) -> List[ExpertDemand]:
        raise NotImplementedError


class GatePredictor(ExpertPredictor):
    """使用下一层 gate 网络预测下一层活跃专家"""

    def predict(
        self,
        hidden_states: torch.Tensor,
        expert_provider,
        current_layer: int,
        lookahead: int = 1,
    ) -> List[ExpertDemand]:
        next_layer_idx = current_layer + lookahead
        num_layers = getattr(expert_provider, "num_layers", None)
        if num_layers is None:
            num_layers = len(expert_provider.layers)
        if next_layer_idx >= num_layers:
            return []
        if hasattr(expert_provider, "get_gate"):
            gate = expert_provider.get_gate(next_layer_idx)
        else:
            gate = expert_provider.layers[next_layer_idx].mlp.gate
        predicted_experts, routing_weights = gate(hidden_states)
        return build_future_demands(next_layer_idx, predicted_experts, routing_weights)
