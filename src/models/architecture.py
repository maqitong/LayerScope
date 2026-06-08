from typing import Any, Protocol

import torch.nn as nn


class ModelArchitectureAdapter(Protocol):
    """Model-specific structure access used by the shared MoE runtime."""

    model: nn.Module
    lm_head: nn.Module
    tokenizer: Any
    config: Any

    @property
    def layers(self):
        ...

    @property
    def first_moe_layer(self) -> int:
        ...

    @property
    def num_layers(self) -> int:
        ...

    @property
    def num_routed_experts(self) -> int:
        ...

    @property
    def num_shared_experts(self) -> int:
        ...

    @property
    def num_experts_per_tok(self) -> int:
        ...

    def get_template_expert(self) -> nn.Module:
        ...

    def get_routed_experts(self, layer: int) -> nn.ModuleList:
        ...

    def get_expert(self, layer: int, expert_id: int) -> nn.Module:
        ...

    def get_gate(self, layer: int) -> nn.Module:
        ...

    def get_shared_experts(self, layer: int) -> nn.Module:
        ...
