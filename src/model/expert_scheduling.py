from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.expert_latency import ExpertLatencyModel
from model.expert_types import (
    ExpertDemand,
    ExpertKey,
    ExpertLayerRequest,
    ExpertSchedule,
    PlacementSnapshot,
    build_assignments,
    build_current_demands,
    build_future_demands,
    unique_demands,
)


def _demand_to_dict(d: ExpertDemand) -> Dict:
    return {
        "layer": d.key.layer,
        "expert_id": d.key.expert_id,
        "token_count": d.token_count,
        "score": round(d.score, 6),
        "source": d.source,
    }


def _placement_summary(placement: Optional[PlacementSnapshot], current_demands: Optional[List[ExpertDemand]] = None, future_demands: Optional[List[ExpertDemand]] = None) -> Dict:
    if placement is None:
        return {}
    current_layer = current_demands[0].key.layer if current_demands else -1
    future_layer = future_demands[0].key.layer if future_demands else current_layer + 1
    current_resident = [d.key.expert_id for d in (current_demands or [])
                        if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
    future_resident = [d.key.expert_id for d in (future_demands or [])
                       if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
    return {
        "gpu_resident_count": len(placement.gpu_resident),
        "placeholder_resident_count": len(placement.placeholder_resident),
        "loading_count": len(placement.loading),
        "cpu_resident_count": len(placement.cpu_resident),
        "ssd_resident_count": len(placement.ssd_resident),
        "free_placeholders": placement.free_placeholders,
        "current_resident": sorted(current_resident),
        "future_resident": sorted(future_resident),
    }


def _latency_summary(latency_model: Optional[ExpertLatencyModel]) -> Dict:
    if latency_model is None:
        return {}
    return {
        "t_io": latency_model.t_io,
        "latency_cpu_table": {str(k): v for k, v in latency_model.latency_cpu_table.items()},
        "latency_gpu_table": {str(k): v for k, v in latency_model.latency_gpu_table.items()},
    }


def _token_counts_by_expert(token_indices_by_expert: Dict[int, torch.Tensor]) -> Dict[int, int]:
    return {eid: int(t.shape[0]) for eid, t in token_indices_by_expert.items()}


class ExpertSchedulingStatsRecorder:
    """Opt-in recorder that collects structured scheduling decision records.

    Each call to ``record()`` appends one JSON-serialisable dict to an
    internal list.  If *output_path* is provided the record is also appended
    as a single JSON line to that file (JSONL format).
    """

    def __init__(self, output_path: Optional[str] = None, runtime_meta: Optional[Dict] = None):
        self.enabled = True
        self.output_path = output_path
        self.runtime_meta: Dict = runtime_meta or {}
        self._records: List[Dict] = []
        self._call_index = 0
        self._file_handle = None
        if self.output_path is not None:
            import os
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            self._file_handle = open(self.output_path, "a", encoding="utf-8")

    def record(self, record: Dict) -> None:
        if not self.enabled:
            return
        record["call_index"] = self._call_index
        record["timestamp"] = time.time()
        if self.runtime_meta:
            record["runtime"] = self.runtime_meta
        self._call_index += 1
        self._records.append(record)
        if self._file_handle is not None:
            self._file_handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._file_handle.flush()

    @property
    def records(self) -> List[Dict]:
        return list(self._records)

    def reset(self) -> None:
        self._records.clear()
        self._call_index = 0

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def summary(self) -> Dict:
        if not self._records:
            return {"total_calls": 0}
        by_key: Dict[Tuple, Dict] = {}
        for rec in self._records:
            key = (rec.get("strategy", ""), rec.get("phase", ""), rec.get("layer", -1))
            bucket = by_key.setdefault(key, {"count": 0, "reasons": {}, "gpu_total": 0, "cpu_total": 0, "preload_total": 0, "unique_gpu": set(), "unique_cpu": set(), "unique_preload": set()})
            bucket["count"] += 1
            reason = rec.get("reason", "")
            bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
            bucket["gpu_total"] += len(rec.get("gpu_experts", []))
            bucket["cpu_total"] += len(rec.get("cpu_experts", []))
            bucket["preload_total"] += len(rec.get("preload_experts", []))
            for e in rec.get("gpu_experts", []):
                eid = e.get("expert_id", e) if isinstance(e, dict) else e
                bucket["unique_gpu"].add(eid)
            for e in rec.get("cpu_experts", []):
                eid = e.get("expert_id", e) if isinstance(e, dict) else e
                bucket["unique_cpu"].add(eid)
            for e in rec.get("preload_experts", []):
                eid = e.get("expert_id", e) if isinstance(e, dict) else e
                bucket["unique_preload"].add(eid)
        rows = []
        for (strategy, phase, layer), bucket in sorted(by_key.items()):
            rows.append({
                "strategy": strategy,
                "phase": phase,
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


def _build_expert_mask(selected_experts: torch.Tensor, n_expert: int) -> torch.Tensor:
    """构建专家掩码张量，将 selected_experts 转换为 one-hot 编码并调整维度顺序

    Args:
        selected_experts: 形状 [batch_size, seq_len, 2]，每个 token 选择的 top-2 专家索引
        n_expert: 总专家数

    Returns:
        形状 [n_expert, 2, batch_size*seq_len] 的 one-hot 掩码张量
    """
    return torch.nn.functional.one_hot(selected_experts, num_classes=n_expert).permute(2, 1, 0)


def _collect_expert_assignments(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    n_expert: int,
) -> Tuple[List[int], Dict[int, torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """直接从稀疏 top-k 路由结果构建活跃专家和 token 分配。

    避免构造 [n_expert, top_k, n_token] 的 dense one-hot mask，也避免
    _collect_active_experts 与 _organize_token_assignments 重复 torch.where。
    """
    flat_experts = selected_experts.reshape(-1)
    top_k = selected_experts.shape[-1]
    flat_weights = routing_weights.reshape(-1)

    if flat_experts.numel() == 0:
        return [], {}, {}

    sorted_experts, order = torch.sort(flat_experts)
    if sorted_experts.numel() == 0:
        return [], {}, {}

    unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    token_positions = torch.div(order, top_k, rounding_mode="floor")
    routing_weight_values = flat_weights.index_select(0, order).unsqueeze(-1)

    unique_cpu = unique_experts.detach().cpu().tolist()
    counts_cpu = counts.detach().cpu().tolist()
    active_experts = [int(expert_id) for expert_id in unique_cpu]
    token_indices_by_expert = {}
    expert_assignments = {}
    start = 0
    for i_expert, count in zip(active_experts, counts_cpu):
        end = start + int(count)
        token_indices = token_positions[start:end]
        routing_weight_subset = routing_weight_values[start:end]
        token_indices_by_expert[i_expert] = token_indices
        expert_assignments[i_expert] = (token_indices, routing_weight_subset)
        start = end

    return active_experts, token_indices_by_expert, expert_assignments


def _collect_active_experts(expert_mask: torch.Tensor, n_expert: int) -> Tuple[List[int], Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """收集有 token 分配的活跃专家及其对应的 token 索引
    
    Args:
        expert_mask: 形状 [n_expert, 2, batch_size*seq_len] 的专家掩码
        n_expert: 总专家数
    
    Returns:
        active_experts: 活跃专家索引列表
        idxs: 字典 {专家索引: 该专家在 top-2 中的位置索引}
        top_2s: 字典 {专家索引: 分配给该专家的 token 索引}
    """
    idxs = {}
    top_2s = {}
    active_experts = []
    for i_expert in range(n_expert):
        idx, top_2 = torch.where(expert_mask[i_expert])
        if top_2.shape[0] > 0:
            idxs[i_expert] = idx
            top_2s[i_expert] = top_2
            active_experts.append(i_expert)
    return active_experts, idxs, top_2s


def _organize_token_assignments(expert_mask: torch.Tensor, routing_weights: torch.Tensor, active_experts: List[int]) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """为每个活跃专家组织 token 分配信息和对应的路由权重
    
    Args:
        expert_mask: 形状 [n_expert, 2, batch_size*seq_len] 的专家掩码
        routing_weights: 形状 [batch_size, seq_len, 2] 的路由权重
        active_experts: 活跃专家索引列表
    
    Returns:
        expert_assignments: 字典 {专家索引: (token索引张量, 对应的路由权重张量)}
    """
    expert_assignments = {}
    for i_expert in active_experts:
        idx, top_2 = torch.where(expert_mask[i_expert])
        routing_weight_subset = routing_weights[top_2, idx, None]
        expert_assignments[i_expert] = (top_2, routing_weight_subset)
    return expert_assignments


class ExpertSchedulingStrategy:
    """专家调度策略基类，定义策略接口"""
    def __init__(self, dev, is_expert_in_gpu):
        self.dev = dev
        self.is_expert_in_gpu = is_expert_in_gpu
        self.stats_recorder: Optional[ExpertSchedulingStatsRecorder] = None

    def set_stats_recorder(self, recorder: Optional[ExpertSchedulingStatsRecorder]):
        self.stats_recorder = recorder

    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        **kwargs,
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """策略决策与预处理接口
        
        Args:
            i_layer: 当前层索引
            experts: 专家模块列表
            selected_experts: 每个 token 选择的 top-2 专家
            routing_weights: 每个 token 对所选专家的路由权重
            n_expert: 总专家数
        
        Returns:
            cpu_experts: 在 CPU 上执行的专家索引列表
            gpu_experts: 在 GPU 上执行的专家索引列表
            expert_assignments: 每个专家的 token 分配信息
        """
        raise NotImplementedError


class GPUOnlyStrategy(ExpertSchedulingStrategy):
    """纯 GPU 调度策略：所有专家都在 GPU 上执行"""
    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        **kwargs,
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """决策：所有活跃专家都在 GPU 上执行"""
        active_experts, token_indices_by_expert, expert_assignments = _collect_expert_assignments(
            selected_experts, routing_weights, n_expert
        )
        if self.stats_recorder is not None:
            current_demands = build_current_demands(i_layer, active_experts, token_indices_by_expert)
            self.stats_recorder.record({
                "strategy": self.__class__.__name__,
                "reason": "gpu-only",
                "layer": i_layer,
                "phase": kwargs.get("is_prefill", True) and "prefill" or "decode",
                "n_expert": n_expert,
                "seq_len": selected_experts.shape[1] if selected_experts.dim() >= 2 else 1,
                "top_k": selected_experts.shape[-1] if selected_experts.dim() >= 1 else 0,
                "active_experts": active_experts,
                "cpu_experts": [],
                "gpu_experts": active_experts,
                "preload_experts": [],
                "token_counts": _token_counts_by_expert(token_indices_by_expert),
                "placement": _placement_summary(kwargs.get("placement"), current_demands),
            })
        return [], active_experts, expert_assignments


class FiddlerStrategy(ExpertSchedulingStrategy):
    """CPU-GPU 混合调度策略：通过代价优化决定每个专家在 CPU 还是 GPU 上执行
    
    职责：
    - 构建每个活跃专家的 CPU/GPU 执行代价表
    - 按每个活跃专家独立选择代价更低的 CPU/GPU 执行位置
    - 统计 GPU 缓存命中率
    """
    def __init__(self, dev, is_expert_in_gpu, latency_cpu, latency_gpu, latency_io):
        super().__init__(dev, is_expert_in_gpu)
        self.latency_cpu = latency_cpu  # CPU 上每个 token 的延迟
        self.latency_gpu = latency_gpu  # GPU 上执行专家的固定延迟
        self.latency_io = latency_io  # 将专家权重从 CPU 拷贝到 GPU 的固定延迟
        self.cnt_expert_hit = 0  # GPU 缓存命中的 token 数
        self.cnt_expert_all = 0  # 总 token 数

    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        **kwargs,
    ) -> Tuple[List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """决策：逐专家选择代价最小的 CPU/GPU 分配方案"""
        active_experts, token_indices_by_expert, expert_assignments = _collect_expert_assignments(
            selected_experts, routing_weights, n_expert
        )
        
        cpu_experts = []
        gpu_experts = []
        per_expert_details = []
        for i_expert in active_experts:
            token_count = token_indices_by_expert[i_expert].shape[0]
            cost_cpu = token_count * self.latency_cpu
            cost_gpu = self.latency_gpu + self.latency_io
            resident = self.is_expert_in_gpu(i_layer, i_expert)
            if resident:
                cost_gpu = 0
                self.cnt_expert_hit += token_count
            self.cnt_expert_all += token_count

            if cost_cpu < cost_gpu:
                cpu_experts.append(i_expert)
                decision = "cpu"
            else:
                gpu_experts.append(i_expert)
                decision = "gpu"
            per_expert_details.append({
                "expert_id": i_expert,
                "token_count": token_count,
                "resident": resident,
                "cost_cpu": round(cost_cpu, 6),
                "cost_gpu": round(cost_gpu, 6),
                "decision": decision,
            })

        if self.stats_recorder is not None:
            self.stats_recorder.record({
                "strategy": self.__class__.__name__,
                "reason": "fiddler-cost-opt",
                "layer": i_layer,
                "phase": kwargs.get("is_prefill", True) and "prefill" or "decode",
                "n_expert": n_expert,
                "seq_len": selected_experts.shape[1] if selected_experts.dim() >= 2 else 1,
                "top_k": selected_experts.shape[-1] if selected_experts.dim() >= 1 else 0,
                "active_experts": active_experts,
                "cpu_experts": cpu_experts,
                "gpu_experts": gpu_experts,
                "preload_experts": [],
                "token_counts": _token_counts_by_expert(token_indices_by_expert),
                "placement": _placement_summary(kwargs.get("placement")),
                "strategy_params": {
                    "latency_cpu": self.latency_cpu,
                    "latency_gpu": self.latency_gpu,
                    "latency_io": self.latency_io,
                },
                "per_expert_details": per_expert_details,
                "counters": {
                    "expert_hit": self.cnt_expert_hit,
                    "expert_all": self.cnt_expert_all,
                },
            })
        
        return cpu_experts, gpu_experts, expert_assignments


def _build_latency_lookup(entries: list) -> Dict[int, float]:
    """从 benchmark JSON 数组构建 {token_count: avg_time_ms} 查找表

    Args:
        entries: benchmark 数据列表，每项包含 token_count 和 avg_time_ms

    Returns:
        {token_count: avg_time_ms} 字典
    """
    table = {}
    for entry in entries:
        tc = entry["token_count"]
        table[tc] = entry["avg_time_ms"]
    return table


def _lookup_latency(table: Dict[int, float], token_count: int) -> float:
    """从查找表获取延迟值

    查找策略：
    - 精确匹配：直接返回
    - 超出最大值：基于最大 token_count 条目线性外推
    - 其他：使用最接近的 token_count 条目

    Args:
        table: {token_count: avg_time_ms} 查找表
        token_count: 要查询的 token 数量

    Returns:
        对应的延迟值（毫秒）
    """
    if token_count in table:
        return table[token_count]
    max_tc = max(table.keys())
    if token_count >= max_tc:
        return table[max_tc] * token_count / max_tc
    closest = min(table.keys(), key=lambda k: abs(k - token_count))
    return table[closest]


class ExpertScheduler(ABC):
    """阶段感知专家调度器抽象接口

    职责：
    - 接收结构化的调度请求（ExpertLayerRequest）和驻留快照（PlacementSnapshot）
    - 结合延迟模型做出调度决策
    - 输出 ExpertSchedule，包含 CPU/GPU/preload/evict 四类专家列表
    - 只负责"决定"，不负责"执行"
    """

    @abstractmethod
    def schedule(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """根据请求、驻留状态和延迟模型生成调度计划"""
        raise NotImplementedError


class PDScopeScheduler(ExpertScheduler):
    """PDScope 论文风格的阶段感知调度器

    职责：
    - 区分 prefill 和 decode 阶段，分别执行不同的调度策略
    - prefill：三步调度法（全局队列 → 当前按需 → 预加载）
    - decode：ABCD 模式决策（当前/下一层驻留是否充足）
    - 结合 I/O 气泡窗口和命中率决定预加载数量

    参数：
    - alpha: GPU 启动开销（秒）
    - t_attn: 注意力计算时间窗口，用于计算预加载可用时间
    - r_hit: 预期的 placeholder 命中率
    """

    def __init__(self, alpha: float = 0.1, t_attn: float = 0.6, r_hit: float = 0.8):
        self.alpha = alpha
        self.t_attn = t_attn
        self.r_hit = r_hit

    def schedule(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """根据阶段分发到 prefill 或 decode 调度逻辑"""
        if request.phase == "prefill":
            return self.schedule_prefill(request, placement, latency)
        return self.schedule_decode(request, placement, latency)

    def schedule_prefill(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """Prefill 阶段调度：三步调度法

        步骤：
        1. 分离当前驻留/非驻留专家和未来非驻留专家
        2. 合并排序后构造全局候选队列
        3. 从全局队列中选出当前层值得按需加载到 GPU 的专家
        4. 根据 I/O 气泡窗口选择未来专家预加载
        """
        current = request.current
        current_resident = [d for d in current if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        current_non_resident = [d for d in current if not placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        future_non_resident = [
            d for d in request.future
            if not placement.is_on_gpu(d.key.layer, d.key.expert_id)
            and (d.key.layer, d.key.expert_id) not in placement.loading
        ]

        combined = current_non_resident + future_non_resident
        combined.sort(key=lambda d: (d.token_count, d.score))

        global_queue = self._select_global_queue(combined, latency)
        current_global = [d for d in global_queue if d.source == "current"]
        ondemand, t_gpu, t_cpu = self._select_current_ondemand(current_global, latency)

        gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
        gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
        cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
        preload = self._select_preload(global_queue, t_gpu, t_cpu, placement, latency)

        return ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="prefill")

    def schedule_decode(
        self,
        request: ExpertLayerRequest,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> ExpertSchedule:
        """Decode 阶段调度：ABCD 模式决策

        步骤：
        1. 计算理想 GPU 专家数 n_g_rho（最小化 max(GPU时间, CPU时间)）
        2. 评估当前层和下一层的驻留是否充足
        3. 根据四种情况决策：
           - 当前不足 + 下一层不足 → fallback 到 prefill 策略
           - 当前不足 + 下一层充足 → 补当前层 GPU 专家（mode-a）
           - 当前充足 + 下一层不足 → 预加载下一层专家（mode-b）
           - 当前充足 + 下一层充足 → 不额外操作（mode-c）
        """
        current = request.current
        if not current:
            print(f"[DecodeSchedule] layer={request.layer} reason=decode-empty")
            return ExpertSchedule(reason="decode-empty")

        k = len(current)
        t_c = latency.cpu(1)
        t_g = latency.gpu_compute(1)
        n_g_rho = min(
            range(k + 1),
            key=lambda n_g: max(n_g * t_g, (k - n_g) * t_c),
        ) # 计算理想 GPU 专家数，最小化 max(GPU时间, CPU时间)

        current_resident = [d for d in current if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        current_non_resident = [d for d in current if not placement.is_on_gpu(d.key.layer, d.key.expert_id)]
        future_non_resident = [
            d for d in sorted(request.future, key=lambda x: (x.score, x.token_count), reverse=True)
            if not placement.is_on_gpu(d.key.layer, d.key.expert_id)
            and (d.key.layer, d.key.expert_id) not in placement.loading
        ]
        next_resident_count = sum(
            1 for d in request.future
            if placement.is_on_gpu(d.key.layer, d.key.expert_id)
        )

        cur_below = len(current_resident) <= n_g_rho # 当前层驻留专家数不足以满足理想 GPU 专家数
        next_below = next_resident_count <= n_g_rho # 下一层驻留专家数不足以满足理想 GPU 专家数

        current_ids = [d.key.expert_id for d in current]
        current_resident_ids = [d.key.expert_id for d in current_resident]
        current_non_resident_ids = [d.key.expert_id for d in current_non_resident]
        future_ids = [d.key.expert_id for d in request.future]
        future_non_resident_ids = [d.key.expert_id for d in future_non_resident]
        # if request.layer < 5:  # 仅打印前几层的调度决策以避免日志过多
        #     print(
        #         "[DecodeSchedule] "
        #         f"layer={request.layer} k={k} t_cpu_1={t_c:.4f} t_gpu_1={t_g:.4f} "
        #         f"t_io={latency.t_io:.4f} n_g_rho={n_g_rho} "
        #         f"current={current_ids} current_resident={current_resident_ids} "
        #         f"current_non_resident={current_non_resident_ids} "
        #         f"future={future_ids} future_non_resident={future_non_resident_ids} "
        #         f"next_resident_count={next_resident_count} "
        #         f"cur_below={cur_below} next_below={next_below}"
        #     )

        if cur_below and next_below:
        # 当前层和下一层驻留都不足，回退到 prefill 策略，尝试通过预加载来提升未来层驻留，从而间接提升当前层驻留
            schedule = self.schedule_prefill(request, placement, latency)
            schedule.reason = "decode-fallback-prefill"
            # if request.layer < 5:  # 仅打印前几层的调度决策以避免日志过多
            #     print(
            #         "[DecodeSchedule] "
            #         f"layer={request.layer} mode={schedule.reason} "
            #         f"gpu={schedule.gpu_expert_ids} cpu={schedule.cpu_expert_ids} "
            #         f"preload={schedule.preload_expert_ids}"
            #     )

            return schedule
        if cur_below and not next_below:
        # 当前层驻留不足但下一层充足，优先补齐当前层 GPU 专家（mode-a），从全局候选中选取分数最高的 n_g_rho 个专家放入 GPU，剩余放 CPU
            need = min(n_g_rho - len(current_resident), len(current_non_resident))
            ondemand = current_non_resident[:need]
            gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
            gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
            cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
            schedule = ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-a")
            # if request.layer < 5:  # 仅打印前几层的调度决策以避免日志过多
            #     print(
            #         "[DecodeSchedule] "
            #         f"layer={request.layer} mode={schedule.reason} need_current={need} "
            #         f"ondemand={[d.key.expert_id for d in ondemand]} "
            #         f"gpu={schedule.gpu_expert_ids} cpu={schedule.cpu_expert_ids} preload=[]"
            #     )
            return schedule
        if not cur_below and next_below:
        # 当前层驻留充足但下一层不足，预加载下一层专家（mode-b）
            need_next = max(0, n_g_rho - next_resident_count)
            preload = future_non_resident[:need_next]
            gpu = current_resident[:n_g_rho] # 当前层 GPU 专家保持在理想数量 n_g_rho，剩余放 CPU
            gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
            cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
            schedule = ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="decode-mode-b")
            # if request.layer < 5:  # 仅打印前几层的调度决策以避免日志过多
            #     print(
            #         "[DecodeSchedule] "
            #         f"layer={request.layer} mode={schedule.reason} need_next={need_next} "
            #         f"gpu={schedule.gpu_expert_ids} cpu={schedule.cpu_expert_ids} "
            #         f"preload={schedule.preload_expert_ids}"
            #     )
            return schedule

        gpu = current_resident[:n_g_rho]
        gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
        cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
        schedule = ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-c")
        # if request.layer < 5:  # 仅打印前几层的调度决策以避免日志过多
        #     print(
        #         "[DecodeSchedule] "
        #         f"layer={request.layer} mode={schedule.reason} "
        #         f"gpu={schedule.gpu_expert_ids} cpu={schedule.cpu_expert_ids} preload=[]"
        #     )
        return schedule

    def _select_global_queue(
        self,
        demands: List[ExpertDemand],
        latency: ExpertLatencyModel,
    ) -> List[ExpertDemand]:
        """构造全局候选队列：从合并需求中选出值得 GPU 化的专家子集

        从右向左扫描，找到第一个满足 T_gpu < T_cpu 的分割点，
        分割点右侧的专家放入 GPU 候选队列。
        """
        if not demands:
            return []
        for i in range(len(demands)):
            gpu_side = demands[i:]
            cpu_side = demands[:i]
            t_gpu = self.alpha + len(gpu_side) * latency.t_io + latency.gpu_compute(1)
            t_cpu = sum(latency.cpu(d.token_count) for d in cpu_side) + self.t_attn
            if t_gpu < t_cpu:
                return gpu_side
        return []

    def _select_current_ondemand(
        self,
        current_global: List[ExpertDemand],
        latency: ExpertLatencyModel,
    ) -> Tuple[List[ExpertDemand], float, float]:
        """从当前层全局候选中选出值得按需加载到 GPU 的专家

        遍历所有分割点，找到第一个 GPU 时间 < CPU 时间的位置，
        该位置右侧的当前层专家值得按需加载到 GPU 执行。

        Returns:
            ondemand: 按需加载到 GPU 的当前层专家列表
            t_gpu: GPU 侧执行时间
            t_cpu: CPU 侧执行时间
        """
        if not current_global:
            return [], 0.0, 0.0
        last_t_gpu = 0.0
        last_t_cpu = sum(latency.cpu(d.token_count) for d in current_global)
        for i in range(len(current_global) + 1):
            gpu_side = current_global[i:]
            cpu_side = current_global[:i]
            t_compute = sum(latency.gpu_compute(d.token_count) for d in gpu_side)
            t_io = len(gpu_side) * latency.t_io
            t_gpu = max(t_compute, self.alpha + t_io) + latency.gpu_compute(1) if gpu_side else 0.0
            t_cpu = sum(latency.cpu(d.token_count) for d in cpu_side)
            last_t_gpu = t_gpu
            last_t_cpu = t_cpu
            if gpu_side and t_gpu < t_cpu:
                return gpu_side, t_gpu, t_cpu
        return [], last_t_gpu, last_t_cpu

    def _select_preload(
        self,
        global_queue: List[ExpertDemand],
        t_gpu: float,
        t_cpu: float,
        placement: PlacementSnapshot,
        latency: ExpertLatencyModel,
    ) -> List[ExpertDemand]:
        """根据 I/O 气泡窗口和 placeholder 容量选择可预加载的未来专家

        计算逻辑：
        1. t_gap = max(0, t_cpu - t_gpu) 为可用气泡时间
        2. capacity = min(空闲placeholder数, 气泡内可传输的专家数)
        3. 从未来预测专家中按 score 降序选取最多 capacity 个
        """
        # Latency model values are milliseconds: benchmark tables store avg_time_ms
        # and t_io is loaded from expert_weight_copy.avg_ms.
        t_gap = max(0.0, t_cpu - t_gpu)
        # print(f"[PreloadSelect] t_gpu={t_gpu:.4f}ms t_cpu={t_cpu:.4f}ms")
        capacity = math.floor((t_gap + self.t_attn) / max(latency.t_io, 1e-9))
        if capacity <= 0:
            return []
        xi = (2 * self.r_hit - 1) * latency.t_io
        if xi <= 0:
            return []
        future = [d for d in global_queue if d.source == "predicted"]
        future.sort(key=lambda d: (d.score, d.token_count), reverse=True)
        return future[:capacity]


class PrefetchHybridStrategy(ExpertSchedulingStrategy):
    """PDScope AdaptSched 调度策略：区分 Prefill 三步调度法和 Decode ABC 策略

    职责：
    - 作为 deepseek.py 与 PDScopeScheduler 之间的适配层
    - 从 gate 输出构造当前专家需求
    - 接收预测出的未来专家需求
    - 接收 placeholder_manager 的驻留快照
    - 构造 ExpertLayerRequest 并调用 PDScopeScheduler
    - 把 ExpertSchedule 转回 deepseek.py 使用的返回格式

    参数：
    - t_io: 专家权重传输延迟（秒）
    - latency_cpu_table: CPU 延迟查找表 {token_count: time_ms}
    - latency_gpu_table: GPU 延迟查找表 {token_count: time_ms}
    """

    def __init__(self, dev, is_expert_in_gpu, t_io: float,
                 latency_cpu_table: Dict[int, float],
                 latency_gpu_table: Dict[int, float]):
        super().__init__(dev, is_expert_in_gpu)
        self.t_io = t_io
        self.latency_cpu_table = latency_cpu_table
        self.latency_gpu_table = latency_gpu_table
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.latency_model = ExpertLatencyModel(t_io, latency_cpu_table, latency_gpu_table)
        self.scheduler = PDScopeScheduler()

    def decide_and_prepare(
        self,
        i_layer: int,
        experts: nn.ModuleList,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        n_expert: int,
        predicted_next_experts: Optional[torch.Tensor] = None,
        predicted_next_weights: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        future_demands: Optional[List[ExpertDemand]] = None,
        placement: Optional[PlacementSnapshot] = None,
    ) -> Tuple[List[int], List[int], List[int], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """策略决策入口：构造请求并调用 PDScopeScheduler"""
        active_experts, token_indices_by_expert, raw_assignments = _collect_expert_assignments(
            selected_experts, routing_weights, n_expert
        )
        current_demands = build_current_demands(i_layer, active_experts, token_indices_by_expert)
        future = future_demands
        if future is None:
            future = build_future_demands(i_layer + 1, predicted_next_experts, predicted_next_weights)
        future = unique_demands(future)
        if placement is None:
            gpu_resident = set()
            for demand in current_demands + future:
                if self.is_expert_in_gpu(demand.key.layer, demand.key.expert_id):
                    gpu_resident.add((demand.key.layer, demand.key.expert_id))
            placement = PlacementSnapshot(gpu_resident=gpu_resident)

        for demand in current_demands:
            on_gpu = placement.is_on_gpu(demand.key.layer, demand.key.expert_id)
            if on_gpu:
                self.cnt_expert_hit += demand.token_count
            self.cnt_expert_all += demand.token_count

        request = ExpertLayerRequest(
            layer=i_layer,
            phase="prefill" if is_prefill else "decode",
            current=current_demands,
            future=future,
            assignments=build_assignments(raw_assignments),
        )
        schedule = self.scheduler.schedule(request, placement, self.latency_model)

        if self.stats_recorder is not None:
            self.stats_recorder.record({
                "strategy": self.__class__.__name__,
                "scheduler": self.scheduler.__class__.__name__,
                "reason": schedule.reason,
                "layer": i_layer,
                "phase": "prefill" if is_prefill else "decode",
                "n_expert": n_expert,
                "seq_len": selected_experts.shape[1] if selected_experts.dim() >= 2 else 1,
                "top_k": selected_experts.shape[-1] if selected_experts.dim() >= 1 else 0,
                "active_experts": active_experts,
                "cpu_experts": [_demand_to_dict(d) for d in schedule.cpu],
                "gpu_experts": [_demand_to_dict(d) for d in schedule.gpu],
                "preload_experts": [_demand_to_dict(d) for d in schedule.preload],
                "current_demands": [_demand_to_dict(d) for d in current_demands],
                "future_demands": [_demand_to_dict(d) for d in future],
                "token_counts": _token_counts_by_expert(token_indices_by_expert),
                "placement": _placement_summary(placement, current_demands, future),
                "latency": _latency_summary(self.latency_model),
                "strategy_params": {
                    "alpha": self.scheduler.alpha,
                    "t_attn": self.scheduler.t_attn,
                    "r_hit": self.scheduler.r_hit,
                },
                "counters": {
                    "expert_hit": self.cnt_expert_hit,
                    "expert_all": self.cnt_expert_all,
                },
            })

        return schedule.cpu_expert_ids, schedule.gpu_expert_ids, schedule.preload_expert_ids, raw_assignments
