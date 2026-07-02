from typing import List, Dict, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.expert_latency import ExpertLatencyModel
from model.expert_monitor import ExpertSchedulingMonitor
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


def _collect_expert_assignments(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    n_expert: int,
) -> Tuple[List[int], Dict[int, torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """直接从稀疏 top-k 路由结果构建活跃专家和 token 分配。

    这个函数替代了旧的三步流程：
    1. _build_expert_mask: 构造 [n_expert, top_k, n_token] 的 dense one-hot mask；
    2. _collect_active_experts: 对每个专家 torch.where 找活跃 token；
    3. _organize_token_assignments: 再次整理 token 索引和 routing weight。

    当前实现直接处理 selected_experts 的稀疏结果：先把所有 top-k 专家 ID 拉平，
    排序后用 torch.unique_consecutive 得到每个活跃专家对应的连续区间，再用这些
    区间切出 token 索引和 routing weight。这样避免构造 dense mask，也避免对每个
    专家重复 torch.where。

    Args:
        selected_experts: gate 输出的专家 ID，通常形状为 [batch, seq_len, top_k]。
        routing_weights: gate 输出的权重，形状与 selected_experts 对应。
        n_expert: 总专家数；当前保留在签名中以兼容策略接口，此实现不需要遍历它。

    Returns:
        active_experts: 当前层实际被选中的专家 ID 列表，按专家 ID 升序排列。
        token_indices_by_expert: {expert_id: token_positions}，token_positions 是展平后的 token 位置。
        expert_assignments: {expert_id: (token_positions, routing_weight_subset)}，供 executor 构造专家输入。
    """
    # [batch, seq_len, top_k] -> [batch * seq_len * top_k]，保留每个 top-k 选择的专家 ID。
    flat_experts = selected_experts.reshape(-1)
    top_k = selected_experts.shape[-1]
    # routing weight 与 flat_experts 使用相同展平顺序，后续可用排序索引同步重排。
    flat_weights = routing_weights.reshape(-1)

    if flat_experts.numel() == 0:
        return [], {}, {}

    # 按专家 ID 排序，让同一专家的所有 token 选择变成连续区间。
    sorted_experts, order = torch.sort(flat_experts)
    if sorted_experts.numel() == 0:
        return [], {}, {}

    # unique_consecutive 只在排序后正确聚合；counts 给出每个专家连续区间长度。
    unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    # order 是 flat_experts 的位置；除以 top_k 可还原到展平 token 位置。
    token_positions = torch.div(order, top_k, rounding_mode="floor")
    # 使用同一个 order 同步重排 routing weight，保证 token 与权重一一对应。
    routing_weight_values = flat_weights.index_select(0, order).unsqueeze(-1)

    # 后续要在 Python 字典中分段组织结果，只把专家 ID 和 counts 同步到 CPU。
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


class ExpertSchedulingStrategy:
    """专家调度策略基类，定义策略接口"""
    def __init__(self, dev, is_expert_in_gpu):
        self.dev = dev
        self.is_expert_in_gpu = is_expert_in_gpu
        self.stats_recorder: Optional[ExpertSchedulingMonitor] = None

    def set_stats_recorder(self, recorder: Optional[ExpertSchedulingMonitor]):
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
    def __init__(self, dev, is_expert_in_gpu, latency_model: ExpertLatencyModel):
        super().__init__(dev, is_expert_in_gpu)
        self.latency_model = latency_model
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
        """决策：所有活跃专家都在 GPU 上执行"""
        active_experts, token_indices_by_expert, expert_assignments = _collect_expert_assignments(
            selected_experts, routing_weights, n_expert
        )
        if self.stats_recorder is not None:
            self.stats_recorder.record_gpu_only(
                strategy_name=self.__class__.__name__,
                layer=i_layer,
                active_experts=active_experts,
            )
        return [], active_experts, expert_assignments


class FiddlerStrategy(ExpertSchedulingStrategy):
    """CPU-GPU 混合调度策略：通过代价优化决定每个专家在 CPU 还是 GPU 上执行
    
    职责：
    - 构建每个活跃专家的 CPU/GPU 执行代价表
    - 按每个活跃专家独立选择代价更低的 CPU/GPU 执行位置
    - 统计 GPU 缓存命中率
    """
    def __init__(self, dev, is_expert_in_gpu, latency_model: ExpertLatencyModel):
        super().__init__(dev, is_expert_in_gpu)
        self.latency_model = latency_model
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
        for i_expert in active_experts:
            token_count = token_indices_by_expert[i_expert].shape[0]
            cost_cpu = self.latency_model.cpu(token_count)
            cost_gpu = self.latency_model.gpu_compute(token_count) + self.latency_model.transfer(i_layer, i_expert)
            resident = self.is_expert_in_gpu(i_layer, i_expert)
            if resident:
                cost_gpu = 0
                self.cnt_expert_hit += token_count
            self.cnt_expert_all += token_count

            if cost_cpu < cost_gpu:
                cpu_experts.append(i_expert)
            else:
                gpu_experts.append(i_expert)

        if self.stats_recorder is not None:
            self.stats_recorder.record_fiddler(
                strategy_name=self.__class__.__name__,
                layer=i_layer,
                active_experts=active_experts,
                cpu_experts=cpu_experts,
                gpu_experts=gpu_experts,
            )
        
        return cpu_experts, gpu_experts, expert_assignments


class PDScopeScheduler(ExpertSchedulingStrategy):
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

    def __init__(
        self,
        dev,
        is_expert_in_gpu,
        latency_model: ExpertLatencyModel,
        alpha: float = 0.1,
        t_attn: float = 0.6,
        r_hit: float = 0.8,
    ):
        super().__init__(dev, is_expert_in_gpu)
        self.latency_model = latency_model
        self.alpha = alpha
        self.t_attn = t_attn
        self.r_hit = r_hit

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
        """策略决策入口：构造请求并执行 PDScope 调度"""
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

        request = ExpertLayerRequest(
            layer=i_layer,
            phase="prefill" if is_prefill else "decode",
            current=current_demands,
            future=future,
            assignments=build_assignments(raw_assignments),
        )
        schedule = self.schedule(request, placement, self.latency_model)

        if self.stats_recorder is not None:
            self.stats_recorder.record_prefetch_hybrid(
                strategy_name=self.__class__.__name__,
                layer=i_layer,
                active_experts=active_experts,
                schedule=schedule,
            )

        return schedule.cpu_expert_ids, schedule.gpu_expert_ids, schedule.preload_expert_ids, raw_assignments

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


        if cur_below and next_below:
        # 当前层和下一层驻留都不足，回退到 prefill 策略，尝试通过预加载来提升未来层驻留，从而间接提升当前层驻留
            schedule = self.schedule_prefill(request, placement, latency)
            schedule.reason = "decode-fallback-prefill"

            return schedule
        if cur_below and not next_below:
        # 当前层驻留不足但下一层充足，优先补齐当前层 GPU 专家（mode-a），从全局候选中选取分数最高的 n_g_rho 个专家放入 GPU，剩余放 CPU
            need = min(n_g_rho - len(current_resident), len(current_non_resident))
            ondemand = current_non_resident[:need]
            gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
            gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
            cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
            schedule = ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-a")
            return schedule

        if not cur_below and next_below:
        # 当前层驻留充足但下一层不足，预加载下一层专家（mode-b）
            need_next = max(0, n_g_rho - next_resident_count)
            preload = future_non_resident[:need_next]
            gpu = current_resident # GPU 驻留专家权重已在 GPU，全部走 GPU 计算；CPU 路径只放非驻留专家
            gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
            cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
            schedule = ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="decode-mode-b")
            return schedule

        gpu = current_resident # GPU 驻留专家权重已在 GPU，全部走 GPU 计算；CPU 路径只放非驻留专家
        gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
        cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
        schedule = ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-c")
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
        # return future[:]


class PregatedStrategy(ExpertSchedulingStrategy):
    """Pregated 调度策略：仅使用门控作为预测器来决定预加载

    与 PDScopeScheduler 的区别：
    - 不做 prefill/decode 阶段拆分，也不做 CPU/GPU 代价最优选择；
    - 所有活跃专家都在 GPU 上执行（gpu 列表 = 全部活跃专家），cpu 列表恒为空；
    - 预加载（preload）来源只有门控预测出的下一层专家。

    输出：gpu 列表（全部活跃专家）与 preload 列表（门控预测的非驻留专家）。

    参数：
    - t_attn: 注意力计算时间窗口（ms），用于估计可用来做 I/O 预加载的气泡。
    - max_preload: 单层最多预加载的专家数；None 表示只受 I/O 气泡与 placeholder 容量限制。
    """

    def __init__(
        self,
        dev,
        is_expert_in_gpu,
        latency_model: ExpertLatencyModel,
        t_attn: float = 0.6,
        max_preload: Optional[int] = None,
    ):
        super().__init__(dev, is_expert_in_gpu)
        self.latency_model = latency_model
        self.t_attn = t_attn
        self.max_preload = max_preload

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
        """决策：活跃专家全部上 GPU，门控预测的下一层专家用于预加载"""
        active_experts, token_indices_by_expert, raw_assignments = _collect_expert_assignments(
            selected_experts, routing_weights, n_expert
        )

        future = future_demands
        if future is None:
            future = build_future_demands(i_layer + 1, predicted_next_experts, predicted_next_weights)
        future = unique_demands(future)

        if placement is None:
            gpu_resident = set()
            for demand in future:
                if self.is_expert_in_gpu(demand.key.layer, demand.key.expert_id):
                    gpu_resident.add((demand.key.layer, demand.key.expert_id))
            placement = PlacementSnapshot(gpu_resident=gpu_resident)

        preload = self._select_preload(active_experts, future, placement)

        gpu = [
            ExpertDemand(
                key=ExpertKey(layer=i_layer, expert_id=eid),
                token_count=int(token_indices_by_expert[eid].shape[0]),
                score=float(token_indices_by_expert[eid].shape[0]),
                source="current",
            )
            for eid in active_experts
        ]
        schedule = ExpertSchedule(gpu=gpu, preload=preload, reason="pregated")

        if self.stats_recorder is not None:
            self.stats_recorder.record_prefetch_hybrid(
                strategy_name=self.__class__.__name__,
                layer=i_layer,
                active_experts=active_experts,
                schedule=schedule,
            )

        return schedule.cpu_expert_ids, schedule.gpu_expert_ids, schedule.preload_expert_ids, raw_assignments

    def _select_preload(
        self,
        active_experts: List[int],
        future: List[ExpertDemand],
        placement: PlacementSnapshot,
    ) -> List[ExpertDemand]:
        """从门控预测的下一层专家中选出可预加载的子集

        选择规则：
        1. 仅保留不在 GPU 上、且未在加载队列中的预测专家；
        2. 按 score 降序排序（分数相同时 token 数多的优先）；
        3. 容量上限取以下三者最小值：
           - I/O 气泡窗口内可传输的专家数（窗口 = 注意力时间 + 当前层 GPU 计算时间）；
           - 空闲 placeholder 数；
           - max_preload（若设置）。
        """
        candidates = [
            d for d in future
            if not placement.is_on_gpu(d.key.layer, d.key.expert_id)
            and (d.key.layer, d.key.expert_id) not in placement.loading
        ]
        if not candidates:
            return []
        candidates.sort(key=lambda d: (d.score, d.token_count), reverse=True)

        # 当前层 GPU 计算可与下一层专家 I/O 传输重叠，构成预加载气泡窗口。
        window = self.t_attn + len(active_experts) * self.latency_model.gpu_compute(1)
        io_capacity = math.floor(window / max(self.latency_model.t_io, 1e-9))
        capacity = max(0, io_capacity)
        if placement.free_placeholders > 0:
            capacity = min(capacity, placement.free_placeholders)
        if self.max_preload is not None:
            capacity = min(capacity, self.max_preload)
        if capacity <= 0:
            return []
        return candidates[:capacity]
