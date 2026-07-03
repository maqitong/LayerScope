import time
import copy
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from model.expert_types import ExpertLayerContext, ExpertSchedule


@dataclass
class _GpuEntry:
    """单个 GPU 专家的执行计划项，由 prepare 阶段在 host 上一次性生成。"""

    expert_id: int
    # static: 常驻 GPU 专家模块；preloaded/ondemand: 占位符模块
    kind: str  # "static" | "preloaded" | "ondemand"
    module: object
    token_indices: torch.Tensor
    routing_weights: torch.Tensor
    # ondemand 需要在 compute 阶段做一次 H2D 权重拷贝
    needs_load: bool = False
    cpu_expert: Optional[object] = None


class ExpertExecutionManager:
    """专家执行器：单调用线程编排 + compute/preload 双流 + 显式阶段。

    每层 execute 按以下顺序执行：
      1. prepare    (纯 host)  解析每个 GPU 专家落点、获取/保护占位符，生成执行计划
      2. barrier    (一次同步) compute_stream 等待 caller_stream(输入就绪) 与上一轮 preload 事件
      3. compute_gpu(纯 GPU)   在 compute_stream 上只跑 kernel/拷贝/index_add，无锁无 host 决策
      4. compute_cpu(CPU 线程) CPU 专家，与 compute_gpu 并行
      5. preload    (纯传输)   在 preload_stream 上对下一层做 H2D 拷贝并记录事件

    跨流正确性靠两条事件依赖保证：
      - preload 事件  -> 下一次 compute 读  (barrier 阶段 wait_event)
      - compute 完成事件 -> 下一次 preload 写 (preload 阶段 wait_event)
    """

    def __init__(self, device, placeholder_manager, model, is_expert_in_gpu: Callable[[int, int], bool], profile_timing: bool = False, cpu_experts=None):
        self.device = device
        self.placeholder_manager = placeholder_manager
        self.model = model
        self.is_expert_in_gpu = is_expert_in_gpu
        self.profile_timing = profile_timing
        self.cpu_experts = cpu_experts if cpu_experts is not None else {}
        self.compute_stream = torch.cuda.Stream(device=self.device) if self._is_cuda_device() else None
        self.preload_stream = torch.cuda.Stream(device=self.device) if self._is_cuda_device() else None
        self._pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=2)
        self._timing_lock = threading.Lock()
        self._timing_stats = defaultdict(lambda: defaultdict(float))
        self._preload_event_lock = threading.Lock()
        self._preload_events: Dict[tuple, torch.cuda.Event] = {}
        # 最近一次 compute_stream 完成事件，用于门控 preload 写入，避免覆写仍在被读取的权重
        self._last_compute_event: Optional[torch.cuda.Event] = None
        # ondemand 专属 scratch：固定 GPU 缓冲，逐专家覆盖使用，不参与 placeholder 管理
        self._ondemand_scratch = None
        self.reset_runtime_counters()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def execute(self, schedule: ExpertSchedule, context: ExpertLayerContext) -> torch.Tensor:
        wall_tick = time.perf_counter()
        is_cuda = self._is_cuda_device()
        caller_stream = torch.cuda.current_stream(self.device) if is_cuda else None

        # Phase 1: prepare (纯 host)
        prepare_tick = time.perf_counter()
        gpu_plan = self._prepare_gpu(schedule, context)
        if self.profile_timing:
            self._add_timing(context.layer, "prepare", time.perf_counter() - prepare_tick)

        # Phase 2: barrier (一次同步)
        self._barrier_into_compute(gpu_plan, context, caller_stream)

        # Phase 3/4: CPU 专家提交到工作线程（与 GPU 计算并行）
        cpu_future = None
        if schedule.cpu_expert_ids:
            pool = self._ensure_pool()
            cpu_future = pool.submit(self._run_cpu_experts, context, schedule.cpu_expert_ids, caller_stream)

        # Phase 3: compute_gpu (纯 GPU，在 compute_stream 上)
        gpu_result = self._compute_gpu(gpu_plan, context, caller_stream)

        # Phase 5: preload (纯传输，在 preload_stream 上)
        self._dispatch_preload(schedule.preload, context.layer)

        # Phase 4 收尾：等待 CPU 专家完成
        cpu_result = cpu_future.result() if cpu_future is not None else None

        # 合并：确保 compute_stream 结果就绪后再在 caller_stream 上累加
        if caller_stream is not None and gpu_result is not None:
            caller_stream.wait_stream(self.compute_stream)
        result = torch.zeros_like(context.inps_flat, device=self.device)
        if gpu_result is not None:
            result += gpu_result
        if cpu_result is not None:
            result += cpu_result.to(self.device)

        if self.profile_timing:
            self._sync_cuda()
            self._add_timing(context.layer, "wall", time.perf_counter() - wall_tick)
            self._add_timing(context.layer, "calls", 1.0)
        return result

    # ------------------------------------------------------------------
    # 计数与重置
    # ------------------------------------------------------------------
    def reset_timing_stats(self) -> None:
        with self._timing_lock:
            self._timing_stats.clear()
        with self._preload_event_lock:
            self._preload_events.clear()
        self._last_compute_event = None

    def reset_runtime_counters(self) -> None:
        self.planned_cpu_count = 0
        self.planned_cpu_tokens = 0
        self.planned_gpu_static_count = 0
        self.planned_gpu_static_tokens = 0
        self.planned_gpu_placeholder_count = 0
        self.planned_gpu_placeholder_tokens = 0
        self.planned_gpu_ondemand_count = 0
        self.planned_gpu_ondemand_tokens = 0
        self.planned_preload_count = 0
        self.actual_preload_request_count = 0
        self.actual_preload_success_count = 0
        self.actual_preload_skip_count = 0
        self.actual_preload_skip_already_gpu_count = 0
        self.actual_preload_skip_loading_count = 0
        self.actual_preload_skip_no_slot_count = 0
        self.actual_preload_hit_count = 0

    def record_planned_execution_stats(self, schedule: ExpertSchedule, placement) -> None:
        self.planned_preload_count += len(schedule.preload)

        for demand in schedule.cpu:
            self.planned_cpu_count += 1
            self.planned_cpu_tokens += demand.token_count

        for demand in schedule.gpu:
            key = (demand.key.layer, demand.key.expert_id)
            if key in placement.gpu_resident:
                self.planned_gpu_static_count += 1
                self.planned_gpu_static_tokens += demand.token_count
            elif key in placement.placeholder_resident:
                self.planned_gpu_placeholder_count += 1
                self.planned_gpu_placeholder_tokens += demand.token_count
            else:
                self.planned_gpu_ondemand_count += 1
                self.planned_gpu_ondemand_tokens += demand.token_count

    # ------------------------------------------------------------------
    # 线程池
    # ------------------------------------------------------------------
    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2)
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ------------------------------------------------------------------
    # 计时汇总
    # ------------------------------------------------------------------
    def timing_summary(self) -> List[Dict[str, float]]:
        with self._timing_lock:
            items = [(layer, dict(stats)) for layer, stats in self._timing_stats.items()]

        summary = []
        for layer, stats in sorted(items):
            gpu = stats.get("gpu", 0.0)
            cpu = stats.get("cpu", 0.0)
            preload = stats.get("preload", 0.0)
            prepare = stats.get("prepare", 0.0)
            wall = stats.get("wall", 0.0)
            component_total = gpu + cpu + preload
            summary.append(
                {
                    "layer": layer,
                    "calls": stats.get("calls", 0.0),
                    "wall": wall,
                    "prepare": prepare,
                    "gpu": gpu,
                    "cpu": cpu,
                    "preload": preload,
                    "gpu_component_share": gpu / component_total if component_total > 0 else 0.0,
                    "cpu_component_share": cpu / component_total if component_total > 0 else 0.0,
                    "preload_component_share": preload / component_total if component_total > 0 else 0.0,
                    "gpu_wall_share": gpu / wall if wall > 0 else 0.0,
                    "cpu_wall_share": cpu / wall if wall > 0 else 0.0,
                    "preload_wall_share": preload / wall if wall > 0 else 0.0,
                }
            )
        return summary

    def format_timing_summary(self) -> List[str]:
        lines = []
        for row in self.timing_summary():
            lines.append(
                "[expert-timing] "
                f"layer={int(row['layer'])} calls={int(row['calls'])} "
                f"wall={row['wall']:.6f}s prepare={row['prepare']:.6f}s "
                f"gpu={row['gpu']:.6f}s cpu={row['cpu']:.6f}s preload={row['preload']:.6f}s "
                "component_share(gpu/cpu/preload)="
                f"{row['gpu_component_share'] * 100:.1f}%/"
                f"{row['cpu_component_share'] * 100:.1f}%/"
                f"{row['preload_component_share'] * 100:.1f}% "
                "wall_share(gpu/cpu/preload)="
                f"{row['gpu_wall_share'] * 100:.1f}%/"
                f"{row['cpu_wall_share'] * 100:.1f}%/"
                f"{row['preload_wall_share'] * 100:.1f}%"
            )
        return lines

    def _add_timing(self, layer: int, name: str, elapsed: float) -> None:
        with self._timing_lock:
            self._timing_stats[layer][name] += elapsed

    def _sync_cuda(self) -> None:
        if self._is_cuda_device():
            torch.cuda.synchronize(self.device)

    def _is_cuda_device(self) -> bool:
        return torch.cuda.is_available() and torch.device(self.device).type == "cuda"

    # ------------------------------------------------------------------
    # Phase 1: prepare (纯 host)
    # ------------------------------------------------------------------
    def _prepare_gpu(self, schedule: ExpertSchedule, context: ExpertLayerContext) -> List[_GpuEntry]:
        plan: List[_GpuEntry] = []
        for expert_id in schedule.gpu_expert_ids:
            assignment = context.assignments[expert_id]
            token_indices = assignment.token_indices.to(self.device).long().contiguous()
            routing_weights = assignment.routing_weights.to(self.device)

            if self.placeholder_manager.is_static_gpu_resident(context.layer, expert_id):
                plan.append(_GpuEntry(expert_id, "static", context.experts[expert_id], token_indices, routing_weights))
                continue

            placeholder = self.placeholder_manager.get_placeholder_for_expert(context.layer, expert_id)
            if placeholder is not None:
                # 上一轮 preload 装载好的占位符；在 barrier 阶段等待其 preload 事件
                plan.append(_GpuEntry(expert_id, "preloaded", placeholder, token_indices, routing_weights))
                continue

            # ondemand：当前层既非常驻也未被预加载；不获取占位符，compute 阶段用专属 scratch
            # 逐专家加载权重并计算（不参与 placeholder 管理）
            plan.append(
                _GpuEntry(
                    expert_id,
                    "ondemand",
                    None,
                    token_indices,
                    routing_weights,
                    needs_load=True,
                    cpu_expert=self.get_cpu_expert(context.layer, expert_id),
                )
            )
        return plan

    # ------------------------------------------------------------------
    # Phase 2: barrier (一次同步)
    # ------------------------------------------------------------------
    def _barrier_into_compute(self, gpu_plan: List[_GpuEntry], context: ExpertLayerContext, caller_stream) -> None:
        if self.compute_stream is None:
            return
        # 输入就绪：compute_stream 等待 caller_stream 把 inps_flat 写完
        if caller_stream is not None:
            self.compute_stream.wait_stream(caller_stream)
        # 预加载就绪：排空本层全部 preload 事件（不仅限于本次用到的），确保本层所有 preload
        # 拷贝完成后才进入 compute；这样 compute 内 ondemand 淘汰任意本层 preload 槽位都是安全的
        preloaded_keys = {(context.layer, e.expert_id) for e in gpu_plan if e.kind == "preloaded"}
        self.actual_preload_hit_count += self._drain_layer_preload_events(context.layer, preloaded_keys)

    def _drain_layer_preload_events(self, layer: int, preloaded_keys: set) -> int:
        if self.compute_stream is None:
            return 0
        with self._preload_event_lock:
            items = [(k, v) for k, v in self._preload_events.items() if k[0] == layer]
            for k, _ in items:
                del self._preload_events[k]
        hits = 0
        for key, event in items:
            self.compute_stream.wait_event(event)
            if key in preloaded_keys:
                hits += 1
        return hits

    # ------------------------------------------------------------------
    # Phase 3: compute_gpu (纯 GPU，在 compute_stream 上)
    # ------------------------------------------------------------------
    def _compute_gpu(self, gpu_plan: List[_GpuEntry], context: ExpertLayerContext, caller_stream) -> Optional[torch.Tensor]:
        if not gpu_plan:
            return None

        is_cuda = self._is_cuda_device()
        stream = self.compute_stream
        start = torch.cuda.Event(enable_timing=True) if (is_cuda and self.profile_timing) else None
        end = torch.cuda.Event(enable_timing=True) if (is_cuda and self.profile_timing) else None
        compute_done = torch.cuda.Event() if is_cuda else None

        with torch.cuda.stream(stream) if is_cuda else nullcontext():
            if start is not None:
                start.record(stream)
            result = torch.zeros_like(context.inps_flat, device=self.device)
            # 热路径：static + preloaded，纯 GPU 读取（无锁、无 host 决策）
            for entry in gpu_plan:
                if entry.kind == "ondemand":
                    continue
                current_state = context.inps_flat.index_select(0, entry.token_indices)
                current_state = entry.module(current_state)
                current_state = current_state * entry.routing_weights
                result.index_add_(0, entry.token_indices, current_state.to(result.dtype))
            # ondemand 路径：用专属 scratch 逐专家"拷贝权重→前向"。
            # 不经过 placeholder_manager（无 acquire/淘汰/释放/驻留登记），与 preload 槽位互不竞争；
            # 逐专家顺序覆盖同一 scratch，compute_stream 同流有序，安全。
            has_ondemand = False
            for entry in gpu_plan:
                if entry.kind != "ondemand":
                    continue
                if not has_ondemand:
                    scratch = self._ensure_ondemand_scratch(entry.cpu_expert)
                    has_ondemand = True
                self._copy_into(scratch, entry.cpu_expert)
                current_state = context.inps_flat.index_select(0, entry.token_indices)
                current_state = scratch(current_state)
                current_state = current_state * entry.routing_weights
                result.index_add_(0, entry.token_indices, current_state.to(result.dtype))
            if end is not None:
                end.record(stream)
            if compute_done is not None:
                compute_done.record(stream)

        # 记录 compute 完成事件，门控下一轮 preload 写入，避免覆写仍在被读取的权重
        if compute_done is not None:
            self._last_compute_event = compute_done

        if end is not None:
            end.synchronize()
            self._add_timing(context.layer, "gpu", start.elapsed_time(end) / 1000.0)

        # host 侧释放本层占位符（仅更新簿记；权重要等被门控的 preload 拷贝才会被覆写）
        self.placeholder_manager.release_by_layer(context.layer)
        return result

    # ------------------------------------------------------------------
    # Phase 4: compute_cpu (CPU 工作线程)
    # ------------------------------------------------------------------
    def _run_cpu_experts(self, context: ExpertLayerContext, expert_ids: List[int], caller_stream) -> torch.Tensor:
        tick = time.perf_counter()
        result = self.execute_cpu_experts(context, expert_ids, caller_stream)
        if self.profile_timing:
            self._add_timing(context.layer, "cpu", time.perf_counter() - tick)
        return result

    def execute_cpu_experts(self, context: ExpertLayerContext, expert_ids: List[int], caller_stream=None) -> torch.Tensor:
        result = torch.zeros_like(context.inps_flat, device="cpu")
        if caller_stream is not None and self._is_cuda_device():
            # 显式等待输入就绪，避免依赖 legacy default stream 的隐式同步
            torch.cuda.current_stream(self.device).wait_stream(caller_stream)
        for expert_id in expert_ids:
            assignment = context.assignments[expert_id]
            token_indices_gpu = assignment.token_indices.to(
                device=context.inps_flat.device,
                dtype=torch.long,
            ).contiguous()
            token_indices_cpu = token_indices_gpu.to("cpu")

            current_state = context.inps_flat.index_select(0, token_indices_gpu)
            current_state = self.get_cpu_expert(context.layer, expert_id)(current_state.to("cpu"))
            current_state = current_state * assignment.routing_weights.to("cpu")
            result.index_add_(0, token_indices_cpu, current_state.to(result.dtype))
        return result

    def get_cpu_expert(self, layer: int, expert_id: int):
        return self.cpu_experts.get((layer, expert_id), self.model.layers[layer].mlp.experts[expert_id])

    def _ensure_ondemand_scratch(self, template):
        """惰性分配 ondemand 专属 scratch：深拷贝一个专家模块到 device，全程复用。"""
        if self._ondemand_scratch is None:
            self._ondemand_scratch = copy.deepcopy(template).to(self.device)
        return self._ondemand_scratch

    @staticmethod
    def _copy_into(dst, src):
        """把 src 的参数/buffer 拷贝到 dst（不加锁，dst 由调用方独占）。"""
        with torch.no_grad():
            for name, dst_param in dst.named_parameters():
                dst_param.copy_(src.get_parameter(name))
            for name, dst_buffer in dst.named_buffers():
                dst_buffer.copy_(src.get_buffer(name))

    # ------------------------------------------------------------------
    # Phase 5: preload (纯传输，在 preload_stream 上)
    # ------------------------------------------------------------------
    def _dispatch_preload(self, demands, issuing_layer: int) -> None:
        if not demands:
            return
        is_cuda = self._is_cuda_device()

        # host：解析槽位
        work: List[tuple] = []
        for demand in demands:
            layer = demand.key.layer
            expert_id = demand.key.expert_id
            self.actual_preload_request_count += 1
            if layer >= len(self.model.layers):
                continue
            if self.is_expert_in_gpu(layer, expert_id) or self.placeholder_manager.is_on_gpu(layer, expert_id):
                self.actual_preload_skip_count += 1
                self.actual_preload_skip_already_gpu_count += 1
                continue
            if self.placeholder_manager.is_loading(layer, expert_id):
                self.actual_preload_skip_count += 1
                self.actual_preload_skip_loading_count += 1
                continue
            self.placeholder_manager.mark_loading(layer, expert_id)
            placeholder = self.placeholder_manager.acquire_free_placeholder(layer, expert_id)
            if placeholder is None:
                self.actual_preload_skip_count += 1
                self.actual_preload_skip_no_slot_count += 1
                self.placeholder_manager.unmark_loading(layer, expert_id)
                continue
            # 不在此处 protect：层内 ondemand 需要靠淘汰复用占位符（占位符总数远小于一层可能
            # 路由到的专家数）。preload 槽位的安全性改由 barrier 排空本层 preload 事件 + compute
            # 内"preloaded 先读、ondemand 后写"的同流顺序来保证。
            work.append((layer, expert_id, placeholder))

        if not work:
            return

        stream = self.preload_stream
        start = torch.cuda.Event(enable_timing=True) if (is_cuda and self.profile_timing) else None
        end = torch.cuda.Event(enable_timing=True) if (is_cuda and self.profile_timing) else None

        with torch.cuda.stream(stream) if is_cuda else nullcontext():
            # hazard 门控：等上一轮 compute 读完后才允许覆写占位符权重
            if is_cuda and self._last_compute_event is not None:
                stream.wait_event(self._last_compute_event)
            if start is not None:
                start.record(stream)
            for layer, expert_id, placeholder in work:
                self.placeholder_manager.load_weights(placeholder, self.get_cpu_expert(layer, expert_id))
                self._record_preload_event(layer, expert_id)
                self.actual_preload_success_count += 1
            if end is not None:
                end.record(stream)

        if end is not None:
            end.synchronize()
            self._add_timing(issuing_layer, "preload", start.elapsed_time(end) / 1000.0)

        for layer, expert_id, _ in work:
            self.placeholder_manager.unmark_loading(layer, expert_id)

    # ------------------------------------------------------------------
    # preload 事件
    # ------------------------------------------------------------------
    def _record_preload_event(self, layer: int, expert_id: int) -> None:
        if self.preload_stream is None:
            return
        event = torch.cuda.Event()
        event.record(self.preload_stream)
        with self._preload_event_lock:
            self._preload_events[(layer, expert_id)] = event

    def _wait_for_preload(self, layer: int, expert_id: int) -> bool:
        if self.compute_stream is None:
            return False
        with self._preload_event_lock:
            event = self._preload_events.pop((layer, expert_id), None)
        if event is not None:
            self.compute_stream.wait_event(event)
            return True
        return False
