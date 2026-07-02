import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

import torch

from model.expert_types import ExpertLayerContext, ExpertSchedule


class ExpertExecutionManager:
    def __init__(self, device, placeholder_manager, model, is_expert_in_gpu: Callable[[int, int], bool], profile_timing: bool = False, cpu_experts=None):
        self.device = device
        self.placeholder_manager = placeholder_manager
        self.model = model
        self.is_expert_in_gpu = is_expert_in_gpu
        self.profile_timing = profile_timing
        self.cpu_experts = cpu_experts if cpu_experts is not None else {}
        self.compute_stream = torch.cuda.Stream(device=self.device) if self._is_cuda_device() else None
        self.preload_stream = torch.cuda.Stream(device=self.device) if self._is_cuda_device() else None
        self._pool = ThreadPoolExecutor(max_workers=3)
        self._timing_lock = threading.Lock()
        self._timing_stats = defaultdict(lambda: defaultdict(float))
        self._preload_event_lock = threading.Lock()
        self._preload_events = {}
        self.reset_runtime_counters()

    def execute(self, schedule: ExpertSchedule, context: ExpertLayerContext) -> torch.Tensor:
        gpu_result = None
        cpu_result = None
        wall_tick = time.perf_counter()
        caller_stream = torch.cuda.current_stream(self.device) if self._is_cuda_device() else None
        copy_stream = self.compute_stream if self._is_cuda_device() else None
        pool = self._ensure_pool()
        futures = {}
        if schedule.gpu_expert_ids:
            futures[pool.submit(self._timed_call, context.layer, "gpu", self.execute_gpu_experts, context, schedule.gpu_expert_ids, caller_stream)] = "gpu"
        if schedule.cpu_expert_ids:
            futures[pool.submit(self._timed_call, context.layer, "cpu", self.execute_cpu_experts, context, schedule.cpu_expert_ids)] = "cpu"
        if schedule.preload:
            futures[pool.submit(self._timed_call, context.layer, "preload", self.preload_experts, schedule.preload)] = "preload"

        results = {}
        for future in list(futures.keys()):
            results[futures[future]] = future.result()
        gpu_result = results.get("gpu")
        cpu_result = results.get("cpu")

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

    def reset_timing_stats(self) -> None:
        with self._timing_lock:
            self._timing_stats.clear()
        with self._preload_event_lock:
            self._preload_events.clear()

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

    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=3)
        return self._pool

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def timing_summary(self) -> List[Dict[str, float]]:
        with self._timing_lock:
            items = [(layer, dict(stats)) for layer, stats in self._timing_stats.items()]

        summary = []
        for layer, stats in sorted(items):
            gpu = stats.get("gpu", 0.0)
            cpu = stats.get("cpu", 0.0)
            preload = stats.get("preload", 0.0)
            wall = stats.get("wall", 0.0)
            component_total = gpu + cpu + preload
            summary.append(
                {
                    "layer": layer,
                    "calls": stats.get("calls", 0.0),
                    "wall": wall,
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
                f"wall={row['wall']:.6f}s gpu={row['gpu']:.6f}s "
                f"cpu={row['cpu']:.6f}s preload={row['preload']:.6f}s "
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

    def _timed_call(self, layer: int, name: str, fn, *args):
        # if layer < 5:  # 仅打印前几层的调用信息以避免日志过多
        if self.profile_timing:
            print(
                f"[timed-call] layer={layer} name={name} "
                f"fn={getattr(fn, '__name__', repr(fn))} "
                f"expert_ids={self._format_timed_call_expert_ids(name, args)}"
            )

        if name == "gpu" and self._is_cuda_device():
            stream = self.compute_stream
            caller_stream = args[-1] if args else None
            fn_args = args[:-1]
            if caller_stream is not None:
                stream.wait_stream(caller_stream)
            start = torch.cuda.Event(enable_timing=True) if self.profile_timing else None
            end = torch.cuda.Event(enable_timing=True) if self.profile_timing else None
            with torch.cuda.stream(stream):
                if start is not None:
                    start.record(stream)
                result = fn(*fn_args)
                if end is not None:
                    end.record(stream)
            if end is not None:
                end.synchronize()
                self._add_timing(layer, name, start.elapsed_time(end) / 1000.0)
            return result

        if name == "preload" and self._is_cuda_device():
            tick = time.perf_counter()
            result = fn(*args)
            if self.profile_timing:
                self._add_timing(layer, name, time.perf_counter() - tick)
            return result

        tick = time.perf_counter()
        result = fn(*args)
        if self.profile_timing:
            self._add_timing(layer, name, time.perf_counter() - tick)
        return result

    def _add_timing(self, layer: int, name: str, elapsed: float) -> None:
        with self._timing_lock:
            self._timing_stats[layer][name] += elapsed
            if name != "calls" :  # 仅打印前几层的时间统计以避免日志过多
                print(f"[Timing] layer={layer} {name} += {elapsed:.6f}s (total={self._timing_stats[layer][name]:.6f}s)")

    def _sync_cuda(self) -> None:
        if self._is_cuda_device():
            torch.cuda.synchronize(self.device)

    def _is_cuda_device(self) -> bool:
        return torch.cuda.is_available() and torch.device(self.device).type == "cuda"

    def _format_timed_call_expert_ids(self, name: str, args) -> str:
        if name in ("gpu", "cpu") and len(args) >= 2:
            return repr(args[1])
        if name == "preload" and args:
            return repr([(demand.key.layer, demand.key.expert_id) for demand in args[0]])
        return "[]"

    def execute_gpu_experts(self, context: ExpertLayerContext, expert_ids: List[int]) -> torch.Tensor:
        result = torch.zeros_like(context.inps_flat, device=self.device)
        for expert_id in expert_ids:
            assignment = context.assignments[expert_id]
            token_indices = assignment.token_indices.to(self.device).long().contiguous()
            n_tokens = token_indices.shape[0]
            current_state = context.inps_flat.index_select(0, token_indices)
            placeholder = self.placeholder_manager.get_placeholder_for_expert(context.layer, expert_id)
            if self.placeholder_manager.is_static_gpu_resident(context.layer, expert_id):
                current_state = context.experts[expert_id](current_state)
            elif placeholder is not None:
                was_preloaded = self._wait_for_preload(context.layer, expert_id)
                if was_preloaded:
                    self.actual_preload_hit_count += 1
                self.placeholder_manager.protect_expert(context.layer, expert_id)
                try:
                    current_state = placeholder(current_state)
                finally:
                    self.placeholder_manager.unprotect_expert(context.layer, expert_id)
            else:
                placeholder = self.placeholder_manager.acquire_placeholder(context.layer, expert_id)
                if placeholder is None:
                    raise RuntimeError(f"No placeholder available for expert ({context.layer}, {expert_id})")
                self._discard_preload_event(context.layer, expert_id)
                self.placeholder_manager.load_weights(placeholder, self.get_cpu_expert(context.layer, expert_id))
                self.placeholder_manager.protect_expert(context.layer, expert_id)
                try:
                    current_state = placeholder(current_state)
                finally:
                    self.placeholder_manager.unprotect_expert(context.layer, expert_id)

            current_state = current_state * assignment.routing_weights
            result.index_add_(0, token_indices, current_state.to(result.dtype))
        self.placeholder_manager.release_by_layer(context.layer)
        return result

    def execute_cpu_experts(self, context: ExpertLayerContext, expert_ids: List[int]) -> torch.Tensor:
        result = torch.zeros_like(context.inps_flat, device="cpu")
        for expert_id in expert_ids:
            assignment = context.assignments[expert_id]
            token_indices_gpu = assignment.token_indices.to(
                device=context.inps_flat.device,
                dtype=torch.long,
            ).contiguous()
            token_indices_cpu = token_indices_gpu.to("cpu")
            n_tokens = token_indices_gpu.shape[0]
             
            current_state = context.inps_flat.index_select(0, token_indices_gpu)
            current_state = self.get_cpu_expert(context.layer, expert_id)(current_state.to("cpu"))
            current_state = current_state * assignment.routing_weights.to("cpu")
            result.index_add_(0, token_indices_cpu, current_state.to(result.dtype))
        return result

    def get_cpu_expert(self, layer: int, expert_id: int):
        return self.cpu_experts.get((layer, expert_id), self.model.layers[layer].mlp.experts[expert_id])

    def preload_experts(self, demands) -> None:
        if self.preload_stream is not None:
            with torch.cuda.stream(self.preload_stream):
                self._preload_experts_impl(demands)
            return
        self._preload_experts_impl(demands)

    def _preload_experts_impl(self, demands) -> None:
        by_layer: Dict[int, List[int]] = {}
        for demand in demands:
            by_layer.setdefault(demand.key.layer, []).append(demand.key.expert_id)

        for layer, expert_ids in by_layer.items():
            if layer >= len(self.model.layers):
                continue
            # tick = time.time()
            loaded = 0
            for expert_id in expert_ids:
                self.actual_preload_request_count += 1
                if self.is_expert_in_gpu(layer, expert_id) or self.placeholder_manager.is_on_gpu(layer, expert_id):
                    self.actual_preload_skip_count += 1
                    self.actual_preload_skip_already_gpu_count += 1
                    continue
                if self.placeholder_manager.is_loading(layer, expert_id):
                    self.actual_preload_skip_count += 1
                    self.actual_preload_skip_loading_count += 1
                    continue
                self.placeholder_manager.mark_loading(layer, expert_id)
                try:
                    placeholder = self.placeholder_manager.acquire_free_placeholder(layer, expert_id)
                    if placeholder is None:
                        self.actual_preload_skip_count += 1
                        self.actual_preload_skip_no_slot_count += 1
                        continue
                    self.placeholder_manager.load_weights(placeholder, self.get_cpu_expert(layer, expert_id))
                    self._record_preload_event(layer, expert_id)
                    loaded += 1
                    self.actual_preload_success_count += 1
                finally:
                    self.placeholder_manager.unmark_loading(layer, expert_id)
            # elapsed = time.time() - tick
            # # replaced by func _add_timing to avoid excessive logging 
            # if loaded > 0:
            #     print(f"  Preload layer {layer}: {loaded} experts loaded in {elapsed*1000:.2f}ms")

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

    def _discard_preload_event(self, layer: int, expert_id: int) -> None:
        with self._preload_event_lock:
            self._preload_events.pop((layer, expert_id), None)
