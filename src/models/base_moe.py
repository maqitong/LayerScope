import copy
import json
import os
import time

import torch
import torch.nn.functional as F
import transformers

from strategies.caching.eviction_strategy import FIFOEvictionStrategy
from runtime.execution.expert_executor import ExpertExecutionManager
from runtime.monitoring.expert_latency import ExpertLatencyModel
from strategies.prediction.expert_predictor import GatePredictor
from strategies.scheduling.expert_scheduling import (
    FiddlerStrategy,
    GPUOnlyStrategy,
    PrefetchHybridStrategy,
    _build_latency_lookup,
)
from runtime.execution.placeholder_manager import ExpertPlaceholderManager
from runtime.monitoring.runtime_monitor import RuntimeMonitor


class BaseMoERuntime:
    """Shared MoE inference runtime for model-specific architecture adapters."""

    adapter_cls = None

    def __init__(self, args):
        self.dtype = torch.bfloat16
        self.dev = torch.device("cuda:0")
        self.adapter = self.create_adapter(args)
        self._bind_adapter_fields()
        self._init_runtime(args)

    def create_adapter(self, args):
        if self.adapter_cls is None:
            raise NotImplementedError("adapter_cls must be set by subclasses")
        return self.adapter_cls(args, dtype=self.dtype, device=self.dev)

    def _bind_adapter_fields(self):
        self.model = self.adapter.model
        self.lm_head = self.adapter.lm_head
        self.tokenizer = self.adapter.tokenizer
        self.config = self.adapter.config
        self.rotary_emb = getattr(self.adapter, "rotary_emb", None)

    def _init_runtime(self, args):
        self.batch_size = args.batch_size
        self.cpu_offload = args.cpu_offload
        self.beam_width = args.beam_width
        self.profile_expert_executor = getattr(args, "profile_expert_executor", False)
        self.sync_timing = getattr(args, "sync_timing", False)
        self.record_hot_experts = getattr(args, "record_hot_experts", False)
        self.n_layer = self.adapter.num_layers
        self.n_expert = self.adapter.num_routed_experts
        self.n_shared_experts = self.adapter.num_shared_experts

        self.placeholder_manager = ExpertPlaceholderManager(
            template_expert=self.adapter.get_template_expert(),
            device=self.dev,
            num_placeholders=2 * self.adapter.num_experts_per_tok,
            eviction_strategy=FIFOEvictionStrategy(),
        )
        self.expert_placeholder = self.placeholder_manager._placeholders[0]

        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0

        self.latency_cpu = 0.142
        self.latency_copy = 0.4
        self.latency_gpu = 0.093
        self.latency_cpu_table = {1: 0.142}
        self.latency_gpu_table = {1: 0.093}
        self._load_latency_tables(args.model)

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.expert_strategy = self._create_expert_strategy(args)
        self.gpu_only_strategy = GPUOnlyStrategy(self.dev, self.is_expert_in_gpu)
        print(f"Initialized expert scheduling strategy: {self.expert_strategy.__class__.__name__}")

        self.monitor = RuntimeMonitor(
            runtime_meta=self._build_runtime_meta(args),
            schedule_log=getattr(args, "expert_schedule_log", None),
            record_schedule=getattr(args, "record_expert_schedule", False),
        )
        self.monitor.attach_schedule_recorder(self.expert_strategy)
        self.schedule_stats_recorder = self.monitor.schedule_stats_recorder
        self.hot_expert_counts = self.monitor.hot_expert_counts
        if self.schedule_stats_recorder is not None:
            schedule_log = getattr(args, "expert_schedule_log", None)
            if schedule_log:
                print(f"[schedule-stats] Recording scheduling decisions to: {schedule_log}")
            else:
                print("[schedule-stats] Recording scheduling decisions (in-memory only)")

        self.expert_predictor = GatePredictor()
        self.predicted_next_demands = []
        self.latency_model = ExpertLatencyModel(
            t_io=self.latency_copy,
            latency_cpu_table=self.latency_cpu_table,
            latency_gpu_table=self.latency_gpu_table,
        )

        load_model_tick = time.time()
        self.bring_non_routed_expert_to_gpu()
        print("shared expert and other modules loaded to GPU, time:", time.time() - load_model_tick)

        n_expert_on_gpu = self.calc_n_expert_on_gpu()
        self.set_expert_loc(n_expert_on_gpu)
        print(
            f"Number of routed experts on GPU: {n_expert_on_gpu}/{(self.n_layer - 1) * self.n_expert}"
        )

        load_model_tick = time.time()
        self.cpu_experts = self.clone_cpu_experts()
        self.bring_expert_to_gpu()
        self.expert_executor = ExpertExecutionManager(
            device=self.dev,
            placeholder_manager=self.placeholder_manager,
            expert_provider=self.adapter,
            is_expert_in_gpu=self.is_expert_in_gpu,
            profile_timing=self.profile_expert_executor,
            cpu_experts=self.cpu_experts,
        )
        print("experts loaded to GPU, time:", time.time() - load_model_tick)
        print("Model is ready.")

    def _create_expert_strategy(self, args):
        if args.cpu_offload == 0:
            return GPUOnlyStrategy(self.dev, self.is_expert_in_gpu)
        if args.cpu_offload == 1:
            return PrefetchHybridStrategy(
                self.dev,
                self.is_expert_in_gpu,
                t_io=self.latency_copy,
                latency_cpu_table=self.latency_cpu_table,
                latency_gpu_table=self.latency_gpu_table,
            )
        return FiddlerStrategy(
            self.dev,
            self.is_expert_in_gpu,
            latency_cpu=self.latency_cpu,
            latency_gpu=self.latency_gpu,
            latency_io=self.latency_copy,
        )

    def _load_latency_tables(self, model_path):
        benchmark_data = self._load_benchmark_data(model_path)
        if benchmark_data is not None:
            try:
                self.latency_cpu = benchmark_data["expert_cpu"][0]["avg_time_ms"]
                self.latency_copy = benchmark_data["expert_weight_copy"]["avg_ms"]
                self.latency_gpu = benchmark_data["expert_gpu"][0]["avg_time_ms"]
                self.latency_cpu_table = _build_latency_lookup(benchmark_data["expert_cpu"])
                self.latency_gpu_table = _build_latency_lookup(benchmark_data["expert_gpu"])
                print(f"Loaded benchmark latency_cpu={self.latency_cpu:.4f}ms, latency_copy={self.latency_copy:.4f}ms, latency_gpu={self.latency_gpu:.4f}ms")
                print(f"Latency CPU table: {self.latency_cpu_table}")
                print(f"Latency GPU table: {self.latency_gpu_table}")
            except (KeyError, TypeError) as e:
                print(f"Warning: Malformed benchmark data, using defaults: {e}")
        else:
            print(f"Benchmark file not found, using default latency_cpu={self.latency_cpu}ms, latency_copy={self.latency_copy}ms")

    def _load_benchmark_data(self, model_path):
        model_name = os.path.basename(model_path.rstrip("/")).lower()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.abspath(os.path.join(script_dir, ".."))
        candidate_paths = [
            os.path.join(script_dir, f"micro_{model_name}.json"),
            os.path.join(src_dir, "benchmark", f"micro_{model_name}.json"),
        ]
        filepath = next((path for path in candidate_paths if os.path.exists(path)), None)
        if filepath is None:
            return None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"Warning: Failed to load benchmark file {filepath}: {e}")
            return None

    def _build_runtime_meta(self, args):
        return {
            "cpu_offload": args.cpu_offload,
            "model": getattr(args, "model", ""),
            "batch_size": args.batch_size,
            "beam_width": args.beam_width,
            "n_layer": self.n_layer,
            "n_expert": self.n_expert,
            "n_shared_experts": self.n_shared_experts,
            "num_placeholders": self.placeholder_manager.num_placeholders,
            "eviction_strategy": type(self.placeholder_manager._eviction_strategy).__name__ if self.placeholder_manager._eviction_strategy else None,
            "latency_cpu": self.latency_cpu,
            "latency_gpu": self.latency_gpu,
            "latency_copy": self.latency_copy,
            "latency_cpu_table": {str(k): v for k, v in self.latency_cpu_table.items()},
            "latency_gpu_table": {str(k): v for k, v in self.latency_gpu_table.items()},
            "strategy": self.expert_strategy.__class__.__name__,
        }

    def bring_non_routed_expert_to_gpu(self):
        raise NotImplementedError

    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        if popular_experts is None:
            hot_experts_file = getattr(self.adapter, "hot_experts_file", "./hot/deep.txt")
            if os.path.exists(hot_experts_file):
                try:
                    with open(hot_experts_file, "r", encoding="utf-8") as f:
                        popular_experts = [
                            tuple(map(int, line.strip().split(",")))
                            for line in f if line.strip()
                        ]
                    print(f"Loaded hot experts from {hot_experts_file}")
                except Exception as e:
                    print(f"Error loading hot experts: {e}")
            if popular_experts is None:
                popular_experts = []
                for layer in range(self.adapter.first_moe_layer, self.n_layer):
                    for expert in range(self.n_expert):
                        popular_experts.append((layer, expert))
        n_expert_on_gpu = min(n_expert_on_gpu, len(popular_experts))
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.placeholder_manager.mark_static_gpu_resident(i_layer, i_expert)

    def bring_expert_to_gpu(self):
        for i in range(self.adapter.first_moe_layer, self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.adapter.get_expert(i, j).to(self.dev)
                    self.placeholder_manager.mark_static_gpu_resident(i, j)
                else:
                    self.placeholder_manager.mark_cpu_resident(i, j)

    def clone_cpu_experts(self):
        cpu_experts = {}
        pinned_count = 0
        for i in range(self.adapter.first_moe_layer, self.n_layer):
            for j in range(self.n_expert):
                expert = copy.deepcopy(self.adapter.get_expert(i, j)).to("cpu")
                pinned_count += self.pin_module_tensors(expert)
                cpu_experts[(i, j)] = expert
        print(f"Pinned CPU expert source tensors: {pinned_count}")
        return cpu_experts

    def pin_module_tensors(self, module):
        pinned_count = 0
        for param in module.parameters():
            if param.device.type == "cpu" and not param.is_pinned():
                param.data = param.data.pin_memory()
                pinned_count += 1
        for buffer in module.buffers():
            if buffer.device.type == "cpu" and not buffer.is_pinned():
                buffer.data = buffer.data.pin_memory()
                pinned_count += 1
        return pinned_count

    def is_expert_in_gpu(self, i_layer, i_expert):
        return self.placeholder_manager.is_on_gpu(i_layer, i_expert)

    def calc_n_expert_on_gpu(self):
        fine_expert = self.adapter.get_template_expert()
        n_param = sum(p.numel() for p in fine_expert.parameters())
        bytes_per_param = 2 if self.dtype == torch.bfloat16 else 4
        expert_mem_mb = n_param * bytes_per_param / 1024 / 1024
        print(f"Number of parameters in a single expert: {n_param}, memory: {expert_mem_mb:.2f} MB")

        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        free_mem = total_mem * 0.20 - torch.cuda.memory_reserved(self.dev)
        print(f"Total GPU memory: {total_mem / 1024 / 1024:.2f} MB, Free GPU memory: {free_mem / 1024 / 1024:.2f} MB")
        return int(free_mem // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        if input_tensor.dim() == 3:
            input_tensor = input_tensor[:, -1, :]
        assert input_tensor.shape[-1] == self.beam_width
        return input_tensor.flatten()

    def tokenize(self, text, input_token=None):
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list):
            raise ValueError("text should be str or list of str")

        if len(text) < self.batch_size:
            text = text + [text[-1]] * (self.batch_size - len(text))
        elif len(text) > self.batch_size:
            text = text[:self.batch_size]

        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=input_token,
            return_tensors="pt",
        )
        input_ids = encodings.input_ids.to(self.dev)
        attention_mask = encodings.attention_mask.to(self.dev)
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.dev
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
        return input_ids, position_ids, attention_mask

    def generate(self, text=None, output_token=20, input_token=None, profiler=None, profile_decode_steps=1):
        torch.set_num_threads(16)
        self.reset_runtime_state(clear_placeholders=False)

        if text is None:
            text = ["default input"] * self.batch_size
        elif isinstance(text, str):
            text = [text] * self.batch_size

        input_ids, position_ids, attention_mask = self.tokenize(text, input_token)
        if self.sync_timing and self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        tick = time.time()
        is_decode = False
        prefill_time, decode_time = 0, 0
        original_batch_size = input_ids.shape[0]
        generated_token_chunks = []
        search_start = False
        probs = torch.full((input_ids.shape[0],), 1.0, device=self.dev)

        for i_token in range(output_token):
            if profiler is not None and i_token == 0:
                profiler.start()
                print("[profiler] profiling started (prefill)")

            if is_decode:
                past_seq_len = self.past_key_values_length
                attention_mask = torch.ones(
                    input_ids.shape[0], past_seq_len + 1,
                    dtype=torch.long, device=self.dev,
                )

            new_position_ids = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

            cache_position = torch.arange(
                self.past_key_values_length,
                self.past_key_values_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.dev,
            )

            logits = self.mixtral_forward(input_ids, new_position_ids, attention_mask, cache_position, is_prefill=not is_decode)
            logits = F.softmax(logits, dim=-1)
            self.past_key_values_length += logits.shape[1]

            if search_start:
                new_probs, output = torch.topk(logits, 1, dim=-1)
                new_probs = new_probs[:, -1].flatten()
                output = output[:, -1].flatten()
                probs = probs * new_probs
            else:
                new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                new_probs = self.initial_beam_tensor(new_probs)
                output = self.initial_beam_tensor(output)
                probs = new_probs
                search_start = True

            generated_token_chunks.append(output.view(-1, 1))
            if search_start:
                input_ids = output.view(-1, 1)
            else:
                input_ids = output.view(-1, self.beam_width)[:, 0]

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )

            if not is_decode:
                if self.sync_timing and self.dev.type == "cuda":
                    print("Prefill阶段完成,等待GPU同步...")
                    torch.cuda.synchronize(self.dev)
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True

            if profiler is not None and is_decode and i_token >= profile_decode_steps:
                profiler.stop()
                print(f"[profiler] profiling stopped after decode step {i_token}")
                profiler = None

        if self.sync_timing and self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)
        decode_time = time.time() - tick
        probs = probs.view(-1, self.beam_width)
        max_ids = torch.argmax(probs, dim=-1)
        decoded_outputs = [""] * original_batch_size
        if generated_token_chunks:
            generated_token_ids = torch.cat(generated_token_chunks, dim=1)
            selected_tokens = generated_token_ids.view(original_batch_size, self.beam_width, -1)[
                torch.arange(original_batch_size, device=self.dev), max_ids
            ]
            decoded_outputs = self.tokenizer.batch_decode(selected_tokens.detach().cpu(), skip_special_tokens=False)

        print("--------------------")
        print(f"Input: {text}")
        print(f"Output: {decoded_outputs[0]}")
        self.monitor.record_generation_timing(prefill_time, decode_time)
        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit / max(self.cnt_expert_all, 1),
        )

    def reset_runtime_state(self, clear_placeholders=False):
        self.past_key_value = transformers.cache_utils.DynamicCache()
        self.past_key_values_length = 0
        self.predicted_next_demands = []
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0

        if hasattr(self.expert_strategy, "cnt_expert_hit"):
            self.expert_strategy.cnt_expert_hit = 0
        if hasattr(self.expert_strategy, "cnt_expert_all"):
            self.expert_strategy.cnt_expert_all = 0

        if hasattr(self, "expert_executor"):
            self.expert_executor.preload_request_count = 0
            self.expert_executor.preload_success_count = 0
            self.expert_executor.preload_skip_count = 0
            self.expert_executor.preload_hit_count = 0
            self.expert_executor.static_gpu_hit_count = 0
            self.expert_executor.static_gpu_hit_tokens = 0
            self.expert_executor.placeholder_hit_count = 0
            self.expert_executor.placeholder_hit_tokens = 0
            self.expert_executor.ondemand_load_count = 0
            self.expert_executor.ondemand_load_tokens = 0
            self.expert_executor.reset_timing_stats()

        if clear_placeholders:
            self.placeholder_manager.clear_dynamic_placeholders()
        self.placeholder_manager.reset_stats()
        self.monitor.reset()

    def reset_hot_expert_stats(self):
        self.monitor.reset_hot_expert_stats()

    def record_hot_expert_selection(self, i_layer, selected_experts):
        self.monitor.record_hot_experts(i_layer, selected_experts)

    def sorted_hot_expert_stats(self):
        return self.monitor.sorted_hot_expert_stats()

    def export_hot_expert_stats(self):
        return self.monitor.export_hot_expert_stats()

    def _prefetch_next_layer_experts(self, i_layer, prefetch_experts):
        next_layer = i_layer + 1
        if next_layer >= self.n_layer:
            return
        next_experts = self.adapter.get_routed_experts(next_layer)
        tick = time.time()
        loaded = 0
        for expert_id in prefetch_experts:
            if self.is_expert_in_gpu(next_layer, expert_id):
                continue
            placeholder = self.placeholder_manager.acquire_placeholder(next_layer, expert_id)
            if placeholder is None:
                break
            self.placeholder_manager.load_weights(placeholder, next_experts[expert_id])
            loaded += 1
        elapsed = time.time() - tick
        if loaded > 0:
            print(f"  Prefetch layer {next_layer}: {loaded} experts loaded in {elapsed*1000:.2f}ms")

    def run_expert_at_cpu(self, i_layer, i_expert, inps):
        return self.adapter.get_expert(i_layer, i_expert)(inps)

    def _execute_gpu_experts(self, i_layer, experts, gpu_experts, expert_assignments, inps_flat, hidden_dim):
        result = torch.zeros_like(inps_flat, device=self.dev)
        for i_expert in gpu_experts:
            top_2, routing_weight_subset = expert_assignments[i_expert]
            current_state = inps_flat[None, top_2.tolist()].reshape(-1, hidden_dim)
            if self.is_expert_in_gpu(i_layer, i_expert):
                current_state = experts[i_expert](current_state)
            else:
                placeholder = self.placeholder_manager.acquire_placeholder(i_layer, i_expert)
                self.placeholder_manager.load_weights(placeholder, experts[i_expert])
                current_state = placeholder(current_state)
                self.placeholder_manager.release_by_layer(i_layer)
            current_state = current_state * routing_weight_subset
            result.index_add_(0, top_2.to(self.dev, non_blocking=True), current_state.to(result.dtype))
        return result

    def _execute_cpu_experts(self, i_layer, experts, cpu_experts, expert_assignments, inps_flat, hidden_dim):
        result = torch.zeros_like(inps_flat, device="cpu")
        for i_expert in cpu_experts:
            top_2, routing_weight_subset = expert_assignments[i_expert]
            current_state = inps_flat[None, top_2.tolist()].reshape(-1, hidden_dim)
            current_state = self.run_expert_at_cpu(i_layer, i_expert, current_state.to("cpu"))
            current_state = current_state * routing_weight_subset.to("cpu")
            result.index_add_(0, top_2.to("cpu", non_blocking=True), current_state.to(result.dtype))
        return result
