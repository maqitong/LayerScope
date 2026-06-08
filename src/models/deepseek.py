from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2RotaryEmbedding
from transformers.masking_utils import create_causal_mask

from strategies.scheduling.expert_scheduling import FiddlerStrategy, PrefetchHybridStrategy
from common.types import ExpertDemand, ExpertKey, ExpertLayerContext, ExpertSchedule, build_assignments
from models.base_moe import BaseMoERuntime

class DeepSeekV2Adapter:
    first_moe_layer = 1
    hot_experts_file = "./hot/deep.txt"

    def __init__(self, args, dtype, device):
        config = transformers.AutoConfig.from_pretrained(args.model)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=dtype,
        )
        self.lm_head = hf_model.lm_head
        self.model = hf_model.model
        self.config = self.model.config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.rotary_emb = DeepseekV2RotaryEmbedding(config=self.config, device=device)

    @property
    def layers(self):
        return self.model.layers

    @property
    def num_layers(self) -> int:
        return len(self.model.layers)

    @property
    def num_routed_experts(self) -> int:
        return self.config.n_routed_experts

    @property
    def num_shared_experts(self) -> int:
        return 2

    @property
    def num_experts_per_tok(self) -> int:
        return self.config.num_experts_per_tok

    def get_template_expert(self):
        return self.get_expert(self.first_moe_layer, 0)

    def get_routed_experts(self, layer: int):
        return self.model.layers[layer].mlp.experts

    def get_expert(self, layer: int, expert_id: int):
        return self.get_routed_experts(layer)[expert_id]

    def get_gate(self, layer: int):
        return self.model.layers[layer].mlp.gate

    def get_shared_experts(self, layer: int):
        return self.model.layers[layer].mlp.shared_experts


class mDeepSeek(BaseMoERuntime):
    adapter_cls = DeepSeekV2Adapter


    def bring_non_routed_expert_to_gpu(self):
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        self.model.layers[0].to(self.dev) # 第一层包含非 MoE MLP，先加载到 GPU
        for i in range(len(self.model.layers)):
            if i != 0:
                self.model.layers[i].self_attn.to(self.dev)
                self.model.layers[i].input_layernorm.to(self.dev)
                self.model.layers[i].mlp.gate.to(self.dev)
                self.model.layers[i].post_attention_layernorm.to(self.dev)
        for i in range(1, self.n_layer):
            self.model.layers[i].mlp.shared_experts.to(self.dev)

    @torch.no_grad()
    def mixtral_forward(self, input_ids, position_ids, attention_mask, cache_position, is_prefill=False):
        hidden_dim = self.model.config.hidden_size

        inps = self.model.embed_tokens(input_ids)

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 计算 position_embeddings (cos, sin tuple)
        position_embeddings = self.rotary_emb(inps, position_ids)

        # 使用 transformers 官方的 create_causal_mask 创建正确的 causal mask
        # 替换原来错误的 padding mask 扩展逻辑
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inps,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=self.past_key_value,
            position_ids=position_ids,
        )

        for i_layer, layer in enumerate(self.model.layers):
            if i_layer == 0:
                # 第一层：完整的 attention + dense FFN（非 MoE）
                original_inps_shape = inps.shape
                inps_residual = inps
                inps = layer.input_layernorm(inps)
                inps = inps.view(batch_size, seq_len, hidden_dim)

                attn_output = layer.self_attn(
                    hidden_states=inps,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=self.past_key_value,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                if isinstance(attn_output, tuple):
                    if len(attn_output) == 2:
                        inps, present_key_value = attn_output
                    else:
                        inps, _, present_key_value = attn_output
                else:
                    present_key_value = None

                inps = inps_residual + inps
                inps_residual = inps
                inps = layer.post_attention_layernorm(inps)
                inps = inps.view(batch_size, seq_len, hidden_dim)
                # 第一层使用 dense MLP（非 MoE）
                inps = layer.mlp(inps)
                inps = inps_residual + inps
                continue

            original_inps_shape = inps.shape
            inps_residual = inps
            inps = layer.input_layernorm(inps)

            inps = inps.view(batch_size, seq_len, hidden_dim)

            attn_output = layer.self_attn(
                hidden_states=inps,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if isinstance(attn_output, tuple):
                if len(attn_output) == 2:
                    inps, present_key_value = attn_output
                else:
                    inps, _, present_key_value = attn_output
            else:
                present_key_value = None

            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            inps = inps.view(batch_size, seq_len, hidden_dim)

            selected_experts, routing_weights = layer.mlp.gate(inps)
            if self.record_hot_experts:
                self.record_hot_expert_selection(i_layer, selected_experts)

            # 与当前层专家执行并行：预测下一层活跃专家
            predict_future = None
            if i_layer + 1 < self.n_layer:
                with ThreadPoolExecutor(max_workers=1) as pred_executor:
                    predict_future = pred_executor.submit(
                        self.expert_predictor.predict, inps, self.adapter, i_layer, 1
                    )

            # 收集预测结果（在专家执行完成后应已就绪）
            if predict_future is not None:
                pred_result = predict_future.result()
                self.predicted_next_demands = pred_result or []
            else:
                self.predicted_next_demands = []

            shared_output = self.adapter.get_shared_experts(i_layer)(inps)

            inps_flat = inps.view(-1, hidden_dim)
            experts = self.adapter.get_routed_experts(i_layer)
        
            # 选择策略
            strategy = self.expert_strategy

            # 策略决策和预处理（PrefetchHybridStrategy 返回 4-tuple）
            result_tuple = strategy.decide_and_prepare(
                i_layer, experts, selected_experts, routing_weights, self.n_expert,
                future_demands=self.predicted_next_demands,
                placement=self.placeholder_manager.snapshot(),
                is_prefill=is_prefill,
            )
            if len(result_tuple) == 4:
                cpu_experts, gpu_experts, prefetch_experts, expert_assignments = result_tuple
            else:
                cpu_experts, gpu_experts, expert_assignments = result_tuple
                prefetch_experts = []
            # print(f"Layer {i_layer}: GPU experts: {gpu_experts}, CPU experts: {cpu_experts}, Prefetch: {prefetch_experts}")

            # 更新统计
            if isinstance(strategy, FiddlerStrategy) or isinstance(strategy, PrefetchHybridStrategy):
                self.cnt_expert_hit = self.expert_strategy.cnt_expert_hit
                self.cnt_expert_all = self.expert_strategy.cnt_expert_all
            

            schedule = ExpertSchedule(
                cpu=[ExpertDemand(ExpertKey(i_layer, eid), expert_assignments[eid][0].shape[0]) for eid in cpu_experts],
                gpu=[ExpertDemand(ExpertKey(i_layer, eid), expert_assignments[eid][0].shape[0]) for eid in gpu_experts],
                preload=[ExpertDemand(ExpertKey(i_layer + 1, eid), 1, source="predicted") for eid in prefetch_experts],
            )
            context = ExpertLayerContext(
                layer=i_layer,
                experts=experts,
                inps_flat=inps_flat,
                hidden_dim=hidden_dim,
                assignments=build_assignments(expert_assignments),
            )
            inps_after_experts = self.expert_executor.execute(schedule, context)
            
            total_expert_output = shared_output.view(-1, hidden_dim) + inps_after_experts
            inps = inps_residual + total_expert_output.reshape(batch_size, seq_len, hidden_dim)

        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)

        self.present_key_value = present_key_value
        return lm_logis
