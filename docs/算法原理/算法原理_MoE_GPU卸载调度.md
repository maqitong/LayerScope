# MoE GPU 卸载调度算法原理

本文档基于当前 `src/model/expert_scheduling.py` 重写，重点说明 MoE 推理中不同调度算法在 prefill 与 decode 阶段的决策逻辑。

当前代码中的调度策略分为三层：

| 层级 | 类 / 函数 | 作用 |
| --- | --- | --- |
| 路由预处理 | `_collect_expert_assignments()` | 从 gate 输出构造活跃专家、token 索引和路由权重 |
| 策略适配 | `GPUOnlyStrategy` / `FiddlerStrategy` / `PrefetchHybridStrategy` | 把当前层 MoE 路由结果转成 CPU/GPU/preload 决策 |
| 阶段感知调度 | `PDScopeScheduler` | 在 `PrefetchHybridStrategy` 内部区分 prefill 与 decode |

---

# 1. 调度输入与输出

## 1.1 gate 输出

每个 MoE 层先执行 gate：

```python
selected_experts, routing_weights = layer.mlp.gate(inps)
```

其中：

| 张量 | 含义 |
| --- | --- |
| `selected_experts` | 每个 token 选择的 top-k expert id |
| `routing_weights` | 每个 token 对应 top-k expert 的路由权重 |

这些张量通常在 GPU 上。

## 1.2 `_collect_expert_assignments()`

位置：`src/model/expert_scheduling.py:169`

该函数是所有策略共同使用的路由预处理入口：

```python
active_experts, token_indices_by_expert, expert_assignments = _collect_expert_assignments(
    selected_experts,
    routing_weights,
    n_expert,
)
```

输出：

| 输出 | 类型 | 含义 |
| --- | --- | --- |
| `active_experts` | `List[int]` | 当前层实际有 token 分配的专家 id |
| `token_indices_by_expert` | `Dict[int, Tensor]` | 每个专家负责哪些 token position |
| `expert_assignments` | `Dict[int, Tuple[Tensor, Tensor]]` | 每个专家对应的 token indices 和 routing weights |

当前实现流程：

```python
flat_experts = selected_experts.reshape(-1)
top_k = selected_experts.shape[-1]
flat_weights = routing_weights.reshape(-1)

sorted_experts, order = torch.sort(flat_experts)
unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
token_positions = torch.div(order, top_k, rounding_mode="floor")
routing_weight_values = flat_weights.index_select(0, order).unsqueeze(-1)

unique_cpu = unique_experts.detach().cpu().tolist()
counts_cpu = counts.detach().cpu().tolist()
```

关键点：

- 先在 GPU 上完成 sort、unique、token position 重排。
- 只把压缩后的 `unique_experts` 和 `counts` 转成 CPU list。
- `token_indices` 与 `routing_weight_subset` 保持 tensor，后续 executor 使用。
- 当前默认假设 gate 输出的 expert id 合法，不再在热路径执行 `torch.all(valid)` 检查。

## 1.3 调度输出

执行器最终消费的是 `ExpertSchedule` 和 `ExpertLayerContext`：

```python
schedule = ExpertSchedule(
    cpu=[...],
    gpu=[...],
    preload=[...],
)

context = ExpertLayerContext(
    layer=i_layer,
    experts=experts,
    inps_flat=inps_flat,
    hidden_dim=hidden_dim,
    assignments=build_assignments(expert_assignments),
)
```

`ExpertSchedule` 中的三类专家含义：

| 字段 | 含义 |
| --- | --- |
| `cpu` | 当前层在 CPU expert copy 上执行的专家 |
| `gpu` | 当前层在 GPU 常驻 expert 或 placeholder 上执行的专家 |
| `preload` | 下一层计划提前加载到 GPU placeholder 的专家 |

---

# 2. 三种策略总览

`mDeepSeek.__init__()` 根据 `args.cpu_offload` 选择策略：

| `cpu_offload` | 策略类 | 语义 |
| --- | --- | --- |
| `0` | `GPUOnlyStrategy` | 所有当前活跃专家都走 GPU |
| `1` | `PrefetchHybridStrategy` | PDScope/AdaptSched 风格，区分 prefill 和 decode |
| `2` | `FiddlerStrategy` | 每个专家独立做 CPU/GPU 成本比较 |

整体差异：

| 策略 | 是否区分 prefill/decode | 是否使用未来专家预测 | 是否产生 preload | 决策粒度 |
| --- | --- | --- | --- | --- |
| `GPUOnlyStrategy` | 不区分 | 否 | 否 | 当前活跃专家全部 GPU |
| `FiddlerStrategy` | 只记录 phase，不改变算法 | 否 | 否 | 每个当前专家独立比较成本 |
| `PrefetchHybridStrategy` | 明确区分 | 是 | 是 | 当前层 + 下一层联合调度 |

---

# 3. GPUOnlyStrategy

位置：`src/model/expert_scheduling.py:290`

## 3.1 核心逻辑

`GPUOnlyStrategy` 是 baseline 策略：所有当前活跃专家都分配给 GPU。

```python
active_experts, token_indices_by_expert, expert_assignments = _collect_expert_assignments(...)
return [], active_experts, expert_assignments
```

返回值含义：

| 返回项 | 内容 |
| --- | --- |
| `cpu_experts` | 空列表 |
| `gpu_experts` | `active_experts` 全部专家 |
| `expert_assignments` | 当前层 token 分配 |

## 3.2 Prefill 阶段

GPUOnly 在 prefill 阶段没有特殊逻辑。

流程：

1. 收集当前层所有活跃专家。
2. 所有活跃专家都放入 `gpu_experts`。
3. 不生成 `preload_experts`。
4. 不考虑 CPU 执行成本、I/O 成本、下一层预测。

伪代码：

```python
if phase == "prefill":
    gpu = active_experts
    cpu = []
    preload = []
```

## 3.3 Decode 阶段

decode 阶段与 prefill 相同。

```python
if phase == "decode":
    gpu = active_experts
    cpu = []
    preload = []
```

## 3.4 适用场景

适合：

- GPU 显存足以容纳当前需要执行的专家。
- 需要 GPU-only baseline。
- 分析 CPU offload 策略收益时作为对照组。

局限：

- 不使用 CPU expert 计算能力。
- 不做下一层预取。
- 如果活跃专家不在 GPU，需要 executor 通过 placeholder 动态加载，实际开销仍取决于执行器和 placeholder 状态。

---

# 4. FiddlerStrategy

位置：`src/model/expert_scheduling.py:325`

## 4.1 核心思想

`FiddlerStrategy` 对每个当前活跃专家独立比较 CPU 执行成本和 GPU 执行成本。

```python
cost_cpu = token_count * self.latency_cpu
cost_gpu = self.latency_gpu + self.latency_io

if resident:
    cost_gpu = 0

if cost_cpu < cost_gpu:
    cpu_experts.append(i_expert)
else:
    gpu_experts.append(i_expert)
```

参数：

| 参数 | 含义 |
| --- | --- |
| `latency_cpu` | CPU 上每 token 专家计算延迟 |
| `latency_gpu` | GPU 上执行一个专家的延迟 |
| `latency_io` | 将专家权重从 CPU 搬到 GPU 的延迟 |

## 4.2 成本模型

对每个专家：

```text
CPU 成本 = token_count × latency_cpu
GPU 成本 = latency_gpu + latency_io
```

如果专家已经在 GPU 上：

```text
GPU 成本 = 0
```

因此，Fiddler 的决策规则是：

```text
如果 CPU 成本 < GPU 成本：放 CPU
否则：放 GPU
```

## 4.3 Prefill 阶段

Fiddler 不显式区分 prefill 和 decode，prefill 只是 `is_prefill=True` 的一次普通调用。

prefill 中 `token_count` 通常更大，因为输入序列长度较长，同一个专家可能处理多个 token。

因此：

- `cost_cpu = token_count × latency_cpu` 更容易变大。
- 当 `token_count` 较大时，GPU 更容易胜出。
- 已经在 GPU 上的专家一定走 GPU，因为 `cost_gpu = 0`。

伪代码：

```python
for expert in active_experts:
    token_count = token_indices_by_expert[expert].shape[0]
    cost_cpu = token_count * latency_cpu
    cost_gpu = 0 if expert_on_gpu else latency_gpu + latency_io
    assign_to_cpu_if(cost_cpu < cost_gpu)
```

## 4.4 Decode 阶段

decode 阶段也使用同一规则。

区别在于 decode 通常 `seq_len = 1`，单个专家的 `token_count` 更小。

因此：

- `cost_cpu` 更小。
- 对不在 GPU 的专家，CPU 执行更可能便宜。
- 对已在 GPU 的专家，仍然直接走 GPU。

## 4.5 命中率统计

Fiddler 维护：

```python
self.cnt_expert_hit
self.cnt_expert_all
```

逻辑：

```python
if resident:
    self.cnt_expert_hit += token_count
self.cnt_expert_all += token_count
```

这里统计的是 token 级别的 GPU resident 命中率。

## 4.6 特点与局限

优点：

- 决策简单。
- 每个专家独立比较成本。
- 对 decode 小 token_count 场景，倾向让冷专家留在 CPU，减少不划算的 HtoD 权重搬运。

局限：

- 不考虑下一层未来需求。
- 不产生 preload。
- 不显式建模 CPU/GPU 两侧整体负载平衡，只做 per-expert 局部决策。
- prefill 和 decode 使用同一算法，阶段差异只通过 `token_count` 间接体现。

---

# 5. PrefetchHybridStrategy 与 PDScopeScheduler

位置：

- `PrefetchHybridStrategy`：`src/model/expert_scheduling.py:745`
- `PDScopeScheduler`：`src/model/expert_scheduling.py:474`

这是当前最完整的阶段感知策略。

## 5.1 两层结构

`PrefetchHybridStrategy` 是适配层，负责把模型 forward 中的原始路由结果转换为结构化请求。

```python
current_demands = build_current_demands(i_layer, active_experts, token_indices_by_expert)
future = future_demands or build_future_demands(i_layer + 1, predicted_next_experts, predicted_next_weights)
future = unique_demands(future)

request = ExpertLayerRequest(
    layer=i_layer,
    phase="prefill" if is_prefill else "decode",
    current=current_demands,
    future=future,
    assignments=build_assignments(raw_assignments),
)

schedule = self.scheduler.schedule(request, placement, self.latency_model)
```

`PDScopeScheduler` 是实际决策层：

```python
if request.phase == "prefill":
    return self.schedule_prefill(request, placement, latency)
return self.schedule_decode(request, placement, latency)
```

## 5.2 调度数据结构

`current_demands` 来自当前层实际 gate 输出：

```python
ExpertDemand(
    key=ExpertKey(layer=i_layer, expert_id=expert_id),
    token_count=token_count,
    score=float(token_count),
    source="current",
)
```

`future_demands` 来自下一层预测：

```python
ExpertDemand(
    key=ExpertKey(layer=i_layer + 1, expert_id=expert_id),
    token_count=count,
    score=score,
    source="predicted",
)
```

`placement` 是当前专家驻留快照：

| 字段 | 含义 |
| --- | --- |
| `gpu_resident` | 静态 GPU 常驻专家 |
| `placeholder_resident` | 当前 placeholder 中已加载专家 |
| `loading` | 正在加载的专家 |
| `cpu_resident` | CPU 侧专家 |
| `free_placeholders` | 空闲 placeholder 数量 |

`placement.is_on_gpu(layer, expert)` 同时检查：

```python
key in gpu_resident or key in placeholder_resident
```

---

# 6. PDScope prefill 调度

位置：`PDScopeScheduler.schedule_prefill()`

## 6.1 目标

prefill 阶段输入 token 多、当前层专家需求较集中。该阶段策略的目标是：

- 当前层必须完成执行，因此先保证当前层专家在 CPU/GPU 间合理分配。
- 同时利用 CPU/GPU 执行时间差形成的 I/O 气泡，预加载下一层预测专家。

## 6.2 步骤 1：划分当前与未来专家

```python
current = request.current
current_resident = [d for d in current if placement.is_on_gpu(d.key.layer, d.key.expert_id)]
current_non_resident = [d for d in current if not placement.is_on_gpu(d.key.layer, d.key.expert_id)]
future_non_resident = [
    d for d in request.future
    if not placement.is_on_gpu(d.key.layer, d.key.expert_id)
    and (d.key.layer, d.key.expert_id) not in placement.loading
]
```

含义：

| 集合 | 含义 |
| --- | --- |
| `current_resident` | 当前层已经在 GPU 的专家 |
| `current_non_resident` | 当前层还不在 GPU 的专家 |
| `future_non_resident` | 下一层预测需要且当前不在 GPU/不在 loading 的专家 |

## 6.3 步骤 2：构造 combined 队列

```python
combined = current_non_resident + future_non_resident
combined.sort(key=lambda d: (d.token_count, d.score))
```

排序方向是升序。后续 `_select_global_queue()` 会从左到右尝试切分，返回右侧子队列，所以 token_count/score 越大的需求越靠右，越可能被选入 GPU 候选。

## 6.4 步骤 3：选择全局 GPU 候选队列

位置：`_select_global_queue()`

```python
for i in range(len(demands)):
    gpu_side = demands[i:]
    cpu_side = demands[:i]
    t_gpu = self.alpha + len(gpu_side) * latency.t_io + latency.gpu_compute(1)
    t_cpu = sum(latency.cpu(d.token_count) for d in cpu_side) + self.t_attn
    if t_gpu < t_cpu:
        return gpu_side
return []
```

含义：

| 变量 | 含义 |
| --- | --- |
| `gpu_side` | 假设搬到 GPU 的候选专家集合 |
| `cpu_side` | 留给 CPU 的候选专家集合 |
| `t_gpu` | GPU 侧搬运 + GPU 计算估计时间 |
| `t_cpu` | CPU 侧计算 + attention 窗口估计时间 |

决策规则：

```text
找到第一个切分点，使 GPU 侧时间 < CPU 侧时间。
该切分点右侧专家进入 global_queue。
```

## 6.5 步骤 4：选择当前层 ondemand 专家

位置：`_select_current_ondemand()`

```python
current_global = [d for d in global_queue if d.source == "current"]
ondemand, t_gpu, t_cpu = self._select_current_ondemand(current_global, latency)
```

内部逻辑：

```python
for i in range(len(current_global) + 1):
    gpu_side = current_global[i:]
    cpu_side = current_global[:i]
    t_compute = sum(latency.gpu_compute(d.token_count) for d in gpu_side)
    t_io = len(gpu_side) * latency.t_io
    t_gpu = max(t_compute, self.alpha + t_io) + latency.gpu_compute(1) if gpu_side else 0.0
    t_cpu = sum(latency.cpu(d.token_count) for d in cpu_side)
    if gpu_side and t_gpu < t_cpu:
        return gpu_side, t_gpu, t_cpu
```

含义：

- 只从当前层候选中选择 ondemand。
- 如果把右侧当前层专家搬到 GPU 后，GPU 时间低于 CPU 时间，则这些专家值得按需加载到 GPU。
- 返回的 `t_gpu/t_cpu` 会继续用于预加载容量计算。

## 6.6 步骤 5：生成当前层 CPU/GPU 决策

```python
gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
```

规则：

```text
当前层已在 GPU 的专家 + 被选中 ondemand 的当前专家 -> GPU
其他当前层专家 -> CPU
```

## 6.7 步骤 6：选择 preload 专家

位置：`_select_preload()`

```python
t_gap = max(0.0, t_cpu - t_gpu)
capacity = math.floor((t_gap + self.t_attn) / max(latency.t_io, 1e-9))
xi = (2 * self.r_hit - 1) * latency.t_io
future = [d for d in global_queue if d.source == "predicted"]
future.sort(key=lambda d: (d.score, d.token_count), reverse=True)
return future[:capacity]
```

含义：

| 变量 | 含义 |
| --- | --- |
| `t_gap` | CPU 比 GPU 更慢时留下的 I/O 气泡 |
| `capacity` | 气泡窗口内最多可搬运的专家数 |
| `r_hit` | 预期 placeholder 命中率 |
| `xi` | 命中收益估计，非正则不预取 |

preload 只从 `global_queue` 中的 `source == "predicted"` 需求选择，并按 `score/token_count` 降序取前 `capacity` 个。

## 6.8 prefill 输出

```python
return ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="prefill")
```

---

# 7. PDScope decode 调度

位置：`PDScopeScheduler.schedule_decode()`

## 7.1 目标

decode 阶段通常每步只有新 token，当前层 token_count 较小，但未来层会重复出现。decode 策略不再直接沿用 prefill 的全局切分，而是先判断当前层和下一层的 GPU 驻留是否充足，再进入不同模式。

## 7.2 空当前层

```python
if not current:
    return ExpertSchedule(reason="decode-empty")
```

如果当前层没有活跃专家，直接返回空 schedule。

## 7.3 计算理想 GPU 专家数 `n_g_rho`

```python
k = len(current)
t_c = latency.cpu(1)
t_g = latency.gpu_compute(1)
n_g_rho = min(
    range(k + 1),
    key=lambda n_g: max(n_g * t_g, (k - n_g) * t_c),
)
```

含义：

| 变量 | 含义 |
| --- | --- |
| `k` | 当前层活跃专家数量 |
| `t_c` | 单 token CPU 专家延迟 |
| `t_g` | 单 token GPU 专家延迟 |
| `n_g_rho` | 使 CPU/GPU 两侧最大耗时最小的 GPU 专家数 |

目标函数：

```text
minimize max(n_g × t_g, (k - n_g) × t_c)
```

即希望当前层分给 GPU 的专家数与分给 CPU 的专家数达到负载平衡。

## 7.4 判断当前层与下一层驻留是否充足

```python
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

cur_below = len(current_resident) <= n_g_rho
next_below = next_resident_count <= n_g_rho
```

注意：当前代码使用 `<=`，因此当 resident 数量等于 `n_g_rho` 时，也被视为 below。

## 7.5 四种 decode 模式

| 条件 | reason | 策略 |
| --- | --- | --- |
| `cur_below and next_below` | `decode-fallback-prefill` | 回退到 prefill 策略 |
| `cur_below and not next_below` | `decode-mode-a` | 补当前层 GPU 专家 |
| `not cur_below and next_below` | `decode-mode-b` | 保持当前层 GPU 数，预加载下一层 |
| `not cur_below and not next_below` | `decode-mode-c` | 不预加载，当前层按理想 GPU 数执行 |

## 7.6 mode fallback：当前层和下一层都不足

```python
if cur_below and next_below:
    schedule = self.schedule_prefill(request, placement, latency)
    schedule.reason = "decode-fallback-prefill"
    return schedule
```

含义：

- 当前层 GPU 驻留不足。
- 下一层预测专家 GPU 驻留也不足。
- 使用 prefill 的全局队列策略，同时考虑当前 ondemand 和未来 preload。

这是 decode 中最激进的补救路径。

## 7.7 mode-a：当前不足，下一层充足

```python
need = min(n_g_rho - len(current_resident), len(current_non_resident))
ondemand = current_non_resident[:need]
gpu_keys = {(d.key.layer, d.key.expert_id) for d in current_resident + ondemand}
gpu = [d for d in current if (d.key.layer, d.key.expert_id) in gpu_keys]
cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
return ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-a")
```

含义：

- 下一层已经足够，不需要预取。
- 优先补齐当前层 GPU 专家数到 `n_g_rho`。
- 未进入 GPU 的当前专家留在 CPU。

注意：`current_non_resident` 没有在 mode-a 中重新按 score/token_count 排序，而是沿用 `current` 原顺序过滤后的结果。

## 7.8 mode-b：当前充足，下一层不足

```python
need_next = max(0, n_g_rho - next_resident_count)
preload = future_non_resident[:need_next]
gpu = current_resident[:n_g_rho]
gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
return ExpertSchedule(cpu=cpu, gpu=gpu, preload=preload, evict=[], reason="decode-mode-b")
```

含义：

- 当前层 GPU resident 数量足够。
- 下一层不足，因此从 `future_non_resident` 中按 `score/token_count` 降序选择 `need_next` 个预加载。
- 当前层只保留前 `n_g_rho` 个 resident 专家走 GPU，其余当前专家走 CPU。

## 7.9 mode-c：当前和下一层都充足

```python
gpu = current_resident[:n_g_rho]
gpu_keys = {(d.key.layer, d.key.expert_id) for d in gpu}
cpu = [d for d in current if (d.key.layer, d.key.expert_id) not in gpu_keys]
return ExpertSchedule(cpu=cpu, gpu=gpu, preload=[], evict=[], reason="decode-mode-c")
```

含义：

- 当前层已满足理想 GPU 专家数。
- 下一层也满足预期驻留数量。
- 不再做 preload。
- 当前层只让 `n_g_rho` 个 resident 专家走 GPU，其余走 CPU。

---

# 8. 三种算法的 prefill/decode 对比

## 8.1 Prefill 阶段

| 策略 | 当前层 GPU 决策 | 当前层 CPU 决策 | 下一层 preload | 使用成本模型 |
| --- | --- | --- | --- | --- |
| GPUOnly | 全部活跃专家 GPU | 无 | 无 | 无 |
| Fiddler | 每专家 `cost_cpu >= cost_gpu` 时 GPU | 每专家 `cost_cpu < cost_gpu` 时 CPU | 无 | per-expert CPU/GPU 成本 |
| PrefetchHybrid | resident + `_select_current_ondemand()` | 未进入 GPU 的 current | `_select_preload()` | global queue + I/O 气泡 |

## 8.2 Decode 阶段

| 策略 | 当前层 GPU 决策 | 当前层 CPU 决策 | 下一层 preload | 阶段特化 |
| --- | --- | --- | --- | --- |
| GPUOnly | 全部活跃专家 GPU | 无 | 无 | 无 |
| Fiddler | 每专家成本比较 | 每专家成本比较 | 无 | 无，token_count 间接影响 |
| PrefetchHybrid | 基于 `n_g_rho` 和 resident 充足性 | 非 GPU current | mode-b/fallback 中可能 preload | ABC/fallback 模式 |

## 8.3 策略行为总结

| 策略 | 优点 | 局限 |
| --- | --- | --- |
| GPUOnly | 简单，适合作为 baseline | 不利用 CPU，不预测未来，不控制 HtoD 成本 |
| Fiddler | 决策直接，decode 小 token 场景倾向减少不划算搬运 | 不看未来层，不做 preload，缺少全局负载规划 |
| PrefetchHybrid | 同时考虑 current/future、prefill/decode、resident/loading、I/O 气泡 | 依赖预测质量和 latency model，逻辑更复杂 |

---

# 9. 调度统计记录

所有策略继承 `ExpertSchedulingStrategy`，可选挂载 `ExpertSchedulingStatsRecorder`：

```python
strategy.set_stats_recorder(recorder)
```

记录内容包括：

| 字段 | 含义 |
| --- | --- |
| `strategy` | 策略类名 |
| `phase` | `prefill` 或 `decode` |
| `reason` | 决策原因，如 `gpu-only`、`fiddler-cost-opt`、`decode-mode-b` |
| `active_experts` | 当前层活跃专家 |
| `cpu_experts` | CPU 执行专家 |
| `gpu_experts` | GPU 执行专家 |
| `preload_experts` | 预加载专家 |
| `token_counts` | 每个当前专家负责的 token 数 |
| `placement` | 当前驻留状态摘要 |
| `latency` | latency model 摘要 |

`ExpertSchedulingStatsRecorder.summary()` 会按 `(strategy, phase, layer)` 聚合调用次数、reason 分布、CPU/GPU/preload 数量和 unique expert 集合。

---

# 10. 当前实现中的关键注意点

## 10.1 prefill 与 decode 的分界

在 `deepseek.py` 中：

```python
strategy.decide_and_prepare(
    ...,
    future_demands=self.predicted_next_demands,
    placement=self.placeholder_manager.snapshot(),
    is_prefill=is_prefill,
)
```

`is_prefill` 决定 `PrefetchHybridStrategy` 构造的 `ExpertLayerRequest.phase`。

## 10.2 future demand 来源

当前 forward 中会预测下一层专家需求：

```python
predict_future = pred_executor.submit(
    self.expert_predictor.predict, inps, self.model, i_layer, 1
)
self.predicted_next_demands = pred_result or []
```

`PrefetchHybridStrategy` 优先使用传入的 `future_demands`；如果没有，则可通过 `predicted_next_experts/predicted_next_weights` 调用 `build_future_demands()`。

## 10.3 preload 只是调度计划

`PrefetchHybridStrategy.decide_and_prepare()` 返回：

```python
return schedule.cpu_expert_ids, schedule.gpu_expert_ids, schedule.preload_expert_ids, raw_assignments
```

真正的加载由后续 `ExpertExecutionManager.execute(schedule, context)` 处理。调度器只决定 preload 哪些专家，不直接搬运权重。

## 10.4 latency 单位

当前注释说明 benchmark 表中是 `avg_time_ms`，`t_io` 来自 `expert_weight_copy.avg_ms`。

相关代码：

```python
# Latency model values are milliseconds: benchmark tables store avg_time_ms
# and t_io is loaded from expert_weight_copy.avg_ms.
```

因此文档中的 `t_io/t_cpu/t_gpu` 均应按毫秒级 latency model 理解。

## 10.5 当前算法未使用 evict

`ExpertSchedule` 包含 `evict` 字段，但当前三个策略返回的 schedule 中 `evict=[]`。placeholder 的释放、替换和加载细节由 placeholder manager / expert executor 负责。

---

# 11. 一句话总结

- `GPUOnlyStrategy`：不区分阶段，当前活跃专家全部走 GPU。
- `FiddlerStrategy`：不显式区分阶段，对每个当前专家比较 CPU 成本和 GPU 搬运+执行成本。
- `PrefetchHybridStrategy`：通过 `PDScopeScheduler` 显式区分 prefill/decode；prefill 使用全局队列和 I/O 气泡选择 ondemand/preload，decode 使用 `n_g_rho` 和当前/下一层驻留充足性进入 fallback、mode-a、mode-b、mode-c。
