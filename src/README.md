# DeepSeek MoE Expert Offloading 推理参数说明

## 基本用法

```bash
PYTHONPATH=src python scripts/infer_deepseek.py \
    --model /path/to/deepseek-v2-lite-chat \
    [选项...]
```

也可使用 `run_deepseek.sh` 一键运行。

---

## 参数列表

### 模型与输入

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | *(必填)* | DeepSeek 模型路径（HuggingFace 格式） |
| `--input` | str | `"Please tell me a joke."` | 输入文本。当指定 `--dataset` 时被忽略 |
| `--dataset` | str | `None` | ShareGPT 格式 JSON 数据集路径。设置后从数据集中随机采样 `batch_size` 条 prompt，覆盖 `--input` |
| `--batch-size` | int | `1` | 推理批次大小 |
| `--beam-width` | int | `1` | Beam search 宽度 |
| `--input-token-num` | int | `None` | tokenizer 截断最大输入 token 数。不设置则不截断 |
| `--output-token-num` | int | `20` | 生成的 token 数量 |

### 调度策略

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cpu-offload` | int | `1` | 专家调度策略选择：<br>`0` — GPUOnlyStrategy：所有活跃专家走 GPU 执行（baseline）<br>`1` — PrefetchHybridStrategy：PDScope 策略，区分 prefill/decode 阶段，支持预取<br>`2` — FiddlerStrategy：逐专家 CPU/GPU 代价最优选择 |

### Warmup 与缓存

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--warmup` | int | `0` | 预热次数。每次预热执行完整的 generate 流程 |
| `--preserve-warmup-cache` | flag | `False` | 保留预热阶段填充的动态 placeholder 缓存。默认情况下预热后会清空所有动态 placeholder |

### 计时与 Profiling

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sync-timing` | flag | `False` | 在 prefill/decode 计时前后插入 `torch.cuda.synchronize()`，使计时更精确，但会增加额外同步开销 |
| `--debug-runtime-state` | flag | `False` | 在预热前后和测量结束后打印 placeholder/preload 计数器状态 |
| `--profile-expert-executor` | flag | `False` | 打印每层 GPU/CPU/preload 执行耗时统计。会引入 CUDA 同步开销，影响推理速度 |
| `--profile-torch` | flag | `False` | 启用 `torch.profiler`，记录 CPU/CUDA 活动，导出 Chrome trace 文件 |
| `--profile-torch-dir` | str | `./logs` | torch.profiler 输出目录 |
| `--profile-decode-steps` | int | `2` | prefill 结束后再 profile 的 decode 步数（仅在 `--profile-torch` 开启时生效） |

### 调度决策记录（JSONL）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record-expert-schedule` | flag | `False` | 开启后，每次调度决策记录一层结构化 JSON 数据（策略、层号、阶段、GPU/CPU/preload 专家列表、placement 快照、延迟参数等） |
| `--expert-schedule-log` | str | `None` | JSONL 输出文件路径（如 `logs/schedule.jsonl`）。不设置则记录仅保存在内存中，程序结束时打印摘要 |

**JSONL 记录字段**：
- `call_index` / `timestamp` — 调用序号与时间戳
- `strategy` / `scheduler` / `reason` — 策略类名、调度器类名、调度原因（如 `decode-mode-c`）
- `layer` / `phase` / `n_expert` — 层号、阶段（prefill/decode）、总专家数
- `active_experts` / `gpu_experts` / `cpu_experts` / `preload_experts` — 各类专家的详细需求（expert_id、token_count、score）
- `current_demands` / `future_demands` — 当前层和预测的未来层需求
- `placement` — GPU 驻留、placeholder 驻留、loading、CPU 驻留等计数和 ID 列表
- `latency` — 延迟模型参数（t_io、CPU/GPU 延迟查找表）
- `strategy_params` — 调度器参数（alpha、t_attn、r_hit）
- `counters` — 累计 hit/all 计数器

### 专家选择记录

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record-hot-experts` | flag | `False` | 记录每次 MoE 路由选择的 (layer, expert) 对，用于离线热点专家分析 |

---

## 输出说明

### 标准输出

程序结束后自动打印以下指标：

```
prefill_time: 0.1234, decode_time: 0.5678, hit_rate: 0.6616
tokens per second (decode): 176.06
```

- `hit_rate`：合并的 GPU 驻留命中率（static + placeholder），来自调度器侧计数

### Hit Source 分析（自动打印）

```
[hit-source] static_gpu_hit=1200 (tokens=2400)
[hit-source] placeholder_hit=80 (tokens=160)
[hit-source] ondemand_load=20 (tokens=40)
[hit-source] preload_hit=45 (of placeholder hits)
[hit-source] total_gpu_experts=1300 (tokens=2600)
[hit-source] placeholder_hit_rate=0.0615 (placeholder / total_gpu)
[hit-source] preload_hit/placeholder=0.5625
[hit-source] preload_hit/preload_success=0.4500
[hit-source] preload_hit/preload_request=0.3750
```

- `static_gpu_hit`：GPU 上永久驻留的专家被实际执行的次数
- `placeholder_hit`：已存在于 placeholder 中的专家被实际执行的次数
- `ondemand_load`：需要临时加载权重的专家次数
- `preload_hit`：placeholder 命中中，来自预取（preload）流的次数
- `preload_hit/placeholder`：placeholder 命中中有多少比例来自预取
- `preload_hit/preload_success`：成功预取的专家中有多少被实际使用
- `preload_hit/preload_request`：预取请求中有多少最终被使用

---

## 离线分析工具

### analyze_schedule.py

```bash
PYTHONPATH=src python scripts/analyze_schedule.py logs/schedule.jsonl [选项]
```

| 选项 | 说明 |
|------|------|
| `--summary` | 打印按调度模式汇总的统计表（次数、次优、平均 slack、GPU/CPU/preload 总数） |
| `--suboptimal` | 只显示次优决策（actual_wall > optimal_wall） |
| `--preload-overlap` | 分析预加载专家与下一层实际需求的重叠率（precision、coverage、waste） |
| `--call-index N` | 查看指定 call_index 的详细决策追踪 |
| `--layer N` | 按层号过滤 |
| `--reason REASON` | 按调度模式过滤（如 `decode-mode-c`） |
| `--phase PHASE` | 按阶段过滤（prefill/decode） |
| `--limit N` | 最多显示 N 条记录 |

### analyze_hot_experts.py

```bash
PYTHONPATH=src python scripts/analyze_hot_experts.py [选项]
```

用于统计热点专家并生成 `hot/deep.txt` 文件，供后续推理使用。
