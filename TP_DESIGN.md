# Tensor Parallel 实现设计文档

## 架构概览

基于 vLLM 的设计理念，将 TP 实现分为 4 层：

```
┌─────────────────────────────────────────────────────────────┐
│ L3: 引擎层 (llm_tp.py)                                       │
│  - Rank 0 驱动调度（接收请求、采样、返回结果）                │
│  - 广播 input_ids 到所有 rank                                │
│  - 所有 rank 同步执行推理                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ L2: 模型层 (model_tp.py)                                     │
│  - AttentionTP: QKV 按 head 切分                            │
│  - MLPTP: gate/up 列并行，down 行并行                        │
│  - 权重加载时自动切分                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ L1: 并行层 (parallel_layers.py)                             │
│  - ColumnParallelLinear: 输出维度切分                        │
│  - RowParallelLinear: 输入维度切分 + all_reduce              │
│  - QKVParallelLinear: 按 head 切分（支持 GQA）               │
│  - VocabParallelEmbedding: 词表切分                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ L0: 分布式运行时 (distributed.py)                            │
│  - 初始化 TP group (torch.distributed)                       │
│  - 封装通信原语 (all_reduce, all_gather, broadcast)          │
│  - Rank 管理和设备映射                                        │
└─────────────────────────────────────────────────────────────┘
```

## 核心设计决策

### 1. TP 切分策略

**Attention:**
- QKV: 按 attention head 切分（每个 rank 有 `num_heads // tp_size` 个完整 head）
- O projection: RowParallel（输入已切分，输出 all_reduce）
- KV cache: 每个 rank 只存储自己的 heads

**MLP:**
- gate_proj, up_proj: ColumnParallel（输出 `intermediate_size // tp_size`）
- down_proj: RowParallel（输入已切分，输出 all_reduce）

**Embedding & LM Head:**
- Embedding: 复制到所有 rank（简化实现）
- LM head: 只在 rank 0 计算（节省通信）

### 2. 通信优化

**最小化通信次数:**
- 每层只有 2 次通信：
  - Attention O proj: all_reduce
  - MLP down proj: all_reduce
- Prefill 阶段无额外通信（KV cache 本地存储）

**广播策略:**
- input_ids: rank 0 → 所有 rank（每个 decode step）
- 采样结果: 只在 rank 0 计算，无需广播回去

### 3. 权重加载

**方案：各 rank 独立加载 + 本地切分**
- 优点：实现简单，无需 rank 间通信
- 缺点：所有 rank 都读取完整权重文件
- 适用场景：单节点多卡（共享文件系统）

**切分逻辑:**
```python
# QKV 切分（按 head）
q_per_rank = num_q_heads * head_dim // tp_size
k_per_rank = num_kv_heads * head_dim // tp_size

# MLP 切分（按维度）
gate_weight_split = split_tensor_along_dim(gate_weight, tp_size, dim=0)
down_weight_split = split_tensor_along_dim(down_weight, tp_size, dim=1)
```

## 使用方式

### 单卡运行（baseline）
```bash
python test_tp.py
```

### 多卡 TP
```bash
# 2 卡
torchrun --nproc_per_node=2 test_tp.py

# 4 卡
torchrun --nproc_per_node=4 test_tp.py
```

### 代码示例
```python
from llm_tp import LLMTP

# 初始化（自动检测分布式环境）
llm = LLMTP(model_path, max_num_seqs=16, enable_tp=True)

# 只在 rank 0 调用
if llm.is_rank_0:
    future = llm.generate(["Hello world"], max_tokens=50)
    results = future.result()
    print(results[0].text)
```

## 性能预期

**理论加速比:**
- 2 卡: ~1.8x（通信开销 ~10%）
- 4 卡: ~3.2x（通信开销 ~20%）
- 8 卡: ~5.5x（通信开销 ~30%）

**内存节省:**
- 模型权重: 1/tp_size
- KV cache: 1/tp_size
- 激活值: 部分切分（约 0.6/tp_size）

## 扩展性

### 支持 Pipeline Parallel (PP)
- 在 L0 添加 PP group 管理
- 在 L3 添加 stage 间通信

### 支持 Data Parallel (DP)
- 在 L0 添加 DP group
- 在 L3 添加梯度同步（训练场景）

### 支持 Expert Parallel (EP)
- 在 L1 添加 ExpertParallelLinear
- 在 L2 添加 MoE 层支持

## 文件清单

```
model/
├── distributed.py          # L0: 分布式运行时
├── parallel_layers.py      # L1: 并行层
├── model_tp.py            # L2: TP 模型
├── model.py               # 原始单卡模型
└── tokenizer.py

llm_tp.py                  # L3: TP 引擎
llm.py                     # 原始单卡引擎
test_tp.py                 # TP 测试脚本
```

## 与 vLLM 的对比

| 特性 | vLLM | 本实现 |
|------|------|--------|
| TP 切分策略 | 按 head 切分 | ✓ 相同 |
| 通信优化 | 自定义 CUDA kernel | 使用 PyTorch 原生 |
| 权重加载 | 分片加载 | 各 rank 独立加载 |
| KV cache | PagedAttention | Slot-based |
| 调度策略 | C++ 调度器 | Python 调度器 |
| 生产就绪 | ✓ | 原型实现 |

## 已知限制

1. **Worker 同步**: 当前 worker ranks 使用简化的等待机制，生产环境需要实现完整的控制流同步
2. **错误处理**: 缺少分布式错误恢复机制
3. **动态批处理**: Prefill 和 decode 的 TP 同步需要更精细的控制
4. **通信优化**: 可以使用 NCCL 的 group 通信进一步优化

## 下一步优化

1. **实现完整的 worker 同步机制**（使用 broadcast 控制信号）
2. **支持 FP8 量化 + TP**
3. **优化小 batch 场景的通信开销**（通信与计算 overlap）
4. **支持跨节点 TP**（需要处理网络延迟）
