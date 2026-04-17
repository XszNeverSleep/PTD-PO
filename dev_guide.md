# verl Training Framework - Developer Guide

> 本文档覆盖 `/mnt/vlm-ks3/xiangshizhe/EasyR1/verl` 目录下的核心训练逻辑，帮助开发者快速理解框架的设计与实现。

---

## 目录

1. [整体架构](#1-整体架构)
2. [目录结构](#2-目录结构)
3. [数据流与训练循环](#3-数据流与训练循环)
4. [核心组件详解](#4-核心组件详解)
   - [4.1 DataProto 数据协议](#41-dataproto-数据协议)
   - [4.2 RayPPOTrainer 训练器](#42-rayppotrainer-训练器)
   - [4.3 Actor (Policy Model)](#43-actor-policy-model)
   - [4.4 Critic (Value Model)](#44-critic-value-model)
   - [4.5 vLLM Rollout 引擎](#45-vllm-rollout-引擎)
   - [4.6 Reward Manager](#46-reward-manager)
   - [4.7 Core Algorithms (优势估计与损失函数)](#47-core-algorithms-优势估计与损失函数)
   - [4.8 PID-GRPO (Policy-Informed Distillation)](#48-pid-grpo-policy-informed-distillation)
5. [分布式架构](#5-分布式架构)
   - [5.1 Single Controller + Ray WorkerGroup](#51-single-controller--ray-workergroup)
   - [5.2 Hybrid Engine (FSDP + vLLM)](#52-hybrid-engine-fsdp--vllm)
   - [5.3 Ulysses 序列并行](#53-ulysses-序列并行)
6. [数据集与 DataLoader](#6-数据集与-dataloader)
7. [配置系统](#7-配置系统)
8. [Checkpoint 系统](#8-checkpoint-系统)
9. [Metrics 与日志](#9-metrics-与日志)

---

## 1. 整体架构

verl 是一个基于 Ray 的分布式 RLHF 训练框架，采用 **Single Controller** 模式：一个中心 Driver 进程协调多个分布式 Worker，完成 PPO/GRPO 等强化学习训练。

核心设计特点：
- **Hybrid Engine**: 同一组 GPU 上交替运行 FSDP（训练）和 vLLM（推理），通过 sleep/wake_up 机制切换，最大化 GPU 利用率
- **DataProto 协议**: 所有组件之间通过统一的 `DataProto` 数据结构通信
- **可插拔算法**: 优势估计器（GAE/GRPO/RLOO/REINFORCE++/REMAX）通过装饰器注册，策略损失支持多种变体
- **多模态支持**: 原生支持 VLM（Qwen2-VL/Qwen2.5-VL/Qwen3-VL）的图像和视频输入

```
┌───────────────────────────────────────────────────────────────────────┐
│                        RayPPOTrainer (Driver)                        │
│            协调所有 Worker，执行训练循环，记录 metrics                  │
└────────┬──────────────────────┬──────────────────────┬────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐  ┌─────────────────┐   ┌──────────────────┐
│ ActorRolloutRef  │  │     Critic      │   │  Reward Manager  │
│  WorkerGroup     │  │   WorkerGroup   │   │   (Ray Actor)    │
│                  │  │                 │   │                  │
│  ┌─ Actor(FSDP) │  │ ┌─ Critic(FSDP) │   │  Rule-based      │
│  ├─ Ref  (FSDP) │  │ └───────────────│   │  Reward Fn       │
│  └─ Rollout     │  │                 │   │                  │
│     (vLLM)      │  │                 │   │                  │
└─────────────────┘  └─────────────────┘   └──────────────────┘
```

---

## 2. 目录结构

```
verl/
├── __init__.py
├── protocol.py                   # DataProto 数据交换协议
│
├── trainer/
│   ├── main.py                   # CLI 入口：配置加载 → Ray 初始化 → 创建 Trainer → fit()
│   ├── config.py                 # PPOConfig (DataConfig, AlgorithmConfig, TrainerConfig)
│   ├── ray_trainer.py            # RayPPOTrainer：训练主循环
│   ├── core_algos.py             # RL 算法：优势估计、策略损失、值函数损失、KL 计算
│   ├── metrics.py                # 训练 metrics 计算
│   └── data_loader.py            # DataLoader 创建逻辑
│
├── workers/
│   ├── config.py                 # WorkerConfig (ActorConfig, CriticConfig, RolloutConfig, ...)
│   ├── fsdp_workers.py           # FSDPWorker：统一的 FSDP Worker 实现（actor/critic/ref/rollout）
│   ├── actor/
│   │   ├── config.py             # ActorConfig, ModelConfig, OptimConfig, FSDPConfig
│   │   └── dp_actor.py           # DataParallelPPOActor：log prob 计算 + 策略梯度更新
│   ├── critic/
│   │   ├── config.py             # CriticConfig
│   │   └── dp_critic.py          # DataParallelPPOCritic：值估计 + 值函数更新
│   ├── rollout/
│   │   ├── config.py             # RolloutConfig
│   │   └── vllm_rollout_spmd.py  # vLLMRollout：基于 vLLM 的序列生成
│   ├── reward/
│   │   ├── config.py             # RewardConfig
│   │   └── function.py           # AutoRewardManager：奖励计算
│   └── sharding_manager/
│       └── fsdp_vllm.py          # FSDPVLLMShardingManager：FSDP ↔ vLLM 权重同步
│
├── models/
│   ├── monkey_patch.py           # Ulysses 序列并行 monkey patch
│   └── transformers/             # VLM 模型适配（flash attention, rope index 等）
│
├── utils/
│   ├── dataset.py                # RLHFDataset：数据集处理
│   ├── tokenizer.py              # Tokenizer 工具
│   ├── fsdp_utils.py             # FSDP 工具函数（wrap policy, offload/load）
│   ├── checkpoint/               # Checkpoint 管理
│   │   ├── checkpoint_manager.py # 基类 + checkpoint 追踪
│   │   └── fsdp_checkpoint_manager.py  # FSDP checkpoint 存取
│   └── logger/                   # 日志系统（wandb, tensorboard, mlflow 等）
│
└── single_controller/            # Ray 分布式调度
    ├── base/
    │   ├── worker.py             # Worker 基类
    │   ├── worker_group.py       # WorkerGroup 基类
    │   └── decorator.py          # @register dispatch 装饰器
    └── ray/
        └── base.py               # RayWorkerGroup：Ray 实现
```

---

## 3. 数据流与训练循环

每个训练 step 的完整数据流：

```
Step 1: 数据加载
  train_dataloader → prompts (input_ids, attention_mask, position_ids, ground_truth, multi_modal_data)

Step 2: 序列生成 (vLLM Rollout)
  prompts → vLLM generate → responses
  组装完整序列 (prompt + response)

Step 3: 奖励计算
  responses + ground_truth → RewardManager → token_level_scores
  (奖励放在 response 最后一个 token 位置)

Step 4: Log Prob 计算
  完整序列 → Actor forward → old_log_probs     [bs, response_length]

Step 5: Reference Log Prob 计算
  完整序列 → Ref forward → ref_log_probs       [bs, response_length]

Step 6: 值估计 (仅 GAE)
  完整序列 → Critic forward → values            [bs, response_length]

Step 7: PID Mask 计算 (enable_pid=true 时)
  accuracy_scores + uid → compute_pid_mask(threshold) → pid_mask, pid_group_mask
  hint_input_ids + responses → Actor eval forward → teacher_topk_log_probs/ids (或 teacher_log_probs)

Step 8: KL 惩罚 (可选)
  token_level_scores += -kl_coef * KL(old_log_probs, ref_log_probs)
  PID 组 (pid_group_mask=True) 的 token_level_rewards 不受 ref KL 影响

Step 9: 优势计算
  token_level_rewards + values → AdvantageEstimator → advantages, returns

Step 10: Critic 更新 (仅 GAE)
  values + returns → value_loss → optimizer.step()

Step 11: Actor 更新
  old_log_probs + advantages → policy_loss
  + kl_coef * kl_loss (PID 组清零)
  + pid_coef * pid_loss (仅 pid_mask=True 的样本)
  → optimizer.step()
```

### 训练主循环伪代码 (`ray_trainer.py: RayPPOTrainer.fit()`)

```python
for epoch in range(total_epochs):
    for batch_idx in range(steps_per_epoch):
        # === 生成阶段 ===
        actor_rollout_ref_wg.prepare_rollout_engine()   # 加载 vLLM，同步权重
        batch = _make_batch_data()                       # 生成序列 + 计算奖励
        actor_rollout_ref_wg.release_rollout_engine()   # 卸载 vLLM

        # === 经验收集 ===
        _balance_batch(batch)                            # 跨 DP rank 均衡序列长度
        old_log_probs = actor_rollout_ref_wg.compute_log_probs(batch)
        ref_log_probs = actor_rollout_ref_wg.compute_ref_log_probs(batch)
        values = critic_wg.compute_values(batch)         # 仅 GAE

        # === PID (enable_pid=true 时) ===
        pid_mask, pid_group_mask = compute_pid_mask(accuracy, uid, threshold)
        if pid_mask.any():
            teacher_output = actor_rollout_ref_wg.compute_teacher_log_probs(batch)
            batch.union(teacher_output)

        # === 奖励处理 ===
        apply_kl_penalty(batch)                          # 可选 KL 惩罚
        # PID 组的 token_level_rewards 回退到原始 scores（不叠加 ref KL）
        compute_advantage(batch, adv_estimator=...)      # 优势估计

        # === 模型更新 ===
        critic_wg.update_critic(batch)                   # 仅 GAE
        actor_rollout_ref_wg.update_actor(batch)         # 策略梯度 + KL loss + PID loss

        # === 验证与保存 ===
        if global_step % val_freq == 0: _validate()
        if global_step % save_freq == 0: _save_checkpoint()
```

---

## 4. 核心组件详解

### 4.1 DataProto 数据协议

> `verl/protocol.py`

`DataProto` 是所有组件之间数据传递的标准格式：

```python
@dataclass
class DataProto:
    batch: TensorDict          # 张量数据（input_ids, attention_mask, log_probs, ...）
    non_tensor_batch: dict     # 非张量数据（字符串, metadata 等），值为 numpy array
    meta_info: dict            # 元信息（eos_token_id, temperature 等配置）
```

核心操作：

| 方法 | 说明 |
|------|------|
| `DataProto.from_single_dict(d)` | 从 flat dict 创建，自动分离 tensor/non-tensor |
| `pop(batch_keys, non_tensor_batch_keys)` | 提取并移除指定 key |
| `union(other)` | 合并两个 DataProto |
| `concat([dp1, dp2, ...])` | 沿 batch 维度拼接 |
| `repeat(n)` | 沿 batch 维度重复 n 次（用于 group sampling） |
| `chunk(n)` | 拆分为 n 个 chunk |
| `to(device)` | 移动张量到指定设备 |
| `select(keys)` / `index_select(indices)` | 选取子集 |

### 4.2 RayPPOTrainer 训练器

> `verl/trainer/ray_trainer.py`

训练器负责协调所有 Worker 完成 RL 训练循环。

**角色定义：**

```python
class Role(IntEnum):
    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()           # Actor + Rollout 合并
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()        # Actor + Rollout + Ref 合并 (最常用)
```

**关键方法：**

| 方法 | 说明 |
|------|------|
| `init_workers()` | 创建 Ray WorkerGroup，初始化 FSDP 模型 |
| `fit()` | 训练主循环 |
| `_make_batch_data()` | 生成序列：获取 batch → vLLM 生成 → 重复 n 次 → 在线过滤 |
| `_balance_batch()` | 跨 DP rank 均衡序列长度，减少 padding 浪费 |
| `_validate()` | 验证循环：生成 + 计算奖励 + 记录日志 |
| `_save_checkpoint()` | 保存 checkpoint + 跟踪文件 |

**在线过滤 (`online_filtering`)：** 当启用时，重复生成直到获得足够满足奖励条件的样本（奖励在 `[filter_low, filter_high]` 范围内）。

### 4.3 Actor (Policy Model)

> `verl/workers/actor/dp_actor.py` — `DataParallelPPOActor`

**Log Prob 计算 (`compute_log_prob`)：**

1. 将 batch 拆分为 micro-batch（按 `micro_batch_size_per_device_for_experience`）
2. 支持动态 batching（按 `max_token_len` 自动调整 batch 大小）
3. 每个 micro-batch 执行 `_forward_micro_batch()`

**Forward 逻辑（padding-free 模式）：**

```
input_ids [B, seq_len]
    → unpad_input → input_ids_rmpad [total_nnz, 1]
    → ulysses_pad_and_slice → 切分到 SP ranks
    → model forward → logits_rmpad
    → log_probs_from_logits(logits, labels_shifted)
    → gather_outputs_and_unpad → 还原完整序列
    → pad_input → full_log_probs [B, seq_len]
    → 截取 response 部分 → log_probs [B, response_length]
```

**策略梯度更新 (`update_policy`)：**

```
for mini_batch in split(data, global_batch_size):
    for epoch in range(ppo_epochs):
        total_response_tokens = all_reduce(sum(response_mask))
        for micro_batch in split(mini_batch, micro_batch_size):
            log_probs = _forward_micro_batch(micro_batch)
            pg_loss = compute_policy_loss(old_log_probs, log_probs, advantages, ...)
            loss = pg_loss * sum(response_mask) * world_size / total_response_tokens
            loss.backward()                    # 梯度累积
        grad_norm = clip_grad_norm_(max_grad_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
        optimizer.zero_grad()
```

- 梯度通过 `world_size / total_response_tokens` 缩放实现正确的跨 GPU 累积
- 可选添加 KL loss（`use_kl_loss=True` 时作为额外损失项）

### 4.4 Critic (Value Model)

> `verl/workers/critic/dp_critic.py` — `DataParallelPPOCritic`

结构与 Actor 对称，区别在于：
- Forward 输出 value 而非 log_prob
- 更新使用 `compute_value_loss`（clipped value loss）
- 应用 `response_mask` 将非 action token 的 value 清零

### 4.5 vLLM Rollout 引擎

> `verl/workers/rollout/vllm_rollout_spmd.py` — `vLLMRollout`

**初始化：**
- 创建 vLLM `LLM` 实例，支持 tensor parallelism
- 启用 `enable_sleep_mode=True`，不推理时可以卸载显存
- 初始化后立即 `sleep(level=1)` 释放显存

**序列生成 (`generate_sequences`)：**

```
Input:  DataProto (input_ids, attention_mask, position_ids, multi_modal_data)
    → 构建 vllm_inputs (prompt_token_ids + multi_modal_data)
    → 处理多模态数据 (images → process_image, videos → process_video)
    → inference_engine.generate(vllm_inputs, sampling_params)
    → 提取 response_ids, 填充到 max_response_length
    → 拼接 prompt + response 的 input_ids, attention_mask, position_ids
    → 创建 response_mask (基于 EOS token 位置)
Output: DataProto (完整序列 + response_mask)
```

**Position IDs 计算：**
```python
# Response 的 position 从 prompt 最后位置继续递增
# [B, prompt_len] + [B, response_len]
delta_position_id = torch.arange(1, response_length + 1)
response_position_ids = position_ids[..., -1:] + delta_position_id
position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
```

**Attention Mask：**
```
prompt 部分:   [0,0,0,1,1,1,1,1]  (左 padding)
response 部分: [1,1,1,1,0,0,0,0]  (右 padding，到 EOS 为止)
```

### 4.6 Reward Manager

> `verl/workers/reward/function.py` — `AutoRewardManager`

**奖励函数接口：**

```python
# 输入
RewardInput = {"response": str, "response_length": int, "ground_truth": str}

# 输出
RewardScore = {"overall": float, "format": Optional[float], "accuracy": Optional[float]}
```

**处理模式：**
- **Sequential**: 逐样本调用 reward_fn，适用于复杂逻辑
- **Batch**: 批量调用 reward_fn，适用于简单规则

**奖励放置：** 奖励值放在 response 最后一个有效 token 位置：
```python
reward_tensor[i, response_length - 1] = score["overall"]
```

**动态加载：** 通过 `importlib` 从配置路径动态加载自定义奖励函数模块。

### 4.7 Core Algorithms (优势估计与损失函数)

> `verl/trainer/core_algos.py`

#### 优势估计器

所有估计器通过 `@register_adv_estimator` 注册，统一接口返回 `(advantages, returns)`。

| 估计器 | 是否需要 Critic | 是否需要 Group (n>1) | 核心思想 |
|--------|:---:|:---:|------|
| **GAE** | ✅ | ❌ | TD-error 反向传播，δ_t = r_t + γV(s_{t+1}) - V(s_t) |
| **GRPO** | ❌ | ✅ | 组内归一化：(score - mean) / (std + ε) |
| **GRPO_PASSK** | ❌ | ✅ | 仅最优 response 获得优势：(r_max - r_2nd) / std |
| **RLOO** | ❌ | ✅ | Leave-one-out 基线：score - mean(其他 responses) |
| **REINFORCE++** | ❌ | ❌ | 折扣回报 + whitening：R_t = r_t + γ·R_{t+1} |
| **REMAX** | ❌ | ❌ | 基线来自 greedy (temperature=0) 生成的奖励 |

**GRPO 实现细节：**
```python
# 按 uid 分组
for group_indices in group_by_uid(index):
    scores = token_level_rewards[group_indices].sum(-1)  # [group_size]
    mean, std = scores.mean(), scores.std()
    advantages[group_indices] = ((scores - mean) / (std + eps)).unsqueeze(-1) * response_mask
```

#### 策略损失函数 (`compute_policy_loss`)

```python
# PPO Clipped Objective
ratio = exp(log_probs - old_log_probs)                        # 重要性采样比
pg_loss1 = -advantages * ratio
pg_loss2 = -advantages * clamp(ratio, 1-clip_low, 1+clip_high)
pg_loss  = max(pg_loss1, pg_loss2)                             # 悲观裁剪

# Dual Clip (可选): 当 advantage < 0 时额外裁剪
if clip_ratio_dual > 0:
    pg_loss = max(pg_loss, clip_ratio_dual * advantages)
```

**损失变体：**
- `default`: 标准 PPO clipped loss
- `gspo`: Group-relative policy optimization
- `cispo`: CISPO loss
- `sapo`: SAPO loss（token 级别的 positive/negative 温度控制）

**损失平均模式：**
- `token`: 按 token 数平均（除以 response_mask.sum()）
- `seq`: 按序列数平均（除以 batch_size）

#### KL 散度计算 (`compute_kl`)

| 类型 | 公式 |
|------|------|
| `kl` | `log_probs - ref_log_probs` |
| `abs` | `\|log_probs - ref_log_probs\|` |
| `mse` | `0.5 * (log_probs - ref_log_probs)²` |
| `low_var_kl` | `(ref_log_probs - log_probs) + ratio - 1` (数值稳定) |
| `full` | `ref_prob * (ref_log_probs - log_probs)` (完整 KL 散度) |

#### 值函数损失 (`compute_value_loss`)

```python
# Clipped Value Loss
vpred_clipped = values + clamp(vpreds - values, -cliprange, +cliprange)
vf_loss1 = (vpreds - returns)²
vf_loss2 = (vpred_clipped - returns)²
vf_loss  = 0.5 * max(vf_loss1, vf_loss2)
```

### 4.8 PID-GRPO (Policy-Informed Distillation)

> `verl/trainer/core_algos.py` + `verl/workers/actor/dp_actor.py` + `verl/trainer/ray_trainer.py`

PID-GRPO 在标准 GRPO 基础上增加了 **在线策略蒸馏**：对于模型做不对的题，用 hint-augmented prompt 喂给同一个 actor（作为 teacher），将 teacher 的输出分布蒸馏回 student，从而在 RL 训练中引入有监督的知识注入。

#### 两种模式

| 参数 | `pid_top_k=100` (Top-K 模式) | `pid_top_k=0` (Single-token 模式) |
|------|------|------|
| Teacher 输出 | top-100 的 log_probs + token IDs `[B, T, K]` | 逐 token 的 log_prob `[B, T]` |
| Student 输出 | 额外计算 top-100 log_probs/probs/IDs | 仅使用已有的 log_probs |
| KL 计算方式 | `compute_topk_kl`：独立 top-K 集合 + tail 补偿 | `compute_kl`：标准逐 token KL |
| 显存开销 | 较高（需存储 `[B, T, K]` 张量） | 较低 |
| 精度 | 更高（近似完整分布 KL） | 较低（仅用单 token log_prob 近似） |

#### 关键配置

```yaml
algorithm:
  enable_pid: true            # 开启 PID
  pid_threshold: 0.2          # 组内正确率 < 20% 时激活 PID
  pid_top_k: 100              # Top-K token 数 (0 = single-token 模式)
  pid_coef: 5.0e-2            # PID 蒸馏损失系数
  pid_kl_direction: jsd_kl    # KL 方向：forward_kl / reverse_kl / jsd_kl
  pid_all_trajectories: false # true = 对 PID 组内所有轨迹蒸馏（含正确的）; false = 仅错误轨迹

  # Standard ref KL (与 PID 共存但互斥作用于不同 sample)
  use_kl_loss: true
  kl_penalty: low_var_kl
  kl_coef: 1.0e-2
  kl_direction: forward_kl
```

#### 数据要求

数据集需包含 `prompt_with_hint` 列（通过 `data.prompt_with_hint_key` 指定），即对原始 prompt 附加了 hint 信息的增强版本。DataLoader 会同时生成：
- 标准 prompt 的 `input_ids/attention_mask/position_ids`
- Hint prompt 的 `hint_input_ids/hint_attention_mask/hint_position_ids`

#### 完整执行流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ray_trainer.py — 训练主循环 fit()                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Rollout + Reward (与标准 GRPO 相同)                            │
│    vLLM 生成 response → RewardManager 计算 accuracy                     │
│                                                                         │
│  Step 2: 计算 PID Mask (core_algos.compute_pid_mask)                    │
│    对每个 group (uid)：                                                  │
│    ├─ group_accuracy = 组内 accuracy==1.0 的比例                         │
│    ├─ 若 group_accuracy < pid_threshold (0.2):                          │
│    │   ├─ pid_group_mask[所有组内样本] = True                            │
│    │   └─ pid_mask[组内 accuracy≠1.0 的样本] = True                     │
│    └─ 额外 AND has_hint：过滤无 hint 数据的样本                          │
│                                                                         │
│  Step 3: 计算 Teacher 输出 (dp_actor.compute_teacher_log_prob)          │
│    ├─ 输入: hint_input_ids + responses (hint prompt 拼接 response)       │
│    ├─ Actor 模型 eval 模式 forward                                      │
│    ├─ pid_top_k=100: 返回 teacher_topk_log_probs [B, T, 100]           │
│    │                       teacher_topk_ids       [B, T, 100]           │
│    └─ pid_top_k=0:  返回 teacher_log_probs       [B, T]                │
│    注: 所有 sample 都过 forward (避免 FSDP all-gather 死锁),             │
│        non-PID 位置结果 * 0 清零                                        │
│                                                                         │
│  Step 4: KL Penalty 互斥处理                                           │
│    ├─ 对 pid_group_mask=True 的样本：                                    │
│    │   token_level_rewards = token_level_scores (跳过 ref KL penalty)   │
│    └─ 对其他样本：正常施加 ref KL penalty                                │
│                                                                         │
│  Step 5: 优势计算 (与标准 GRPO 相同)                                    │
│    compute_advantage → GRPO 组内归一化                                   │
│                                                                         │
│  Step 6: Actor 更新 (dp_actor.update_policy)                            │
│    对每个 micro-batch:                                                   │
│    ├─ Student forward:                                                  │
│    │   ├─ pid_top_k=100: 额外返回 student top-K (log_probs/probs/ids)  │
│    │   └─ pid_top_k=0:   仅返回 log_probs                              │
│    ├─ pg_loss = compute_policy_loss(...)  # 标准 PPO clip loss          │
│    ├─ kl_loss (use_kl_loss=true):                                       │
│    │   ├─ kld = compute_kl(actor, ref, low_var_kl, forward_kl)          │
│    │   └─ kld[pid_group_mask] = 0  # PID 组不受 ref KL 约束            │
│    ├─ pid_loss (_compute_pid_loss):                                     │
│    │   ├─ 仅对 pid_mask=True 的样本计算                                  │
│    │   ├─ pid_top_k=100: compute_topk_kl(student_topK, teacher_topK)   │
│    │   └─ pid_top_k=0:   compute_kl(student, teacher)                  │
│    ├─ loss = pg_loss + kl_coef * kl_loss + pid_coef * pid_loss          │
│    └─ loss.backward() + optimizer.step()                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Top-K 模式实现细节 (`pid_top_k > 0`)

当 `pid_top_k > 0`（如 100）时，PID 不使用逐 token 的单一 log_prob，而是比较 student 和 teacher 的 top-K 分布。
这种方式更接近完整分布的 KL，但需要额外处理显存和跨集合匹配问题。

**端到端数据流：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Teacher 端 (compute_teacher_log_prob → _forward_teacher)            │
│     hint_prompt + response → model forward → logits [P, T, V]          │
│     → torch.topk(logits, K) → topk_ids [P, T, K]                      │
│     → logsumexp + gather → topk_log_probs [P, T, K]   (避免 full softmax)│
│     → 存入 batch: teacher_topk_log_probs, teacher_topk_ids (detached)  │
│                                                                         │
│  2. Student 端 (update_policy → _forward_micro_batch)                   │
│     input_ids → model forward → logits [N, V] (unpadded)               │
│     → torch.utils.checkpoint.checkpoint(_compact_fn, logits):           │
│       ├─ log_softmax(logits) → log_probs_full [N, V]                   │
│       ├─ topk(logits, K) → idx [N, K]                                  │
│       ├─ gather(log_probs_full, idx) → topk_lp [N, K]                  │
│       └─ topk_lp.exp() → topk_p [N, K]                                │
│     → pad_input → [B, T, K] 的 topk_log_probs, topk_probs, topk_ids   │
│                                                                         │
│  3. PID Loss (_compute_pid_loss → compute_topk_kl)                     │
│     student top-K (with grad) + teacher top-K (detached)               │
│     → _topk_match_and_gather 交叉匹配                                  │
│     → Top-K KL with tail compensation → scalar loss                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Teacher 端显存优化** (`dp_actor.py:_forward_teacher`)：
- 不使用 padding-free 模式（简化实现），直接 padded forward
- `output.logits[:, -T-1:-1, :].detach().clone()` 后立即 `del output`，只保留 response 部分
- 计算 top-K 时用 `logsumexp` + `gather` 替代完整 softmax，避免 `[P, T, V]` 的 float32 张量
- 每个 micro-batch 后调用 `torch.cuda.empty_cache()` 回收碎片

**Student 端显存优化** (`dp_actor.py:_forward_micro_batch`)：
- 使用 `torch.utils.checkpoint.checkpoint` 包裹 top-K 计算，反向传播时重算 `[N, V]` 的 softmax，避免保存在前向图中
- 在 padding-free 模式下，先在 unpadded 空间 `(total_nnz, V)` 上计算 top-K，再 `pad_input` 回 `[B, T, K]`——K 维很小，pad 开销低

**FSDP 死锁预防** (`dp_actor.py:compute_teacher_log_prob`)：
- Teacher forward 必须处理**所有** batch（不仅是 PID 样本），因为 FSDP all-gather 要求所有 rank 执行相同次数的 forward
- Non-PID 位置的结果通过 `pid_mask * tensor` 清零，下游不会读取

#### Top-K KL with Tail Compensation (`compute_topk_kl`)

> `verl/trainer/core_algos.py:662`

Student 和 Teacher 各自独立计算 top-K，两个 top-K 集合可能不同（如 student 把 token A 排在 top-K 但 teacher 没有）。跨集合匹配通过 `_topk_match_and_gather` 完成：

**`_topk_match_and_gather` 匹配算法** (`core_algos.py:628`)：
```python
# 输入: query_ids [B, T, K], key_topk_log_probs [B, T, M], key_topk_ids [B, T, M]
# 目标: 对 query 中的每个 token，在 key 的 top-M 中查找对应的 log_prob

match = query_ids[:,:,:,None] == key_topk_ids[:,:,None,:]  # [B, T, K, M] 广播比较
lp_matched = (match * key_topk_log_probs).sum(-1)          # 命中 → 取 key 的 log_prob
has_match = match.any(-1)                                    # [B, T, K] 是否命中

# 未命中 → tail 均匀近似: log(tail_mass / (vocab_size - M))
key_tail_mass = 1.0 - key_topk_probs.sum(-1)               # [B, T]
miss_lp = log(tail_mass) - log(150000 - M)                  # 假设 vocab ≈ 150K

result = where(has_match, lp_matched, miss_lp)              # [B, T, K]
```

**KL 计算（以 student 的 top-K 为支撑集）：**
```python
# 先用 _topk_match_and_gather 获取 teacher 在 student top-K 位置的 log_probs
teacher_lp_at_student = _topk_match_and_gather(student_ids, teacher_lp, teacher_ids)  # [B,T,K]

# 计算 tail mass
p_tail_s = 1 - student_probs.sum(-1)   # student top-K 外的概率质量
p_tail_t = 1 - teacher_probs_at_student.sum(-1)  # teacher 在 student 支撑集上的 tail

# Forward KL: KL(S||T) = Σ_k p_s[k] * (log p_s[k] - log p_t[k]) + p_tail_s * log(p_tail_s/p_tail_t)
# Reverse KL: KL(T||S), 交换 P 和 Q
# JSD: 0.5 * KL(S||M) + 0.5 * KL(T||M), M = 0.5 * (S + T)
```

**JSD 模式 (`jsd_kl`) 的 Top-K 计算：**

```python
# M = 0.5 * (P_student + P_teacher), 均在 student 的 top-K 支撑上评估
m_probs = 0.5 * (student_topk_probs + teacher_probs_at_student)     # [B, T, K]
m_tail  = 0.5 * (p_tail_student + p_tail_teacher)                   # [B, T]

# JSD = 0.5 * KL(S || M) + 0.5 * KL(T || M)
kl_s_m = (S_probs.detach() * (S_log_probs - log(M))).sum(-1) + tail_s  # stop-grad on weight
kl_t_m = (T_probs         * (T_log_probs - log(M))).sum(-1) + tail_t
jsd    = 0.5 * (kl_s_m + kl_t_m)                                   # [B, T]
```

注: JSD 中对 `S_probs` 的 weight 做了 `detach()`，避免当 `p_s → 0` 时梯度爆炸。

#### PID Mask 逻辑详解 (`compute_pid_mask`)

> `verl/trainer/core_algos.py:733`

```
假设 group n=5, pid_threshold=0.2:

Group A: [正确, 错误, 错误, 错误, 错误]  → accuracy=0.2 → < 0.2? No  → 不激活
Group B: [错误, 错误, 错误, 错误, 错误]  → accuracy=0.0 → < 0.2? Yes → 激活
  └─ pid_group_mask: 全组 True (5个)
  └─ pid_mask:       仅 accuracy≠1.0 的 True (5个, 因为全错)

Group C: [正确, 正确, 错误, 错误, 错误]  → accuracy=0.4 → < 0.2? No  → 不激活
```

两个 mask 的不同作用：
- `pid_mask`: 决定哪些样本接受 PID 蒸馏损失。默认仅答错的样本；当 `pid_all_trajectories=true` 时，PID 组内所有样本（含正确的）都接受蒸馏
- `pid_group_mask`: 决定哪些样本跳过 ref KL penalty（整个 PID 组的所有样本，包括答对的，不受 `pid_all_trajectories` 影响）

#### Metrics

| Metric | 说明 |
|--------|------|
| `pid/pid_groups` | 激活 PID 的 group 数 |
| `pid/total_groups` | 总 group 数 |
| `pid/pid_trajectories` | PID mask 为 True 的样本数 |
| `pid/mask_ratio` | PID mask 占比 |
| `pid/group_mask_ratio` | PID group mask 占比 |
| `actor/pid_kl` | PID KL 散度（未乘系数） |
| `actor/pid_loss` | PID 损失（已乘 pid_coef） |
| `actor/pid_ratio` | 当前 micro-batch 中 PID 样本占比 |

---

## 5. 分布式架构

### 5.1 Single Controller + Ray WorkerGroup

> `verl/single_controller/`

采用 **Single Controller** 模式：中心 Driver 通过 Ray RPC 调度 Worker。

**核心类：**

- `RayResourcePool`: GPU 资源池管理，创建 Ray placement group
- `RayClassWithInitArgs`: 包装 Ray Actor 的创建参数
- `RayWorkerGroup`: Worker 组管理，支持同步/异步调用

**调度方式：**

```python
# 广播：所有 worker 收到相同参数
worker_group.execute_all_sync("compute_log_probs", batch)

# 分发：每个 worker 收到不同参数（参数为 list 且长度等于 worker 数）
worker_group.execute_all_sync("update_actor", [chunk1, chunk2, ...])
```

**Worker 环境变量：**
每个 Worker 启动时自动设置 `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT` 等。

### 5.2 Hybrid Engine (FSDP + vLLM)

> `verl/workers/sharding_manager/fsdp_vllm.py` — `FSDPVLLMShardingManager`

Hybrid Engine 的核心思想是在同一组 GPU 上交替运行 FSDP 和 vLLM，通过权重同步避免显存翻倍。

**切换流程：**

```
训练模式 (FSDP)
    │
    ▼  prepare_rollout_engine()
    │  ┌─ vLLM.wake_up(tags=["weights"])     # 分配 vLLM 权重显存
    │  ├─ _sync_weight_to_vllm()              # FSDP 权重 → vLLM 权重
    │  └─ vLLM.wake_up(tags=["kv_cache"])     # 分配 KV cache 显存
    │
推理模式 (vLLM)
    │  generate_sequences()
    │
    ▼  release_rollout_engine()
    │  ┌─ vLLM.sleep(level=1)                 # 释放权重 + KV cache 显存
    │  ├─ model.train()                        # 恢复训练模式
    │  └─ torch.cuda.empty_cache()
    │
训练模式 (FSDP)
```

**权重同步 (`_sync_weight_to_vllm`)：**

- **Full Model**: `get_model_state_dict(fsdp_model)` → 重命名 key → `vllm_model.load_weights()`
- **LoRA Model**: 逐层收集 LoRA 权重 → 创建 `TensorLoRARequest` → `vllm_engine.add_lora()`
- 可选 `use_param_offload`：同步时先将 FSDP 参数加载到 GPU，同步完再卸载到 CPU

### 5.3 Ulysses 序列并行

> `verl/models/monkey_patch.py`

对 Attention 层做 monkey patch，实现序列维度的并行：

```
原始输入: qkv [B, seq_len/SP, num_heads, head_dim]
    → gather_seq_scatter_heads:  [B, seq_len, num_heads/SP, head_dim]  (all-to-all)
    → Flash Attention
    → gather_heads_scatter_seq:  [B, seq_len/SP, num_heads, head_dim]  (all-to-all)
```

支持的模型：LLaMA, Gemma, Mistral, Qwen2/3 (含 MoE), Qwen2-VL, Qwen2.5-VL, Qwen3-VL。

---

## 6. 数据集与 DataLoader

### RLHFDataset

> `verl/utils/dataset.py`

**数据格式要求：**

```jsonl
{"prompt": "问题文本", "answer": "标准答案"}
{"prompt": "带图片的<image>问题", "images": ["path/to/img.jpg"], "answer": "答案"}
{"prompt": "带视频的<video>问题", "videos": ["path/to/vid.mp4"], "answer": "答案"}
```

**PID 模式额外字段** (`prompt_with_hint_key` 非空时)：
```jsonl
{"prompt": "原始问题", "prompt_with_hint": "附加了 hint 的问题", "images": [...], "answer": "答案"}
```
DataLoader 会同时 tokenize `prompt` 和 `prompt_with_hint`，生成 `hint_input_ids/hint_attention_mask/hint_position_ids`。

**处理流程：**

1. **消息构建** (`_build_messages`): 将 prompt 转换为 chat message 格式
   - 文本: `[{"role": "user", "content": "..."}]`
   - 图片: 按 `<image>` 分隔，插入 `{"type": "image"}` 标记
   - 视频: 类似图片处理

2. **Tokenization**: 使用 processor 的 `apply_chat_template` 生成 token IDs

3. **Position IDs**:
   - Qwen2-VL/Qwen3-VL: 使用 MROPE `get_rope_index()` 生成 3D/4D position IDs
   - 其他模型: 从 attention_mask 累加生成

4. **Left Padding**: 所有序列左填充到 `max_prompt_length`

**数据过滤：** `filter_overlong_prompts=True` 时，过滤超过 `max_prompt_length` 的样本。

### DataLoader

> `verl/trainer/data_loader.py`

- **训练**: `StatefulDataLoader`（支持 checkpoint 恢复），随机采样，`drop_last=True`
- **验证**: 顺序采样，`drop_last=False`，默认加载全部验证集

---

## 7. 配置系统

> `verl/trainer/config.py` + `verl/workers/config.py` + 各子目录的 `config.py`

总配置 `PPOConfig` 的树形结构：

```yaml
PPOConfig:
  data:                              # 数据配置
    train_files: ""
    val_files: ""
    prompt_key: "prompt"
    answer_key: "answer"
    image_key: "images"
    video_key: "videos"
    max_prompt_length: 512
    max_response_length: 512
    rollout_batch_size: 512
    min_pixels: 262144               # 图片最小像素数
    max_pixels: 4194304              # 图片最大像素数
    filter_overlong_prompts: true
    prompt_with_hint_key: null       # PID hint prompt 列名 (e.g. "prompt_with_hint")
    max_hint_prompt_length: null     # hint prompt 最大长度 (默认 = max_prompt_length)

  algorithm:                         # 算法配置
    adv_estimator: "grpo"            # gae | grpo | rloo | reinforce_plus_plus | remax
    gamma: 1.0                       # 折扣因子
    lam: 1.0                         # GAE lambda
    kl_penalty: "kl"                 # kl | abs | mse | low_var_kl | full
    kl_coef: 0.001
    kl_direction: "reverse_kl"      # reverse_kl | forward_kl | jsd_kl
    disable_kl: false                # 禁用 reference model
    use_kl_loss: false               # KL 作为损失 vs 作为奖励惩罚
    online_filtering: false          # 在线过滤
    # --- PID-GRPO ---
    enable_pid: false                # 开启 PID 在线蒸馏
    pid_threshold: 1.0               # 组内正确率阈值 (< threshold 激活)
    pid_coef: 1.0                    # PID 损失系数
    pid_top_k: 64                    # Top-K token 数 (0 = single-token 模式)
    pid_kl_penalty: "low_var_kl"     # PID single-token 模式的 KL 估计器
    pid_kl_direction: "forward_kl"   # PID KL 方向: forward_kl | reverse_kl | jsd_kl
    pid_all_trajectories: false      # true = 对 PID 组内所有轨迹蒸馏; false = 仅错误轨迹

  trainer:                           # 训练器配置
    total_epochs: 15
    max_steps: null                  # 覆盖 total_epochs
    nnodes: 1
    n_gpus_per_node: 8
    critic_warmup: 0                 # 前 N 步只更新 critic
    val_freq: -1                     # 验证频率 (-1 = 禁用)
    save_freq: -1                    # 保存频率
    save_limit: -1                   # 最多保留 checkpoint 数
    logger: ["console", "wandb"]

  worker:                            # Worker 配置
    hybrid_engine: true

    actor:
      strategy: "fsdp"
      global_batch_size: 256         # mini-batch 大小
      micro_batch_size_per_device_for_update: 4
      micro_batch_size_per_device_for_experience: 16
      ppo_epochs: 1
      clip_ratio_low: 0.2
      clip_ratio_high: 0.3
      loss_type: "default"           # default | gspo | cispo | sapo
      loss_avg_mode: "token"         # token | seq
      padding_free: true
      dynamic_batching: true
      ulysses_size: 1                # 序列并行度
      use_torch_compile: true
      model:
        model_path: null
        enable_gradient_checkpointing: true
        freeze_vision_tower: false
        lora: {enable: false, rank: 8, alpha: 16, ...}
      optim:
        lr: 1e-6
        lr_scheduler_type: "constant"
      fsdp:
        enable_full_shard: true
        mp_param_dtype: "bf16"
        mp_reduce_dtype: "fp32"

    critic:
      strategy: "fsdp"
      cliprange_value: 0.5
      # ... (结构类似 actor)

    rollout:
      name: "vllm"
      n: 1                           # 每个 prompt 采样数
      temperature: 1.0
      top_p: 1.0
      gpu_memory_utilization: 0.6
      tensor_parallel_size: 2
      max_num_batched_tokens: 8192

    reward:
      reward_function: null           # 自定义奖励函数路径
      num_cpus: 1

    ref:
      offload: false
      fsdp: {enable_full_shard: true, ...}
```

---

## 8. Checkpoint 系统

> `verl/utils/checkpoint/`

### 保存内容

每个 rank 保存自己的 shard：
```
global_step_100/
├── model_world_size_8_rank_0.pt   # FSDP 模型 shard
├── model_world_size_8_rank_1.pt
├── ...
├── optim_world_size_8_rank_0.pt   # 优化器 shard
├── ...
├── extra_state_world_size_8_rank_0.pt  # LR scheduler + RNG state
├── ...
├── huggingface/                    # HuggingFace 格式 (仅 rank 0)
│   ├── config.json
│   └── tokenizer files
└── lora/                           # LoRA 权重 (如果使用)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

### Checkpoint 追踪

```json
// checkpoint_tracker.json
{
    "last_global_step": 100,
    "best_global_step": 50,
    "best_metric": 0.85,
    "last_dataloader_state_dict": {...}
}
```

### 自动清理

`save_limit > 0` 时，自动删除旧 checkpoint，仅保留最新的 N 个（best checkpoint 始终保留）。

### 恢复训练

`find_last_checkpoint=True` 时自动查找最新 checkpoint 并恢复：
- 模型权重 + 优化器状态
- LR scheduler 状态
- RNG 状态（保证可复现）
- DataLoader 状态（从中断位置继续）

---

## 9. Metrics 与日志

> `verl/trainer/metrics.py`

### 记录的 Metrics

| 类别 | Metrics |
|------|---------|
| **奖励** | `critic/score/{mean,max,min}`, `critic/rewards/{mean,max,min}` |
| **优势** | `critic/advantages/{mean,max,min}`, `critic/returns/{mean,max,min}` |
| **值函数** | `critic/values/{mean,max,min}`, `critic/vf_explained_var` |
| **序列长度** | `response_length/{mean,max,min,clip_ratio}`, `prompt_length/{mean,max,min,clip_ratio}` |
| **Actor 更新** | `actor/pg_loss`, `actor/pg_clipfrac`, `actor/pg_ratio`, `actor/entropy` |
| **Critic 更新** | `critic/vf_loss`, `critic/vf_clipfrac` |
| **KL** | `critic/kl/mean`, `critic/kl_coef` |
| **PID** | `pid/{pid_groups,total_groups,pid_trajectories,mask_ratio,group_mask_ratio}`, `actor/{pid_kl,pid_loss,pid_ratio}` |
| **耗时** | `timing_s/{gen,reward,log_probs,ref_log_probs,values,adv,teacher,update_actor,update_critic}` |
| **吞吐** | `perf/total_num_tokens`, `perf/time_per_step`, `perf/throughput` (tokens/s/GPU) |

### 支持的日志后端

- `console`: 控制台输出
- `wandb`: Weights & Biases
- `tensorboard`: TensorBoard
- `mlflow`: MLflow
- `swanlab`: SwanLab
