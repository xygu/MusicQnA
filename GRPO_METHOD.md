# 本仓库中的 GRPO：原理、实现与常见坑

说明 `musiccaps/train_grpo.py` 中的训练流程：**同一音频条件下并行采样多条 caption**，用**组内标准化后的优势**做策略梯度，并可加 **KL 惩罚**。下文与当前代码一致，便于对照。

**建议阅读顺序**：§1 术语 → §2 本仓库在干什么 → §3 数学与奖励细节 → §4 背景（RLHF / PPO / DPO / GRPO）与对照表 → §5 起：配置、实现、排错、PEFT。

---

## 1. 术语与对象一览

读后面章节前先对齐这些词在本仓库里的含义。

| 术语 | 含义（本仓库） |
|------|----------------|
| **策略 \(\pi_\theta\)** | 带 **LoRA** 的多模态语言模型（如 Qwen2.5-Omni）：输入含音频与文本指令，输出 caption。 |
| **参考策略 \(\pi_{\text{ref}}\)** | **同一基座、关闭 LoRA**（`disable_adapter()`）时的模型，用于 KL；**不是**「上一轮 LoRA 权重快照」。 |
| **\(G\)（组大小）** | 每条训练音频上 **并行采样** 的 caption 条数；组内算优势。 |
| **Aspect** | MusicCaps 里的 **人工标注维度**（如乐器、情绪）。`aspect_coverage_score` 用 **子串 + 启发式分词** 看 caption 是否「提到」各维，再平均，分数量级约在 \([0,1]\)。偏 **可解释、可检查**，但易被 **堆关键词** 投机。 |
| **CLAP** | **Contrastive Language-Audio Pretraining** 一类模型：把 **音频** 和 **文本** 映射到**同一向量空间**，语义相近的音频–文本对 embedding 更接近。本仓库用 Hugging Face 的 **LAION CLAP**（`ClapModel` / `ClapProcessor`），对每条 \((\text{wav}, \text{caption})\) 分别取音频特征与文本特征，**L2 归一化**后算 **余弦相似度** \(\in [-1,1]\)，再线性映射为 \(\frac{\text{sim}+1}{2}\)，落到 **[0, 1]** 作为 **`ClapScorer.audio_text_scores`**。CLAP 在 GRPO 里充当 **冻结的打分器**（不随 LoRA 一起训练），衡量 **生成描述与真实音频听感/语义是否对齐**，与 Aspect 的「是否覆盖标签」互补。调试非 Omni 路径时可关 CLAP。 |
| **GRPO（本文档所指）** | **G**roup 内 **R**elative：对同一条件下的 \(G\) 条回答算标量奖励，用 **组内 z-score** 当优势，再乘 \(\log\pi_\theta\) 做梯度；**无** PPO 的 ratio clip、**无** 单独 critic，接近 **REINFORCE + 组内 baseline**。 |

---

## 2. 本仓库在干什么（流程）

1. 取一条样本：音频路径、`aspects` 等（见 manifest）。
2. 固定多模态 prompt，用当前 **LoRA 策略** 采样 **\(G\) 条** caption：`generate_group(..., group_size=G)`。
3. 对每条 caption 算 **标量奖励** \(R_i\)：`combine_rewards` 融合 **Aspect 分** \(a_i\) 与 **CLAP 分** \(c_i\)（权重可配）。
4. `group_advantages`：对 \(\{R_i\}_{i=1}^G\) 做 **均值方差标准化** 得 \(A_i\)。
5. `completion_log_probs`：对每条 completion **teacher forcing** 算 \(\sum_t \log\pi_\theta(y_{i,t}\mid\cdots)\)。
6. 损失：**策略梯度项** \(\mathcal{L}_{PG}=-\frac{1}{G}\sum_i A_i \log\pi_\theta(y^{(i)}\mid x)\)，加上可选 **KL**（当前策略相对 **无 LoRA 底座** 的 log 概率差）。
7. 反向更新 **LoRA 参数**（AdamW 最小化总损失）。

一句话：**用同一段音频上「谁比同组其他人更好」的相对优势，推高好 caption 的概率；KL 拉住不要离底座太远。**

---

## 3. 数学对象与代码对应

### 3.1 采样

对每条 `row`（含 `wav_path`、`aspects` 等）：`generate_group` → \(\{y^{(i)}\}_{i=1}^G\)。

### 3.2 标量奖励

- **Aspect**：`aspect_coverage_score`，约在 \([0,1]\)。
- **CLAP**：`ClapScorer.audio_text_scores`：同上节，余弦相似度映射到 \([0,1]\)。
- **融合**（`combine_rewards`）：

\[
R_i = \frac{w_{\text{asp}}\, a_i + w_{\text{clap}}\, c_i}{w_{\text{asp}} + w_{\text{clap}}}
\]

（某权重为 0 即关闭该项。）

### 3.3 组内优势

`group_advantages(rewards)`：

\[
\mu = \frac{1}{G}\sum_i R_i,\quad
\sigma = \sqrt{\frac{1}{G}\sum_i (R_i-\mu)^2},\quad
A_i = \frac{R_i - \mu}{\sigma + \varepsilon}.
\]

- 同一条音频的一组里，**高于组平均**的样本 \(A_i>0\)，梯度 **提高** 其 \(\log\pi\)。
- \(\varepsilon=10^{-8}\)；**G=1** 时 \(\sigma=0\)，优势为 0，**几乎学不动**（见 §8）。

### 3.4 策略梯度损失

`completion_log_probs` 对 assistant 段求 \(\sum_t \log \pi_\theta(y_t|\cdot)\)，得长度 \(G\) 的 `logp`。

\[
\mathcal{L}_{\text{PG}} = -\frac{1}{G}\sum_{i=1}^G A_i \cdot \log \pi_\theta(y^{(i)} \mid x).
\]

### 3.5 KL 惩罚

`_kl_penalty`：`PeftModel.disable_adapter()` 下对同一 \((x,y^{(i)})\) 再算 `completion_log_probs` → ref。

\[
\mathcal{L}_{\text{KL}} = \beta_{\text{kl}} \cdot \mathbb{E}_i\bigl[\log\pi_\theta - \log\pi_{\text{ref}}\bigr].
\]

---

## 4. 背景：RLHF、PPO、DPO、GRPO 各是什么，如何对比

四者**不在同一抽象层**，不宜当成四个平级「算法名」硬比。

- **RLHF**：**整条对齐管线**（SFT → 用人类偏好训奖励模型 RM → 用标量奖励 + KL 做强化学习）。
- **PPO**：常作为 RLHF **第三阶段**的**优化算法**（clip、旧策略 ratio、critic/GAE 等）。
- **DPO**：**第三阶段的一种替代**：**偏好对**数据上直接优化策略，训练里**不单独 forward RM**。
- **GRPO（本文）**：**优势怎么构造**的一种做法：组内相对奖励 + 策略梯度；可接在「已有标量奖励」的 RL 设定上，**不必**等于 PPO。

### 4.1 RLHF（简要）

**目的**：在模仿示范（SFT）之外，贴合人类「更喜欢哪种回答」，并控制漂移。

1. **SFT**：\((\text{prompt}, \text{示范回答})\)。
2. **RM**：人类排序/打分 → \(r_\phi(x,y)\)。
3. **RL**：最大化 \(\mathbb{E}[r_\phi]\)，对 \(\pi_{\text{ref}}\) 加 KL。

第三阶段需要 **能对任意 \((x,y)\) 打标量分** 的模块；本仓库用 **规则 Aspect + 冻结 CLAP**，**不训**经典意义上的 RM。

### 4.2 PPO（简要）

**目的**：策略梯度更新 **别太猛**（稳定）。

用 **裁剪后的替代目标**（\(\pi_\theta/\pi_{\text{old}}\) 被 clip）配合 **优势** \(A\)；\(A\) 常来自 **critic + GAE** 等。需维护旧策略，常 **多 epoch** 复用样本。

**与 RLHF**：PPO 是 **实现 RLHF 第三阶段的常见选择**，不是 RLHF 的同义词。

### 4.3 DPO（简要）

**目的**：只用 **偏好对** \((x,y^+,y^-)\) 调策略，**不显式**训 RM、**不**走 PPO 采样环。

损失在 \(\pi_\theta\)、\(\pi_{\text{ref}}\) 与偏好上；**数据形态是成对偏好**，不是「每条回答一个标量 \(R\)」。

### 4.4 GRPO（本文档语境，简要）

**目的**：已有 **标量** \(R(x,y)\) 时，**少用或不用 critic**，仍做策略梯度。

**本仓库**：每组 \(G\) 条采样 → \(R_i\) → 组内 z-score 得 \(A_i\) → \(-\sum A_i\log\pi\) + KL。**无** PPO clip、**无** critic。

### 4.5 对照表（同一套坐标轴）

| 维度 | RLHF（整管线） | PPO（典型：RLHF 第三阶段） | DPO | GRPO（本仓库一类） |
|------|----------------|---------------------------|-----|---------------------|
| **层次** | 流程 | 优化算法 | 目标 + 范式 | 优势构造 + 策略梯度形式 |
| **数据** | SFT；RM 用偏好；RL 在线采样 | 在线 \((x,y)\)，\(y\sim\pi_\theta\) | 离线 \((x,y^+,y^-)\) | 在线每组 \(G\) 条 + 标量奖励 |
| **中间模块** | 常显式 RM | 常 critic + 旧策略 | 无 RM、无 critic | 无 critic；组内统计当 baseline |
| **稳定性** | KL + 常配合 PPO clip | clip + 价值 bootstrap | \(\beta\) 控离 ref | 本仓库：KL；无 ratio clip |
| **\(\pi_{\text{ref}}\)** | RL 阶段常见 SFT | 同左 | 损失显式出现 | 本仓库：底座无 LoRA |

**易混**：

- **RLHF vs PPO**：管线 vs 管线上的一步算法。
- **DPO vs GRPO**：前者要 **偏好对**；后者要 **每条一个 \(R\)**。有标量奖励（规则+CLAP）时走 GRPO 式更新更直接。

### 4.6 本仓库 vs「典型 PPO + 价值网络」

| 维度 | 常见 PPO + critic | 本仓库 `train_grpo.py` |
|------|-------------------|-------------------------|
| 优势 | critic / GAE 等 | 组内 **z-score** |
| 目标 | clip + 常多 epoch | **无 clip**：\(-\mathbb{E}[A\log\pi]\) + KL |
| ref | 常 frozen 整模 | **PEFT**：无 LoRA 的底座算 ref logp |

### 4.7 附录：管线心智模型

```text
RLHF（大框）
  ├─ SFT
  ├─ 训练 RM（人类偏好）
  └─ 标量奖励对齐策略
        ├─ 常见：PPO（critic + clip + 在线采样）
        ├─ 替代：DPO（偏好对）
        └─ 变体：组内相对优势 + REINFORCE（本仓库：无 clip、无 critic）
```

本仓库：**没有**「人类排序训 RM」的经典前两步，但第三阶段 **形态**仍是采样 → 标量奖励 → KL；奖励来自 **Aspect + CLAP**。

---

## 5. 训练配置要点（`TrainConfig`）

- **`grpo_batch_size` 必须为 1**（Omni + 逐条 reward）；更大则 `ValueError`。
- **`grpo_group_size`**：\(G\) 越大组内 baseline 越稳，算力约乘 \(G\)。
- **`grpo_temperature`**：过低则组内 \(R_i\) 趋同 → \(A_i\approx 0\)。
- **`beta_kl`**：0 关闭 KL；非 PEFT 时 KL 跳过。
- **`grpo_clap_device`**：CLAP 可与主模型分卡；注意总显存。
- **`grpo_init_adapter`**：若目录存在且启用 LoRA，先 `load_adapter_checkpoint`（见 §7）。

---

## 6. 实现细节（易忽略）

### 6.1 `generate_group` 与 `completion_log_probs`

- 生成：`model.eval()`，随机性主要来自 **softmax + temperature**。
- 算 logp：`model.train()`；若 LoRA dropout>0，与采样时前向**略有差异**。

### 6.2 梯度裁剪

`clip_grad_norm_` 作用在整模参数上；仅 LoRA 可训时仍只更新 `requires_grad=True` 部分。

### 6.3 保存

`save_trainable` 一般为 **适配器目录**，非完整基座。

---

## 7. `PeftModel.load_adapter`（PEFT 新版 API）

`TypeError: missing adapter_name` 来自需显式 `adapter_name`。本仓库：

```text
load_adapter(path, adapter_name="default", is_trainable=True)
```

### 7.1 参数简表（以 `peft` 文档为准）

| 参数 | 作用 |
|------|------|
| `model_id` | 含 `adapter_config.json` 与权重的路径或 Hub id |
| `adapter_name` | 逻辑名；与 `get_peft_model` 默认 `"default"` 对齐时多为主加载权重到已有槽位 |
| `is_trainable` | GRPO 继续训应 **`True`** |
| `torch_device` / `autocast_adapter_dtype` / `low_cpu_mem_usage` 等 | 设备与加载优化 |
| `**kwargs` | 如 `ignore_mismatched_sizes`、`device_map` 等 |

### 7.2 注意

- 新适配器有时需 `set_adapter`；向已有 `default` 加载通常与当前流程一致。
- 磁盘 `adapter_config` 应与当前 `LoraConfig` **一致**，否则可能报错或静默不匹配。

---

## 8. 常见坑与调试

### 8.1 学不动

- \(G\) 小或 temperature 低 → 组内奖励几乎相同。
- Aspect 全空时 aspect 分恒 1.0；若再关 CLAP → 奖励常数 → 优势为 0。

### 8.2 奖励投机

- Aspect：**堆关键词**；CLAP：**句式长度等偏置** 仍可能存在。看 `reward_weight_aspect` / `reward_weight_clap` 与日志 `R_mean`、`sample0`。

### 8.3 KL

- Ref 是 **无 LoRA 底座**，不是「SFT 结束时的 LoRA」。若要相对 SFT LoRA 约束，需改实现（双适配器或冻结拷贝）。

### 8.4 显存

- 峰值 ≈ 主模型 + **CLAP** + \(G\) 次生成与 logp 相关激活。

### 8.5 文献中的「GRPO」

- 不同论文可能含排序、DPO 式对比、别的归一化。本仓库以 `train_grpo.py` + `rewards.py` 为准：**组内 z-score + 可选 KL + 无 PPO clip**。

### 8.6 `debug_use_mock_model`

- Tiny GPT-2 + LoRA 可能出现 `fan_in_fan_out` 警告，一般可忽略；与 Omni 主路径无关。

---

## 9. 相关文件

| 文件 | 作用 |
|------|------|
| `musiccaps/train_grpo.py` | 主循环、KL、优化器 |
| `musiccaps/rewards.py` | Aspect、融合、组内优势 |
| `musiccaps/clap_scorer.py` | **CLAP** 加载、重采样、\([0,1]\) 分数 |
| `musiccaps/lm_backend.py` | 生成、logp、适配器存取 |
| `musiccaps/config.py` / `configs/default.yaml` | 超参 |

---

若需与某篇论文（如 DeepSeek-R1）中的 GRPO 定义 **逐条对齐**，可指定版本后单独加「与论文差异」表。
