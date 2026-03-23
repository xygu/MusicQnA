# MusicCaps 任务二：SFT + GRPO 流水线说明

**按步骤运行与注意事项**见根目录 [README.md](./README.md)。

面向 **Qwen2.5-Omni Thinker（仅文本输出）** + **MusicCaps**：先 **SFT** 拟合参考描述，再 **GRPO** 用 **aspect 覆盖分** 与 **CLAP 音频–文本对齐分** 做组内相对优化。

## 目录结构（便于定位问题）

```
musiccaps_pipeline/
  MUSICCAPS_PIPELINE.md          # 本文档
  requirements.txt
  configs/
    default.yaml                 # 默认超参（可复制改 local.yaml）
  musiccaps/                     # Python 包
    __init__.py
    __main__.py                  # python -m musiccaps --help
    config.py                    # 单一配置入口：YAML -> dataclass
    schema.py                    # Manifest 行类型与校验（早失败）
    dataset.py                   # 只读 manifest，不做隐式 I/O
    chat.py                      # 组装 Omni 多轮对话（音频 path + 指令）
    rewards.py                   # aspect 覆盖 + CLAP 分数（纯函数 + 可单测）
    clap_scorer.py               # CLAP 加载与 batch 推理（与 GRPO 解耦）
    lm_backend.py                # Omni Thinker / mock GPT2：SFT loss、generate、completion logprob
    train_sft.py / train_grpo.py # CLI 入口（薄，只解析参数 + 调上面模块）
  scripts/
    build_manifest.py            # HF MusicCaps 元数据 + 本地 wav 目录 -> manifest.jsonl
```

设计原则：

1. **数据只通过 manifest 进入训练**：路径、caption、aspects、split 一行一条，缺文件可在 `build_manifest` 阶段发现，训练时不再猜路径。
2. **奖励与策略梯度拆开**：`rewards.py` 不 import 大模型；`clap_scorer.py` 单独加载 CLAP，避免 GRPO 里堆叠难查的依赖。
3. **先 mock 再真机**：`debug_use_mock_model: true` 时用极小 LM 跑通数据与 GRPO 形状，再切 Omni。

## 环境

```bash
cd musiccaps_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# 可选：开发模式
pip install -e .
```

依赖要点：`torch`、`transformers`（需支持你本机的 **Qwen2.5-Omni Thinker** 版本）、`peft`、`soundfile`、`datasets`、`pyyaml`、`tqdm`。

### 中国大陆访问 GitHub / Hugging Face

**Hugging Face（数据集与模型权重）**

- 本仓库在 `import musiccaps`（含 `python -m musiccaps …`）或运行 `scripts/build_manifest.py` 时，若你**未**设置环境变量 `HF_ENDPOINT`，会自动使用国内常用镜像 **`https://hf-mirror.com`**（与 `huggingface_hub` / `transformers` / `datasets` 兼容）。
- 若需**始终走官方 Hub**，请在运行前显式设置，例如：`export HF_ENDPOINT=https://huggingface.co`
- 若已设置 `HF_ENDPOINT`，代码**不会覆盖**你的值。
- 若不想应用上述默认镜像逻辑，可设置：`export MUSICCAPS_NO_CN_MIRROR=1`

**GitHub（例如 `pip install git+https://github.com/...` 或 `git clone`）**

Python 依赖若来自 GitHub，可在本机配置 Git 走代理镜像（择一即可，第三方服务可用性请自行评估）：

```bash
# 例：将 github.com 前缀替换为 ghproxy 镜像（按你环境可用的镜像调整）
git config --global url."https://ghproxy.net/https://github.com/".insteadOf "https://github.com/"
```

`pip` 从 Git 安装时也可把 URL 写成镜像前缀形式，例如：`git+https://ghproxy.net/https://github.com/org/repo.git`（具体以你选用的镜像文档为准）。

## 数据准备（必做）

MusicCaps 在 Hugging Face 上多为 **YouTube id + 时间戳**，你需要把 **~10s wav** 落到本地，并与元数据对齐（`build_manifest` 拉取元数据时会走上文「中国大陆访问」中的 Hub 镜像策略）。

1. 自行用合规方式下载/切片音频（本仓库**不包含**爬虫逻辑，避免版权与失效链接问题）。
2. 约定命名（与 `scripts/build_manifest.py` 一致）：

   `{ytid}_{start_s}_{end_s}.wav`  
   例：`--abc123--_30_40.wav`（具体以你落盘脚本为准，只要与 `build_manifest` 里函数一致即可）。

3. 生成 manifest：

```bash
export PYTHONPATH="$(pwd)"
python scripts/build_manifest.py \
  --wav-dir /path/to/musiccaps_wavs \
  --out-jsonl ./data/musiccaps_manifest.jsonl \
  --require-file
```

脚本会按 HuggingFace 的 **split 名** 与样本内 `split` 字段（若有）写入 `train` / `valid` / `test`；训练入口当前只使用 `split=train`。

**Manifest 字段（每行 JSON）**

| 字段 | 说明 |
|------|------|
| `id` | 稳定主键 |
| `wav_path` | 绝对或相对路径，训练时 `soundfile.read` |
| `caption` | 参考英文描述（SFT 监督信号） |
| `aspects` | 字符串列表，来自 MusicCaps `aspect_list` |
| `split` | `train` / `valid` / `test` |

## 阶段 A：SFT

**目标**：给定音频 + 固定指令，生成接近参考 `caption` 的文本。

**流程**（见 `musiccaps/lm_backend.py` 中 `OmniThinkerBackend`）：

1. `chat.build_supervised_conversation`：`system` + `user`（`audio` path + user text）+ `assistant`（caption）。
2. `processor.apply_chat_template` 得到 `input_ids` 等；用 **仅含 user 的对话 + `add_generation_prompt=True`** 再跑一次，对齐前缀长度，将 `labels` 中前缀置 `-100`，只对 **assistant** 算 CE。
3. `Qwen2_5OmniThinkerForConditionalGeneration` forward，`loss.backward()`；可选 **LoRA**（`peft`）。

**运行**（在 `musiccaps_pipeline` 目录下）：

```bash
export PYTHONPATH="$(pwd)"
python -m musiccaps.train_sft --config configs/default.yaml
# 或
python -m musiccaps sft --config configs/default.yaml
```

检查点目录由 `config.checkpoint_dir` 控制。

**调试建议**：`configs/default.yaml` 里 `debug_use_mock_model: true`，确认 loss 非 NaN、步数能跑完；再改为 `false` 加载 Omni。

## 阶段 B：GRPO

**目标**：在 SFT 模型上，用 **组内多条采样** 的相对优势更新策略；奖励为：

\[
R = w_{\text{asp}} \cdot R_{\text{aspect}} + w_{\text{clap}} \cdot R_{\text{clap}}
\]

- **Aspect 覆盖 \(R_{\text{aspect}}\)**（`rewards.py`）：将每条 aspect 视为短语，**子串命中** 或 **非停用词 token 过半命中** 记 1，否则 0；对 aspects 取平均，范围 \([0,1]\)。
- **CLAP 对齐 \(R_{\text{clap}}\)**（`clap_scorer.py`）：用 **LAION CLAP**（默认 `laion/larger_clap_music`）分别编码 **wav** 与 **生成句**，余弦相似度线性缩放到 \([0,1]\)。

**GRPO 一步**（见 `musiccaps/train_grpo.py` + `lm_backend.py`）：

1. 对每个样本构造 **仅 prompt** 的 `inputs`（与 SFT 同一指令模板，无 assistant）。
2. `generate` **G 次**（`num_return_sequences=G` 或循环），温度 `temperature`。
3. 对 **同一音频** 的 G 条文本算 \(R_1,\ldots,R_G\)，组内标准化：
   \(A_g = (R_g - \mu) / (\sigma + \epsilon)\)。
4. 对第 g 条 completion，用 **teacher forcing** 在 **生成 token 段** 上求 \(\sum_t \log \pi(a_t\mid \cdot)\)（`completion_log_probs`）。
5. **策略梯度损失**：\(-\mathbb{E}[A \cdot \log \pi]\)；可选 **KL(ref)**：`ref` 为冻结 SFT 权重副本，`beta_kl * KL` 加回 loss。

**运行**（需已存在 SFT adapter 或整模 checkpoint）：

```bash
export PYTHONPATH="$(pwd)"
python -m musiccaps.train_grpo --config configs/default.yaml
# 或
python -m musiccaps grpo --config configs/default.yaml
```

**快速冒烟（无 Omni、无音频）**：在配置中设 `debug_use_mock_model: true`，`manifest_path` 指向 `examples/mock_manifest.jsonl`，并放置任意短 `wav` 于 `examples/placeholder.wav`（mock 训练不校验文件，但 GRPO 若开启 CLAP 仍需真实音频；调试时可设 `reward_weight_clap: 0`）。

**资源**：CLAP 与 Omni 可同时占显存；若 OOM，可在 `default.yaml` 将 `grpo_clap_device` 设为 `cpu`（慢但稳）。

## 评估（流水线外可单独写 eval 脚本）

- **与参考 caption**：BLEU / ROUGE / CIDEr（需 `pycocoevalcap` 等）。
- **CLAP 分数**：本仓库 `ClapScorer` 对 (audio, pred) 打分的均值。
- **Aspect recall**：与 GRPO 中 `aspect_coverage_score` 一致，在 test manifest 上统计。

## 常见问题

| 现象 | 排查 |
|------|------|
| `apply_chat_template` 报错 | `transformers` 版本是否支持 `Qwen2_5OmniProcessor`；音频字段是否为 `{"type":"audio","path": "..."}`。 |
| SFT loss 不变 | `labels` 是否全为 -100（mask 错位）；确认 assistant 在对话里且模板包含生成内容。 |
| GRPO 全负 advantage | 提高 `temperature` 或增大 G，使组内方差 > 0。 |
| CLAP 很慢 | `clap_batch_size` 减小，或 CLAP 放 CPU。 |

## 许可

MusicCaps、Jamendo、Qwen、LAION CLAP 各有许可证；商用前请分别确认。
