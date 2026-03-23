# MusicCaps：音乐描述 SFT + GRPO 流水线

本目录实现 **MusicCaps** 任务：用 **Qwen2.5-Omni Thinker**（或调试用的极小 GPT2）先做 **SFT**，再做 **GRPO**；GRPO 奖励 = **aspect 覆盖** + **CLAP 音频–文本对齐**。

更偏设计的说明见 [MUSICCAPS_PIPELINE.md](./MUSICCAPS_PIPELINE.md)。

---

## 运行前约定

- 下文所有命令默认在 **`musiccaps_pipeline` 根目录**执行（含 `configs/`、`musiccaps/`、`scripts/` 的那一层）。
- 使用相对路径时，请保持当前工作目录为该根目录，否则请把配置里的 `manifest_path`、`checkpoint_dir` 改成绝对路径。

---

## 第 0 步：进入目录并创建虚拟环境

```bash
cd /path/to/musiccaps_pipeline
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**注意事项**

- **Python 版本**：建议 **3.10+**。
- **GPU**：正式训练 Omni 需要足够显存的 NVIDIA GPU；CPU 仅适合 mock 或极小规模试验。
- **可选安装**：`pip install -e .` 后可在任意目录执行 `python -m musiccaps`，但仍需通过 `--config` 指到本目录下的 yaml，或在该目录设置工作目录。

---

## 第 1 步：设置 `PYTHONPATH`（未 `pip install -e .` 时必做）

```bash
export PYTHONPATH="$(pwd)"
```

**注意事项**

- 每次新开终端都要重新执行，或把该行写进你的 shell 配置。
- 若已 `pip install -e .` 且只在包内开发，有时仍可保留 `PYTHONPATH`，避免导入到旧副本。

---

## 第 2 步：准备配置文件

```bash
cp configs/default.yaml configs/local.yaml
# 编辑 configs/local.yaml：至少改 manifest_path、model_id、checkpoint_dir 等
```

**注意事项**

- **`manifest_path`**：指向你将要生成或已生成的 `manifest.jsonl`（默认示例为 `./data/musiccaps_manifest.jsonl`）。
- **`model_id`**：与 Hugging Face 上 Omni Thinker 权重一致（如 `Qwen/Qwen2.5-Omni-3B`）；显存不够时换更小档或先开 mock。
- **`debug_use_mock_model: true`**：不加载 Omni，改用 **tiny GPT2** 做**流程冒烟**（不验证真实音频理解）。
- **`lora_r: 0`**：关闭 LoRA 时，本仓库的 **adapter 保存/加载路径（SFT→GRPO）不按 LoRA 场景测试**；正式用 Omni 建议 **`lora_r > 0`**。
- **Hugging Face 下载（含中国大陆）**：在 **`import musiccaps` 之前**未设置 `HF_ENDPOINT` 时，包内会默认尝试使用镜像 `https://hf-mirror.com`（见 `musiccaps/hub_mirrors.py`）。若要**始终走官方**：`export HF_ENDPOINT=https://huggingface.co`。若不想任何默认镜像：`export MUSICCAPS_NO_CN_MIRROR=1`。
- **仅运行 `scripts/build_manifest.py` 时**：该脚本**不会**自动 `import musiccaps`，因此不会触发上述默认镜像；在中国大陆可手动：`export HF_ENDPOINT=https://hf-mirror.com`，或在运行前执行一次：  
  `python -c "import sys; sys.path.insert(0,'.'); import musiccaps"`。

---

## 第 3 步：准备音频文件（正式训练必做）

MusicCaps 元数据是 **YouTube id + 起止秒**，需按合规方式下载并切成约 **10 秒 wav**。可用 **`scripts/download_musiccaps_wavs.py`**（依赖 `yt-dlp` 与系统 **`ffmpeg`**，见脚本顶部说明）。

**文件命名（与 `scripts/build_manifest.py` 一致）**

```text
{ytid}_{start_s}_{end_s}.wav
```

例如 `ytid` 为 11 位 id、`start_s=30`、`end_s=40`，则文件名为：`xxxxxxxxxxx_30_40.wav`（以你本地实际 `ytid` 为准）。

**注意事项**

- 命名必须与脚本中 `default_wav_name` 一致，否则 `build_manifest --require-file` 会大量跳过。
- 注意 **版权与平台服务条款**；MusicCaps 本身也有数据许可说明，商用前请自行核对。

---

## 第 4 步：生成 `manifest.jsonl`

```bash
export PYTHONPATH="$(pwd)"
# 可选：中国大陆访问 Hub
# export HF_ENDPOINT=https://hf-mirror.com

python scripts/build_manifest.py \
  --wav-dir /你的/wav/目录 \
  --out-jsonl ./data/musiccaps_manifest.jsonl \
  --require-file
```

**注意事项**

- **`--require-file`**：只写入**磁盘上真实存在**的 wav，避免训练时才报错；首次对齐数据时强烈建议打开。
- **split 字段**：脚本会按 HuggingFace 数据集上的 **split 名**（如 `train` / `test`）以及行内 `split` 字段（若有）映射到 `train` / `valid` / `test`。训练脚本**当前只读 `split=train`**，请确认 manifest 里训练集行是 `train`。
- **依赖**：需要能 `pip install datasets` 且能访问 Hub（或镜像）以下载 `google/MusicCaps` 元数据。

然后在 `configs/local.yaml` 里把 **`manifest_path`** 设为该 `musiccaps_manifest.jsonl`。

---

## 第 5 步：监督微调（SFT）

```bash
export PYTHONPATH="$(pwd)"
python -m musiccaps.train_sft --config configs/local.yaml
# 等价：python -m musiccaps sft --config configs/local.yaml
```

**注意事项**

- **输出位置**：默认在 `./checkpoints/sft_lora/`（由 `checkpoint_dir` + `sft_adapter_name` 决定）。
- **首次拉 Omni**：耗时长、占磁盘大；确保磁盘与网络稳定。
- **`transformers` 版本**：需支持你选的 `Qwen2_5OmniThinkerForConditionalGeneration` 与 `Qwen2_5OmniProcessor`；版本过旧会报 `apply_chat_template` 或参数错误。
- **batch**：`sft_batch_size` 默认 1，利于 Omni 多模态调试；调大前需确认显存与 processor 是否支持批处理。
- **mock 模式**：`debug_use_mock_model: true` 时**不检查 wav 是否存在**，仅用文本模板训练 tiny GPT2，用于验证 loss 与保存逻辑。

---

## 第 6 步：GRPO（依赖 SFT 的 LoRA 目录）

```bash
export PYTHONPATH="$(pwd)"
python -m musiccaps.train_grpo --config configs/local.yaml
# 等价：python -m musiccaps grpo --config configs/local.yaml
```

**注意事项**

- **`grpo_init_adapter`**：默认 `sft_lora`，即加载 **`./checkpoints/sft_lora`**（相对 `checkpoint_dir`）。**务必先成功跑完 SFT** 且该目录存在，否则会报错。
- **`grpo_batch_size` 必须为 1**：当前实现按单条样本做组内采样与奖励，若配置成其他值会直接 `raise`。
- **CLAP 与显存**：会额外加载 `laion/larger_clap_music`。若 OOM，可在配置里设 **`grpo_clap_device: "cpu"`**（慢但省 GPU），或暂时 **`reward_weight_clap: 0`**（仅 aspect + KL）。
- **mock + CLAP**：`debug_use_mock_model: true` 时默认**不加载 CLAP**；若你手动把 `reward_weight_clap` 调大，仍需要**真实 wav** 供 CLAP 读音频，否则请保持 **`reward_weight_clap: 0`**。
- **KL**：`beta_kl > 0` 时使用 PEFT 的 `disable_adapter()` 对**基座**算参考 log prob；非 PEFT 模型则 KL 项为 0。

**输出位置**：默认 `./checkpoints/grpo_lora/`。

---

## 第 7 步（可选）：不下载数据时的冒烟

1. 复制 `examples/mock_manifest.jsonl` 或在其基础上改 `caption` / `aspects`。
2. 在配置中设 **`debug_use_mock_model: true`**，`manifest_path` 指向该 jsonl。
3. **`reward_weight_clap: 0`**，避免 CLAP 读盘。
4. 运行第 5、6 步命令。

**注意事项**

- 这只能验证 **代码路径与梯度**，**不能**代表 Omni 在音乐上的真实效果。

---

## 命令速查

| 目的 | 命令 |
|------|------|
| SFT | `python -m musiccaps.train_sft --config configs/local.yaml` |
| GRPO | `python -m musiccaps.train_grpo --config configs/local.yaml` |
| 生成 manifest | `python scripts/build_manifest.py --wav-dir ... --out-jsonl ... --require-file` |

---

## 许可与数据

MusicCaps、Qwen、LAION CLAP 等各有许可证；商用与再分发前请分别阅读并遵守。
