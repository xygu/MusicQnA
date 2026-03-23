from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-Omni-3B"
    dtype: str = "bfloat16"
    debug_use_mock_model: bool = False

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    manifest_path: str = "./data/musiccaps_manifest.jsonl"
    max_samples: Optional[int] = None

    sft_epochs: int = 1
    sft_batch_size: int = 1
    sft_learning_rate: float = 1e-5
    sft_max_grad_norm: float = 1.0
    sft_log_every: int = 10
    sft_save_every: int = 200

    grpo_epochs: int = 1
    grpo_batch_size: int = 1
    grpo_group_size: int = 4
    grpo_learning_rate: float = 5e-6
    grpo_max_new_tokens: int = 128
    grpo_temperature: float = 0.9
    grpo_max_grad_norm: float = 1.0
    grpo_log_every: int = 5
    grpo_save_every: int = 100
    reward_weight_aspect: float = 0.5
    reward_weight_clap: float = 0.5
    beta_kl: float = 0.02

    clap_model_id: str = "laion/larger_clap_music"
    clap_batch_size: int = 2
    grpo_clap_device: Optional[str] = None

    checkpoint_dir: str = "./checkpoints"
    sft_adapter_name: str = "sft_lora"
    grpo_adapter_name: str = "grpo_lora"
    grpo_init_adapter: Optional[str] = "sft_lora"
    seed: int = 42


def load_train_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = TrainConfig()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
