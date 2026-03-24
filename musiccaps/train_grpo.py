from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from musiccaps.clap_scorer import ClapScorer
from musiccaps.config import TrainConfig, load_train_config
from musiccaps.dataset import training_rows
from musiccaps.lm_backend import build_backend
from musiccaps.prompts import SYSTEM_OMNI, USER_CAPTION_INSTRUCTION
from musiccaps.rewards import aspect_coverage_score, combine_rewards, group_advantages


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _kl_penalty(
    backend,
    logp: torch.Tensor,
    row,
    completions: list[str],
    beta: float,
) -> torch.Tensor:
    if beta <= 0:
        return torch.zeros((), device=logp.device)
    from peft import PeftModel

    if not isinstance(backend.model, PeftModel):
        return torch.zeros((), device=logp.device)
    with backend.model.disable_adapter():
        with torch.no_grad():
            ref = backend.completion_log_probs(row, completions)
    return beta * (logp - ref).mean()


def main(config_path: str | None = None) -> None:
    ap = argparse.ArgumentParser(description="MusicCaps GRPO (aspect + CLAP)")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    if config_path is None:
        args = ap.parse_args()
    else:
        args = argparse.Namespace(config=config_path)

    cfg = load_train_config(args.config)
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clap_dev = cfg.grpo_clap_device or ("cuda" if torch.cuda.is_available() else "cpu")
    clap_device = torch.device(clap_dev)
    print(
        "[grpo] run summary:\n"
        f"  config={Path(args.config).resolve()}\n"
        f"  model_id={cfg.model_id}\n"
        f"  debug_use_mock_model={cfg.debug_use_mock_model}\n"
        f"  dtype={cfg.dtype} device={device} clap_device={clap_device}\n"
        f"  lora_r={cfg.lora_r} lora_alpha={cfg.lora_alpha} lora_dropout={cfg.lora_dropout}\n"
        f"  system_prompt={SYSTEM_OMNI}\n"
        f"  user_prompt={USER_CAPTION_INSTRUCTION}"
    )

    rows = training_rows(cfg)
    if not rows:
        raise RuntimeError("No training rows (split=train). Check manifest.")
    if cfg.grpo_batch_size != 1:
        raise ValueError("This GRPO loop assumes grpo_batch_size==1 (Omni + per-sample rewards).")

    backend = build_backend(cfg, device)
    init_path = Path(cfg.checkpoint_dir) / (cfg.grpo_init_adapter or "")
    if cfg.grpo_init_adapter and init_path.is_dir() and cfg.lora_r and cfg.lora_r > 0:
        backend.load_adapter_checkpoint(init_path)
        print(f"Loaded adapter init from {init_path.resolve()}")

    clap: ClapScorer | None = None
    if cfg.reward_weight_clap > 0 and not cfg.debug_use_mock_model:
        clap = ClapScorer(cfg.clap_model_id, clap_device)

    opt = torch.optim.AdamW(
        (p for p in backend.model.parameters() if p.requires_grad),
        lr=cfg.grpo_learning_rate,
    )

    ckpt_root = Path(cfg.checkpoint_dir)
    save_dir = ckpt_root / cfg.grpo_adapter_name

    step = 0
    for epoch in range(cfg.grpo_epochs):
        random.shuffle(rows)
        bar = tqdm(range(0, len(rows), cfg.grpo_batch_size), desc=f"grpo epoch {epoch+1}/{cfg.grpo_epochs}")
        for start in bar:
            batch = rows[start : start + cfg.grpo_batch_size]
            row = batch[0]
            texts = backend.generate_group(
                row,
                group_size=cfg.grpo_group_size,
                max_new_tokens=cfg.grpo_max_new_tokens,
                temperature=cfg.grpo_temperature,
            )
            asp = [aspect_coverage_score(row.aspects, t) for t in texts]
            if clap is not None:
                clap_scores = clap.audio_text_scores([row.wav_path] * len(texts), texts)
            else:
                clap_scores = [0.0] * len(texts)
            rewards = combine_rewards(asp, clap_scores, cfg.reward_weight_aspect, cfg.reward_weight_clap)
            adv = group_advantages(rewards)
            logp = backend.completion_log_probs(row, texts)
            adv_t = torch.tensor(adv, device=logp.device, dtype=logp.dtype)
            loss_pg = -(adv_t * logp).mean()
            loss_kl = _kl_penalty(backend, logp, row, texts, cfg.beta_kl)
            loss = loss_pg + loss_kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backend.model.parameters(), cfg.grpo_max_grad_norm)
            opt.step()
            step += 1
            bar.set_postfix(
                R_mean=float(sum(rewards) / len(rewards)),
                loss=float(loss.item()),
            )
            if step % cfg.grpo_log_every == 0:
                tqdm.write(
                    f"[grpo] step={step} loss={loss.item():.4f} "
                    f"R_mean={sum(rewards)/len(rewards):.3f} sample0={texts[0][:80]!r}"
                )
            if step % cfg.grpo_save_every == 0:
                backend.save_trainable(save_dir)
                tqdm.write(f"[grpo] saved -> {save_dir}")

    backend.save_trainable(save_dir)
    print(f"GRPO done. Adapter saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
