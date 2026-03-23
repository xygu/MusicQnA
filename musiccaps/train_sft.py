from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from musiccaps.config import TrainConfig, load_train_config
from musiccaps.dataset import training_rows
from musiccaps.lm_backend import build_backend


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str | None = None) -> None:
    ap = argparse.ArgumentParser(description="MusicCaps SFT")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    if config_path is None:
        args = ap.parse_args()
    else:
        args = argparse.Namespace(config=config_path)

    cfg = load_train_config(args.config)
    _seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = training_rows(cfg)
    if not rows:
        raise RuntimeError("No training rows (split=train). Check manifest and splits.")

    backend = build_backend(cfg, device)
    opt = torch.optim.AdamW(
        (p for p in backend.model.parameters() if p.requires_grad),
        lr=cfg.sft_learning_rate,
    )

    ckpt_root = Path(cfg.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    save_dir = ckpt_root / cfg.sft_adapter_name

    step = 0
    for epoch in range(cfg.sft_epochs):
        random.shuffle(rows)
        bar = tqdm(range(0, len(rows), cfg.sft_batch_size), desc=f"sft epoch {epoch+1}/{cfg.sft_epochs}")
        for start in bar:
            batch = rows[start : start + cfg.sft_batch_size]
            opt.zero_grad(set_to_none=True)
            loss = backend.supervised_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(backend.model.parameters(), cfg.sft_max_grad_norm)
            opt.step()
            step += 1
            bar.set_postfix(loss=float(loss.item()))
            if step % cfg.sft_log_every == 0:
                tqdm.write(f"[sft] step={step} loss={loss.item():.4f}")
            if step % cfg.sft_save_every == 0:
                backend.save_trainable(save_dir)
                tqdm.write(f"[sft] saved -> {save_dir}")

    backend.save_trainable(save_dir)
    print(f"SFT done. Adapter saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
