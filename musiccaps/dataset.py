from __future__ import annotations

from pathlib import Path

from musiccaps.config import TrainConfig
from musiccaps.schema import ManifestRow, load_manifest_jsonl, rows_split


def load_rows(cfg: TrainConfig) -> list[ManifestRow]:
    mp = Path(cfg.manifest_path)
    if not mp.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {mp.resolve()}. "
            "Run scripts/build_manifest.py after placing wav files."
        )
    base = mp.parent
    rows = load_manifest_jsonl(mp, base_dir=base)
    if not cfg.debug_use_mock_model:
        for r in rows:
            if not r.wav_path.is_file():
                raise FileNotFoundError(f"Missing wav for id={r.id}: {r.wav_path}")
    if cfg.max_samples is not None:
        rows = rows[: cfg.max_samples]
    return rows


def training_rows(cfg: TrainConfig) -> list[ManifestRow]:
    return rows_split(load_rows(cfg), "train")
