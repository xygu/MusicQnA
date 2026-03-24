from __future__ import annotations

from pathlib import Path

from musiccaps.config import TrainConfig
from musiccaps.schema import load_manifest_jsonl, rows_split


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
        missing_rows = []
        existing_rows = []
        for r in rows:
            if r.wav_path.is_file():
                existing_rows.append(r)
            else:
                missing_rows.append(r)
        if missing_rows:
            if cfg.skip_missing_wavs:
                print(
                    f"[dataset] skip_missing_wavs=True, skipped {len(missing_rows)} rows "
                    f"with missing wav; kept {len(existing_rows)} rows."
                )
                rows = existing_rows
            else:
                first = missing_rows[0]
                raise FileNotFoundError(
                    f"Missing wav for id={first.id}: {first.wav_path}. "
                    "Set skip_missing_wavs=true to skip missing audio rows."
                )
    if cfg.max_samples is not None:
        rows = rows[: cfg.max_samples]
    return rows


def training_rows(cfg: TrainConfig) -> list[ManifestRow]:
    return rows_split(load_rows(cfg), "train")
