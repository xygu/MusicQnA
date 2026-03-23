#!/usr/bin/env python3
"""
Build manifest.jsonl from HuggingFace ``google/MusicCaps`` metadata + local wav files.

Expected wav filenames (default): ``{ytid}_{start_s}_{end_s}.wav``

Example:
  cd musiccaps_pipeline
  python scripts/build_manifest.py --wav-dir ./wavs --out-jsonl ./data/musiccaps_manifest.jsonl --require-file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _split_map(hf_split_name: str) -> str:
    s = hf_split_name.lower()
    if s in ("train", "training"):
        return "train"
    if s in ("validation", "valid", "val", "dev"):
        return "valid"
    if s in ("test", "eval"):
        return "test"
    return "train"


def _parse_aspects(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,;]", s)
    return [p.strip() for p in parts if p.strip()]


def default_wav_name(ytid: str, start_s: int, end_s: int) -> str:
    return f"{ytid}_{start_s}_{end_s}.wav"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from musiccaps.hub_mirrors import ensure_hf_cn_access

    ensure_hf_cn_access()

    from datasets import load_dataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", type=Path, required=True, help="Directory containing sliced wav files")
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="google/MusicCaps")
    ap.add_argument(
        "--require-file",
        action="store_true",
        help="Only emit rows whose wav file exists",
    )
    args = ap.parse_args()

    args.wav_dir = args.wav_dir.resolve()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset)
    n_out = 0
    n_skip = 0
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for hf_split, split_ds in ds.items():
            split = _split_map(str(hf_split))
            for row in split_ds:
                ytid = str(row["ytid"])
                start_s = int(row["start_s"])
                end_s = int(row["end_s"])
                caption = str(row["caption"]).strip()
                row_split = row.get("split")
                row_split_tag = _split_map(str(row_split)) if row_split is not None else None
                aspects_raw = row.get("aspect_list", "")
                if isinstance(aspects_raw, list):
                    aspects = [str(a).strip() for a in aspects_raw if str(a).strip()]
                else:
                    aspects = _parse_aspects(str(aspects_raw))
                fname = default_wav_name(ytid, start_s, end_s)
                wav_path = args.wav_dir / fname
                rec_id = f"{ytid}_{start_s}_{end_s}"
                if args.require_file and not wav_path.is_file():
                    n_skip += 1
                    continue
                out_split = row_split_tag if row_split_tag is not None else split
                obj = {
                    "id": rec_id,
                    "wav_path": str(wav_path),
                    "caption": caption,
                    "aspects": aspects,
                    "split": out_split,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"Wrote {n_out} lines to {args.out_jsonl} (skipped missing: {n_skip})")


if __name__ == "__main__":
    main()
