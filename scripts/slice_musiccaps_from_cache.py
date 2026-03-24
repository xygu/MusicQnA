#!/usr/bin/env python3
"""
Slice MusicCaps clips to WAV using ONLY existing local cache files.

No yt-dlp download is attempted in this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def default_wav_name(ytid: str, start_s: int, end_s: int) -> str:
    return f"{ytid}_{start_s}_{end_s}.wav"


def _split_map(hf_split_name: str) -> str:
    s = hf_split_name.lower()
    if s in ("train", "training"):
        return "train"
    if s in ("validation", "valid", "val", "dev"):
        return "valid"
    if s in ("test", "eval"):
        return "test"
    return "train"


def _which_or_exit(name: str) -> str:
    p = shutil.which(name)
    if not p:
        print(f"error: `{name}` not found on PATH (install it and retry)", file=sys.stderr)
        sys.exit(1)
    return p


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{err}")


def _cached_media_path(cache_dir: Path, ytid: str) -> Path | None:
    hits = list(cache_dir.glob(f"{ytid}.*"))
    hits = [p for p in hits if p.suffix.lower() not in (".part", ".ytdl", ".temp")]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _slice_wav(
    ffmpeg: str,
    src: Path,
    start_s: int,
    end_s: int,
    out_wav: Path,
    *,
    sample_rate: int,
    audio_channels: int,
) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end_s - start_s))
    if duration <= 0:
        raise ValueError("end_s must be greater than start_s")
    ac = "1" if audio_channels == 1 else "2"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(float(start_s)),
        "-i",
        str(src),
        "-t",
        str(duration),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        ac,
        str(out_wav),
    ]
    _run(cmd)


def _load_rows(
    dataset_id: str,
    *,
    only_splits: set[str] | None,
    max_clips: int | None,
) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from musiccaps.hub_mirrors import ensure_hf_cn_access

    ensure_hf_cn_access()
    from datasets import load_dataset

    ds = load_dataset(dataset_id)
    rows: list[dict[str, Any]] = []
    for hf_split, split_ds in ds.items():
        split_tag = _split_map(str(hf_split))
        if only_splits is not None and split_tag not in only_splits:
            continue
        for row in split_ds:
            rows.append(dict(row))
            if max_clips is not None and len(rows) >= max_clips:
                return rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice MusicCaps clips from existing cache only.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for {ytid}_{start}_{end}.wav")
    ap.add_argument("--cache-dir", type=Path, required=True, help="Per-video audio cache (reused across clips)")
    ap.add_argument("--dataset", type=str, default="google/MusicCaps")
    ap.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated: train,valid,test. Empty = all.",
    )
    ap.add_argument("--max-clips", type=int, default=None, help="Stop after this many dataset rows (debug).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel ffmpeg slice jobs.")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--mono", action="store_true", help="Output mono wav (default: stereo).")
    ap.add_argument("--no-skip-existing", action="store_true", help="Re-slice even if output wav exists.")
    ap.add_argument(
        "--failures-jsonl",
        type=Path,
        default=None,
        help="Append one JSON object per failed clip (ytid, start_s, end_s, error).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    ffmpeg = _which_or_exit("ffmpeg")

    only_splits: set[str] | None = None
    if args.splits.strip():
        only_splits = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
        allowed = {"train", "valid", "test"}
        bad = only_splits - allowed
        if bad:
            print(f"error: unknown split(s) {bad}; use train,valid,test", file=sys.stderr)
            sys.exit(1)

    raw_rows = _load_rows(args.dataset, only_splits=only_splits, max_clips=args.max_clips)
    skip_existing = not args.no_skip_existing
    ch = 1 if args.mono else 2

    jobs: list[tuple[str, Path, int, int, Path]] = []
    failures: list[dict[str, Any]] = []
    n_skipped_existing = 0
    n_missing_cache = 0

    for row in raw_rows:
        ytid = str(row["ytid"])
        start_s = int(row["start_s"])
        end_s = int(row["end_s"])
        out_wav = out_dir / default_wav_name(ytid, start_s, end_s)
        if skip_existing and out_wav.is_file():
            n_skipped_existing += 1
            continue
        src = _cached_media_path(cache_dir, ytid)
        if src is None:
            n_missing_cache += 1
            failures.append(
                {
                    "ytid": ytid,
                    "start_s": start_s,
                    "end_s": end_s,
                    "stage": "cache_lookup",
                    "error": "missing cache media for ytid",
                }
            )
            continue
        jobs.append((ytid, src, start_s, end_s, out_wav))

    print(
        f"Rows: {len(raw_rows)}; jobs: {len(jobs)}; missing cache: {n_missing_cache}; "
        f"skipped existing: {n_skipped_existing}"
    )

    n_ok = 0

    def work(item: tuple[str, Path, int, int, Path]) -> tuple[bool, dict[str, Any] | None]:
        ytid, src, start_s, end_s, out_wav = item
        try:
            _slice_wav(
                ffmpeg,
                src,
                start_s,
                end_s,
                out_wav,
                sample_rate=args.sample_rate,
                audio_channels=ch,
            )
            return True, None
        except Exception as e:
            return False, {
                "ytid": ytid,
                "start_s": start_s,
                "end_s": end_s,
                "stage": "slice",
                "error": str(e),
                "wav": str(out_wav),
            }

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for fut in as_completed(futs):
                ok, info = fut.result()
                if ok:
                    n_ok += 1
                elif info:
                    failures.append(info)
                    print(f"slice failed: {info.get('wav')} {info.get('error')}", file=sys.stderr)

    if args.failures_jsonl is not None:
        args.failures_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.failures_jsonl.open("a", encoding="utf-8") as f:
            for rec in failures:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"Done. Sliced ok: {n_ok}; failures: {len(failures)}; "
        f"missing cache: {n_missing_cache}; skipped existing: {n_skipped_existing}; outputs under {out_dir}"
    )
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
