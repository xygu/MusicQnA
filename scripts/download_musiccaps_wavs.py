#!/usr/bin/env python3
"""
Download MusicCaps source audio and slice each row to WAV.

Metadata comes from Hugging Face ``google/MusicCaps`` (``ytid``, ``start_s``, ``end_s``).
Output filenames match ``scripts/build_manifest.py``::

    {ytid}_{start_s}_{end_s}.wav

**Compliance**

You are responsible for copyright, YouTube Terms of Service, and the MusicCaps / dataset
license. This script is a technical helper only; use it only where you have the right to
access and process the content.

**Dependencies**

- ``yt-dlp`` on ``PATH`` (``pip install yt-dlp``)
- ``ffmpeg`` on ``PATH`` (system install)

Example::

  export PYTHONPATH="$(pwd)"
  # optional, same as other scripts in this repo
  # export HF_ENDPOINT=https://hf-mirror.com

  python scripts/download_musiccaps_wavs.py --out-dir ./wavs --cache-dir ./yt_audio_cache

Then build the manifest::

  python scripts/build_manifest.py --wav-dir ./wavs --out-jsonl ./data/musiccaps_manifest.jsonl --require-file
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def default_wav_name(ytid: str, start_s: int, end_s: int) -> str:
    """Keep in sync with ``scripts/build_manifest.py::default_wav_name``."""
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


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{err}")


def _cached_media_path(cache_dir: Path, ytid: str) -> Path | None:
    hits = list(cache_dir.glob(f"{ytid}.*"))
    # Ignore sidecar / partial files
    hits = [p for p in hits if p.suffix.lower() not in (".part", ".ytdl", ".temp")]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def _download_video_audio(
    yt_dlp: str,
    ytid: str,
    cache_dir: Path,
    *,
    extra_ytdlp_args: list[str],
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = _cached_media_path(cache_dir, ytid)
    if existing is not None:
        return existing

    url = f"https://www.youtube.com/watch?v={ytid}"
    out_tmpl = str(cache_dir / f"{ytid}.%(ext)s")
    cmd = [
        yt_dlp,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "-o",
        out_tmpl,
        *extra_ytdlp_args,
        url,
    ]
    _run(cmd)
    found = _cached_media_path(cache_dir, ytid)
    if found is None:
        raise RuntimeError(f"yt-dlp finished but no media file found for ytid={ytid}")
    return found


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
    ap = argparse.ArgumentParser(description="Download and slice MusicCaps clips to wav (see module docstring).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for {ytid}_{start}_{end}.wav")
    ap.add_argument("--cache-dir", type=Path, required=True, help="Per-video audio cache (reused across clips)")
    ap.add_argument("--dataset", type=str, default="google/MusicCaps")
    ap.add_argument(
        "--splits",
        type=str,
        default="",
        help="Comma-separated: train,valid,test (HF split names mapped like build_manifest). Empty = all.",
    )
    ap.add_argument("--max-clips", type=int, default=None, help="Stop after this many dataset rows (debug).")
    ap.add_argument("--workers", type=int, default=4, help="Parallel ffmpeg slice jobs.")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--mono", action="store_true", help="Output mono wav (default: stereo).")
    ap.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-slice even if the output wav already exists.",
    )
    ap.add_argument(
        "--ytdlp-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra argument passed to yt-dlp (repeatable), e.g. --ytdlp-arg --cookies cookies.txt",
    )
    ap.add_argument(
        "--failures-jsonl",
        type=Path,
        default=None,
        help="Append one JSON object per failed clip (ytid, start_s, end_s, error).",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    yt_dlp = _which_or_exit("yt-dlp")
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
    by_ytid: dict[str, list[tuple[int, int, Path]]] = defaultdict(list)
    for row in raw_rows:
        ytid = str(row["ytid"])
        start_s = int(row["start_s"])
        end_s = int(row["end_s"])
        out_wav = out_dir / default_wav_name(ytid, start_s, end_s)
        by_ytid[ytid].append((start_s, end_s, out_wav))

    skip_existing = not args.no_skip_existing
    n_skipped = 0
    n_need_clips = 0
    for ytid, clips in by_ytid.items():
        for start_s, end_s, out_wav in clips:
            if skip_existing and out_wav.is_file():
                n_skipped += 1
                continue
            n_need_clips += 1

    # Resolve cache paths per ytid (download sequential to reduce throttling)
    ytid_to_src: dict[str, Path] = {}
    failures: list[dict[str, Any]] = []
    print(f"Unique videos: {len(by_ytid)}; clips needed: {n_need_clips} (skipped existing: {n_skipped})")

    for i, ytid in enumerate(sorted(by_ytid.keys()), start=1):
        need_any = False
        for start_s, end_s, out_wav in by_ytid[ytid]:
            if skip_existing and out_wav.is_file():
                continue
            need_any = True
            break
        if not need_any:
            continue
        try:
            ytid_to_src[ytid] = _download_video_audio(
                yt_dlp, ytid, cache_dir, extra_ytdlp_args=list(args.ytdlp_arg)
            )
            print(f"[{i}/{len(by_ytid)}] cached audio ytid={ytid} -> {ytid_to_src[ytid].name}")
        except Exception as e:
            msg = str(e)
            print(f"[{i}/{len(by_ytid)}] download failed ytid={ytid}: {msg}", file=sys.stderr)
            for start_s, end_s, out_wav in by_ytid[ytid]:
                if skip_existing and out_wav.is_file():
                    continue
                failures.append(
                    {
                        "ytid": ytid,
                        "start_s": start_s,
                        "end_s": end_s,
                        "stage": "download",
                        "error": msg,
                    }
                )

    jobs: list[tuple[str, Path, int, int, Path]] = []
    for ytid, clips in by_ytid.items():
        src = ytid_to_src.get(ytid)
        if src is None:
            continue
        for start_s, end_s, out_wav in clips:
            if skip_existing and out_wav.is_file():
                continue
            jobs.append((ytid, src, start_s, end_s, out_wav))

    ch = 1 if args.mono else 2
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
        f"skipped existing: {n_skipped}; outputs under {out_dir}"
    )
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
