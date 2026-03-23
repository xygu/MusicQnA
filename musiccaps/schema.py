from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

Split = Literal["train", "valid", "test"]


@dataclass(frozen=True)
class ManifestRow:
    """One training/eval example. All paths resolved at load time."""

    id: str
    wav_path: Path
    caption: str
    aspects: tuple[str, ...]
    split: Split

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "wav_path": str(self.wav_path),
            "caption": self.caption,
            "aspects": list(self.aspects),
            "split": self.split,
        }


def _as_split(s: str) -> Split:
    x = s.strip().lower()
    if x in ("train", "training"):
        return "train"
    if x in ("valid", "val", "validation", "dev"):
        return "valid"
    if x in ("test", "eval"):
        return "test"
    raise ValueError(f"Unknown split: {s!r}")


def manifest_row_from_dict(obj: dict[str, Any], base_dir: Path | None = None) -> ManifestRow:
    mid = str(obj["id"])
    wp = Path(obj["wav_path"])
    if not wp.is_absolute() and base_dir is not None:
        wp = (base_dir / wp).resolve()
    cap = str(obj["caption"])
    aspects_raw = obj.get("aspects", [])
    if isinstance(aspects_raw, str):
        aspects = tuple(a.strip() for a in aspects_raw.split(",") if a.strip())
    else:
        aspects = tuple(str(a).strip() for a in aspects_raw if str(a).strip())
    split = _as_split(str(obj.get("split", "train")))
    return ManifestRow(
        id=mid,
        wav_path=wp,
        caption=cap,
        aspects=aspects,
        split=split,
    )


def load_manifest_jsonl(path: Path, base_dir: Path | None = None) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rows.append(manifest_row_from_dict(obj, base_dir=base_dir))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"{path}:{line_no}: bad manifest line: {e}") from e
    return rows


def rows_split(rows: Sequence[ManifestRow], split: Split) -> list[ManifestRow]:
    return [r for r in rows if r.split == split]
