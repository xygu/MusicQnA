from __future__ import annotations

import re
from typing import Sequence

_STOP = frozenset(
    "a an the to of in on for and or is are was were be been being it at by as from".split()
)


def parse_aspect_list(s: str) -> list[str]:
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,;]", s)
    return [p.strip() for p in parts if p.strip()]


def _aspect_hit(aspect: str, caption_lower: str) -> bool:
    a = aspect.lower().strip()
    if not a:
        return True
    if a in caption_lower:
        return True
    tokens = [t for t in re.findall(r"[a-z0-9]+", a) if len(t) > 2 and t not in _STOP]
    if not tokens:
        return False
    hit = sum(1 for t in tokens if t in caption_lower)
    return hit >= max(1, (len(tokens) + 1) // 2)


def aspect_coverage_score(aspects: Sequence[str], caption: str) -> float:
    """
    Mean per-aspect hit in [0, 1]. Empty aspects -> 1.0 (no constraint).
    """
    asp = [a for a in aspects if str(a).strip()]
    if not asp:
        return 1.0
    cap = caption.lower()
    return sum(_aspect_hit(a, cap) for a in asp) / len(asp)


def combine_rewards(
    aspect_scores: Sequence[float],
    clap_scores: Sequence[float],
    w_asp: float,
    w_clap: float,
) -> list[float]:
    w_sum = w_asp + w_clap
    if w_sum <= 0:
        raise ValueError("reward weights must sum to > 0")
    asp = list(aspect_scores)
    clp = list(clap_scores)
    if len(asp) != len(clp):
        raise ValueError(
            f"aspect_scores and clap_scores length mismatch: {len(asp)} vs {len(clp)}"
        )
    return [(w_asp * a + w_clap * c) / w_sum for a, c in zip(asp, clp)]


def group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> list[float]:
    """GRPO-style normalization within one group (one prompt, G samples)."""
    rs = list(rewards)
    if not rs:
        return []
    mu = sum(rs) / len(rs)
    var = sum((x - mu) ** 2 for x in rs) / max(len(rs), 1)
    sigma = var**0.5
    return [(x - mu) / (sigma + eps) for x in rs]
