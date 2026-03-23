"""
Hugging Face Hub: default to a mainland-China-friendly endpoint when the user has not set one.

``huggingface_hub`` / ``transformers`` / ``datasets`` read ``HF_ENDPOINT``. Official default is
https://huggingface.co ; many CN users use https://hf-mirror.com .

- Set ``HF_ENDPOINT`` before any Hub I/O to force a specific endpoint (this module uses
  ``setdefault`` and will not override).
- Set ``MUSICCAPS_NO_CN_MIRROR=1`` (or ``true``) to skip applying the default mirror entirely.
"""

from __future__ import annotations

import os

_DEFAULT_CN_HF_ENDPOINT = "https://hf-mirror.com"


def ensure_hf_cn_access() -> None:
    if os.environ.get("MUSICCAPS_NO_CN_MIRROR", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    os.environ.setdefault("HF_ENDPOINT", _DEFAULT_CN_HF_ENDPOINT)
