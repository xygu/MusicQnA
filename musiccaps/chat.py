from __future__ import annotations

from typing import Any

from musiccaps.prompts import SYSTEM_OMNI, USER_CAPTION_INSTRUCTION, USER_MOCK_PREFIX
from musiccaps.schema import ManifestRow


def _system_message() -> dict[str, Any]:
    return {"role": "system", "content": [{"type": "text", "text": SYSTEM_OMNI}]}


def build_omni_conversation(
    wav_path: str,
    *,
    caption: str | None = None,
    include_assistant: bool = True,
) -> list[dict[str, Any]]:
    """Chat messages for Qwen2.5-Omni processor (audio path + text)."""
    user_content: list[dict[str, Any]] = [
        {"type": "audio", "path": wav_path},
        {"type": "text", "text": USER_CAPTION_INSTRUCTION},
    ]
    conv: list[dict[str, Any]] = [
        _system_message(),
        {"role": "user", "content": user_content},
    ]
    if include_assistant and caption is not None:
        conv.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption.strip()}],
            }
        )
    return conv


def build_mock_conversation(
    row: ManifestRow,
    *,
    include_assistant: bool = True,
) -> list[dict[str, str]]:
    """Plain text chat for tiny GPT2 debug (no multimodal)."""
    user_text = USER_MOCK_PREFIX + USER_CAPTION_INSTRUCTION
    conv: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_OMNI},
        {"role": "user", "content": user_text},
    ]
    if include_assistant:
        conv.append({"role": "assistant", "content": row.caption.strip()})
    return conv
