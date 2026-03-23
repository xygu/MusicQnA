from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import ClapModel, ClapProcessor


def _mono_float32(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    return data, int(sr)


class ClapScorer:
    """
    Audio–text alignment via LAION CLAP: L2-normalized embeddings, cosine -> [0, 1].
    Processes one (wav, text) pair per call to keep processor/model kwargs unambiguous.
    """

    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.processor = ClapProcessor.from_pretrained(model_id)
        self.model = ClapModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.target_sr = int(getattr(self.processor.feature_extractor, "sampling_rate", 48000))

    @torch.inference_mode()
    def audio_text_scores(self, wav_paths: list[Path], texts: list[str]) -> list[float]:
        if len(wav_paths) != len(texts):
            raise ValueError("wav_paths and texts length mismatch")
        scores: list[float] = []
        for p, text in zip(wav_paths, texts, strict=True):
            w, sr = _mono_float32(p)
            if sr != self.target_sr:
                t = torch.from_numpy(w).float().unsqueeze(0)
                t = torchaudio_resample(t, sr, self.target_sr)
                w = t.squeeze(0).numpy()
            inputs = self.processor(
                text=[text],
                audios=[w],
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device)
            a_feat = self.model.get_audio_features(input_features=inputs["input_features"])
            t_kw = {"input_ids": inputs["input_ids"]}
            if "attention_mask" in inputs:
                t_kw["attention_mask"] = inputs["attention_mask"]
            t_feat = self.model.get_text_features(**t_kw)
            a_feat = a_feat / a_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim = (a_feat * t_feat).sum(dim=-1)
            x = float(((sim + 1.0) / 2.0).clamp(0.0, 1.0).item())
            scores.append(x)
        return scores


def torchaudio_resample(waveform: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    try:
        import torchaudio.functional as F  # type: ignore

        return F.resample(waveform, orig_sr, new_sr)
    except Exception:
        return waveform
