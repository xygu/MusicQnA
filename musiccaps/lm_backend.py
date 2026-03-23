from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from musiccaps.chat import build_mock_conversation, build_omni_conversation
from musiccaps.config import TrainConfig
from musiccaps.schema import ManifestRow


def _pick_dtype(name: str) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available():
        return torch.bfloat16
    if name == "float16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


class CaptionLMBackend(ABC):
    device: torch.device
    model: nn.Module

    @abstractmethod
    def supervised_loss(self, rows: list[ManifestRow]) -> torch.Tensor:
        """Mean CE over batch (caller does backward)."""

    @abstractmethod
    def generate_group(self, row: ManifestRow, *, group_size: int, max_new_tokens: int, temperature: float) -> list[str]:
        """G independent samples for one row."""

    @abstractmethod
    def completion_log_probs(
        self,
        row: ManifestRow,
        completions: list[str],
    ) -> torch.Tensor:
        """Shape [G] — sum of log pi over generated tokens for each completion."""

    @abstractmethod
    def save_trainable(self, path: Path) -> None:
        ...

    def load_adapter_checkpoint(self, path: Path) -> None:
        """Optional: load PEFT weights from ``save_trainable`` output directory."""
        from peft import PeftModel

        if not path.is_dir():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")
        if isinstance(self.model, PeftModel):
            self.model.load_adapter(str(path))
            return
        raise TypeError("load_adapter_checkpoint only supported for PEFT-wrapped models")


class OmniThinkerBackend(CaptionLMBackend):
    def __init__(self, cfg: TrainConfig, device: torch.device):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        self.cfg = cfg
        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(cfg.model_id)
        dtype = _pick_dtype(cfg.dtype)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            device_map=None,
        )
        self.model.to(device)
        if cfg.lora_r and cfg.lora_r > 0:
            lc = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lc)  # type: ignore[assignment]
        self.model.train()

    def _apply_template(self, conversations: list[dict[str, Any]], *, add_generation_prompt: bool) -> dict[str, torch.Tensor]:
        kwargs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        out = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return out

    def _prompt_inputs(self, row: ManifestRow) -> dict[str, torch.Tensor]:
        conv = build_omni_conversation(str(row.wav_path.resolve()), caption=None, include_assistant=False)
        return self._apply_template(conv, add_generation_prompt=True)

    def _supervised_inputs(self, row: ManifestRow) -> dict[str, torch.Tensor]:
        conv = build_omni_conversation(str(row.wav_path.resolve()), caption=row.caption, include_assistant=True)
        full = self._apply_template(conv, add_generation_prompt=False)
        prompt_conv = build_omni_conversation(str(row.wav_path.resolve()), caption=None, include_assistant=False)
        prompt = self._apply_template(prompt_conv, add_generation_prompt=True)
        pl = int(prompt["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        labels[:, :pl] = -100
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels = labels.masked_fill(full["input_ids"] == pad_id, -100)
        full["labels"] = labels
        return full

    def supervised_loss(self, rows: list[ManifestRow]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for row in rows:
            batch = self._supervised_inputs(row)
            out = self.model(**batch)
            losses.append(out.loss)
        return torch.stack(losses).mean()

    @torch.inference_mode()
    def generate_group(
        self,
        row: ManifestRow,
        *,
        group_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        self.model.eval()
        try:
            inputs = self._prompt_inputs(row)
            tok = self.processor.tokenizer
            pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                num_return_sequences=group_size,
                pad_token_id=pad_id,
            )
            # Strip prompt: each row shares same prompt length
            pl = inputs["input_ids"].shape[1]
            gen_part = out_ids[:, pl:]
            texts = self.processor.batch_decode(gen_part, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return [t.strip() for t in texts]
        finally:
            self.model.train()

    def completion_log_probs(self, row: ManifestRow, completions: list[str]) -> torch.Tensor:
        """Teacher-forced log prob on assistant tokens only (per completion)."""
        self.model.train()
        logps: list[torch.Tensor] = []
        prompt = self._prompt_inputs(row)
        prompt_ids = prompt["input_ids"]
        pl = prompt_ids.shape[1]
        for text in completions:
            conv = build_omni_conversation(str(row.wav_path.resolve()), caption=text, include_assistant=True)
            full = self._apply_template(conv, add_generation_prompt=False)
            full_ids = full["input_ids"]
            labels = full["input_ids"].clone()
            labels[:, :pl] = -100
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                labels = labels.masked_fill(full["input_ids"] == pad_id, -100)
            labels[:, -1] = -100
            out = self.model(**{k: v for k, v in full.items()})
            logits = out.logits.float()[:, :-1, :]
            targets = full["input_ids"][:, 1:]
            mask = labels[:, 1:].ne(-100)
            lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            logps.append((lp * mask.float()).sum(dim=-1))
        return torch.cat(logps, dim=0)

    def save_trainable(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)


class TinyGpt2Backend(CaptionLMBackend):
    """Text-only smoke test: ignores wav_path content."""

    def __init__(self, cfg: TrainConfig, device: torch.device):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.cfg = cfg
        self.device = device
        self.tokenizer = GPT2TokenizerFast.from_pretrained("sshleifer/tiny-gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2").to(device)
        if cfg.lora_r and cfg.lora_r > 0:
            from peft import LoraConfig, TaskType, get_peft_model

            lc = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=["c_attn", "c_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lc)  # type: ignore[assignment]
        self.model.train()

    def _encode_supervised(self, row: ManifestRow) -> tuple[torch.Tensor, torch.Tensor]:
        conv = build_mock_conversation(row, include_assistant=True)
        sys_t = conv[0]["content"]
        usr_t = conv[1]["content"]
        asst_t = conv[2]["content"]
        prompt = f"### System\n{sys_t}\n### User\n{usr_t}\n### Assistant\n"
        full = prompt + asst_t + self.tokenizer.eos_token
        p_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        f_ids = self.tokenizer.encode(full, return_tensors="pt", add_special_tokens=False)
        return p_ids.to(self.device), f_ids.to(self.device)

    def supervised_loss(self, rows: list[ManifestRow]) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for row in rows:
            p_ids, f_ids = self._encode_supervised(row)
            pl = p_ids.shape[1]
            labels = f_ids.clone()
            labels[:, :pl] = -100
            out = self.model(input_ids=f_ids, labels=labels)
            losses.append(out.loss)
        return torch.stack(losses).mean()

    def _prompt_ids(self, row: ManifestRow) -> torch.Tensor:
        conv = build_mock_conversation(row, include_assistant=False)
        sys_t = conv[0]["content"]
        usr_t = conv[1]["content"]
        prompt = f"### System\n{sys_t}\n### User\n{usr_t}\n### Assistant\n"
        return self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

    @torch.inference_mode()
    def generate_group(
        self,
        row: ManifestRow,
        *,
        group_size: int,
        max_new_tokens: int,
        temperature: float,
    ) -> list[str]:
        self.model.eval()
        try:
            p_ids = self._prompt_ids(row)
            out_ids = self.model.generate(
                p_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-5),
                num_return_sequences=group_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            pl = p_ids.shape[1]
            gen = out_ids[:, pl:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            return [t.strip() for t in texts]
        finally:
            self.model.train()

    def completion_log_probs(self, row: ManifestRow, completions: list[str]) -> torch.Tensor:
        self.model.train()
        p_ids = self._prompt_ids(row)
        pl = p_ids.shape[1]
        logps: list[torch.Tensor] = []
        for text in completions:
            asst = text + self.tokenizer.eos_token
            c_ids = self.tokenizer.encode(asst, return_tensors="pt", add_special_tokens=False).to(self.device)
            full = torch.cat([p_ids, c_ids], dim=-1)
            labels = full.clone()
            labels[:, :pl] = -100
            labels[:, -1] = -100
            out = self.model(input_ids=full)
            logits = out.logits.float()[:, :-1, :]
            targets = full[:, 1:]
            mask = labels[:, 1:].ne(-100)
            lp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            logps.append((lp * mask.float()).sum(dim=-1))
        return torch.cat(logps, dim=0)

    def save_trainable(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)


def build_backend(cfg: TrainConfig, device: torch.device) -> CaptionLMBackend:
    if cfg.debug_use_mock_model:
        return TinyGpt2Backend(cfg, device)
    return OmniThinkerBackend(cfg, device)
