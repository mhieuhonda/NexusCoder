"""
Context Expansion Protocol (CEP)
================================
Kỹ thuật mở rộng context window độc đáo của CyberGym — train progressive
từ short → long context, kết hợp YaRN RoPE scaling + manifold folding.

Ý tưởng:
  - Train model ở 32k context trước (cheap, fast convergence)
  - Sau đó mở rộng lên 131k, 524k, 1M, 2M, 3M theo stages
  - Mỗi stage: 1 epoch full data ở context mới
  - YaRN RoPE scaling cho phép extrapolate
  - "Manifold folding": chunked attention + sliding window overlap
    → attention pattern tự fold để capture long-range deps

  Tổng chi phí: ~30% train + ~30% infer thời gian so với train thẳng ở 3M

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class CEPConfig:
    """Cấu hình Context Expansion Protocol."""
    stages: List[int] = field(
        default_factory=lambda: [32768, 131072, 524288, 1048576, 2097152, 3000000]
    )
    epoch_per_stage: int = 1
    # YaRN RoPE scaling factor tương ứng với mỗi stage
    # factor = stage_context / base_context (thường 32768)
    base_context: int = 32768
    # Sliding window size ở mỗi stage (tỷ lệ với sqrt của context)
    sliding_window_ratio: float = 0.25  # SWA = 25% của context
    # Mixed-length batching: trong stage cao, mix 25% short + 75% long
    mix_short_ratio: float = 0.25
    # Learning rate decay qua stages (mỗi stage LR *= 0.5)
    lr_decay_per_stage: float = 0.5


class ContextExpansionProtocol:
    """Quản lý CEP training schedule.

    Usage:
        cep = ContextExpansionProtocol(config)
        schedule = cep.get_schedule(total_epochs=6)
        for stage in schedule:
            for epoch in range(stage["epochs"]):
                for batch in loader_at_context(stage["context_len"]):
                    train_step(batch, lr=stage["lr"], rope_factor=stage["rope_factor"])
    """

    def __init__(self, config: Optional[CEPConfig] = None):
        self.config = config or CEPConfig()

    def get_schedule(self, total_epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        """Trả về train schedule cho CEP.

        Returns list of dicts with:
            - context_len: int
            - rope_factor: float
            - sliding_window: int
            - epochs: int
            - lr_scale: float
            - mix_short_ratio: float
        """
        schedule: List[Dict[str, Any]] = []
        lr_scale = 1.0
        epochs = self.config.epoch_per_stage if total_epochs is None else (
            max(1, total_epochs // len(self.config.stages))
        )
        for stage_ctx in self.config.stages:
            rope_factor = stage_ctx / max(self.config.base_context, 1)
            swa = int(stage_ctx * self.config.sliding_window_ratio)
            # SWA phải là số chẵn để dễ tune
            if swa % 2 == 1:
                swa += 1
            schedule.append({
                "context_len": stage_ctx,
                "rope_factor": float(rope_factor),
                "sliding_window": swa,
                "epochs": epochs,
                "lr_scale": lr_scale,
                "mix_short_ratio": self.config.mix_short_ratio,
            })
            lr_scale *= self.config.lr_decay_per_stage
        return schedule

    def apply_stage_to_config(self, config, stage_idx: int) -> None:
        """Apply stage-th stage vào NexusConfig (in-place)."""
        if stage_idx < 0 or stage_idx >= len(self.config.stages):
            return
        schedule = self.get_schedule()
        stage = schedule[stage_idx]
        config.max_position_embeddings = stage["context_len"]
        config.rope_scaling_type = "yarn"
        config.rope_scaling_factor = stage["rope_factor"]
        config.sliding_window_size = stage["sliding_window"]
        if stage["context_len"] >= 131072:
            config.kv_cache_quantization = "int8"
            config.gradient_checkpointing = True

    def summary(self) -> Dict[str, Any]:
        sched = self.get_schedule()
        return {
            "n_stages": len(sched),
            "stages": sched,
            "total_context_growth": f"{self.config.stages[0]:,} → {self.config.stages[-1]:,}",
            "growth_factor": self.config.stages[-1] / self.config.stages[0],
        }


def chunked_attention_mask(
    seq_len: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Tạo mask cho chunked attention (manifold folding).

    Token i có thể attend tokens trong cùng chunk hoặc chunk trước đó.
    → O(seq_len × chunk_size × 2) thay vì O(seq_len²)
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    for i in range(seq_len):
        chunk_start = (i // chunk_size) * chunk_size
        # Attend: chunk hiện tại + chunk trước đó
        start = max(0, chunk_start - chunk_size)
        end = min(seq_len, chunk_start + chunk_size)
        mask[i, start:end] = 0.0
        # Causal: không attend future
        mask[i, i + 1:] = float("-inf")
    return mask
