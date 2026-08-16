"""
Rotary Position Embedding (RoPE) cho Nexus Coder
================================================
Hỗ trợ context window 50k tokens với RoPE base tùy chỉnh.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 50000,
        base: float = 10000.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Pre-compute inv_freq
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len, ...].to(x.dtype),
            self.sin_cached[:seq_len, ...].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Xoay một nửa tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Áp dụng RoPE cho q và k."""
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
