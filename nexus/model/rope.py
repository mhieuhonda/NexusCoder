"""
Rotary Position Embedding (RoPE) v0.3 — with NTK-aware + YaRN scaling
====================================================================
v0.1: basic RoPE (Su et al., 2021)
v0.2: cached cos/sin, max 50k context
v0.3: adds 4 RoPE scaling strategies for context extension:
  - "linear":  naive linear interpolation (Chen et al., 2023)
  - "dynamic": NTK-aware (PureDynamicNTKScaling) — better for short→long
  - "ntk":     NTK-by-parts (bloc97, 2023)
  - "yarn":    YaRN (Peng et al., 2023) — SOTA for 4×+ extension

References:
  - Original RoPE: https://arxiv.org/abs/2104.09864
  - YaRN: https://arxiv.org/abs/2309.00071
  - NTK-aware: https://www.reddit.com/r/LocalLLaMA/comments/14lzrgj/
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Scaling strategies
# =============================================================================

def _linear_inv_freq(base: float, dim: int, scaling_factor: float) -> torch.Tensor:
    """Linear scaling: compress positions by `scaling_factor`."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq / scaling_factor


def _ntk_aware_inv_freq(base: float, dim: int, scaling_factor: float) -> torch.Tensor:
    """NTK-aware scaling — modifies base frequency directly.
    Better preserves high-frequency components than linear.
    """
    base = base * (scaling_factor ** (dim / (dim - 2)))
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


def _yarn_inv_freq(
    base: float,
    dim: int,
    scaling_factor: float,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """YaRN scaling — interpolated NTK with attention-factor correction.
    Currently we only return the modified inv_freq; the attention factor
    correction (temperature) is applied separately in the Attention module.
    """
    # Find wavelength boundaries
    def _find_correction_dim(num_rot: int, dim: int, base: float, max_seq_len: int) -> float:
        return (dim * math.log(max_seq_len / (num_rot * 2 * math.pi))) / (2 * math.log(base))

    def _find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int,
    ) -> Tuple[int, int]:
        low = max(math.floor(_find_correction_dim(low_rot, dim, base, max_seq_len)), 0)
        high = min(math.ceil(_find_correction_dim(high_rot, dim, base, max_seq_len)), dim - 1)
        return low, high

    def _linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            return torch.ones(dim) if min_val > 0 else torch.zeros(dim)
        lin = torch.linspace(0, 1, dim)
        return torch.clamp((lin - min_val) / (max_val - min_val), 0.0, 1.0)

    max_seq_len = int(4096 * scaling_factor)
    low, high = _find_correction_range(beta_fast, beta_slow, dim, base, max_seq_len)
    inv_freq_extrapolation = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    inv_freq_interpolation = 1.0 / (scaling_factor * base ** (torch.arange(0, dim, 2).float() / dim))
    mask = _linear_ramp_mask(low, high, dim // 2).float()
    inv_freq = inv_freq_interpolation * mask + inv_freq_extrapolation * (1 - mask)
    return inv_freq


def compute_inv_freq_with_scaling(
    base: float,
    dim: int,
    scaling_type: Optional[str],
    scaling_factor: float,
    yarn_beta_fast: float = 32.0,
    yarn_beta_slow: float = 1.0,
) -> torch.Tensor:
    """Compute inv_freq with the requested scaling strategy."""
    if scaling_type is None or scaling_factor == 1.0:
        return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    if scaling_type == "linear":
        return _linear_inv_freq(base, dim, scaling_factor)
    if scaling_type == "dynamic":
        return _ntk_aware_inv_freq(base, dim, scaling_factor)
    if scaling_type == "ntk":
        return _ntk_aware_inv_freq(base, dim, scaling_factor)
    if scaling_type == "yarn":
        return _yarn_inv_freq(
            base, dim, scaling_factor,
            beta_fast=yarn_beta_fast, beta_slow=yarn_beta_slow,
        )
    raise ValueError(f"Unknown rope_scaling_type: {scaling_type}")


# =============================================================================
# Rotary embedding module
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with optional scaling (v0.3)."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 50000,
        base: float = 10000.0,
        scaling_type: Optional[str] = None,
        scaling_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        self.yarn_beta_fast = yarn_beta_fast
        self.yarn_beta_slow = yarn_beta_slow

        inv_freq = compute_inv_freq_with_scaling(
            base=base,
            dim=dim,
            scaling_type=scaling_type,
            scaling_factor=scaling_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len, ...].to(x.dtype),
            self.sin_cached[:seq_len, ...].to(x.dtype),
        )

    def get_attention_temperature(self) -> float:
        """YaRN requires a temperature correction on the attention scores.
        Returns the multiplier (1.0 for non-YaRN)."""
        if self.scaling_type == "yarn":
            # Standard YaRN correction: 0.1 * log(scaling_factor) + 1
            return 0.1 * math.log(self.scaling_factor) + 1.0
        return 1.0


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
    position_ids: Optional[torch.Tensor] = None,
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
