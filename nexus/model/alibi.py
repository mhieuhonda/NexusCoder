"""
ALiBi (Attention with Linear Biases) position bias for Nexus Coder v0.3
======================================================================
Alternative to RoPE. No positional embeddings — biases are added directly
to attention scores. Extrapolates better to longer sequences than RoPE.

Reference: Press et al., "Train Short, Test Long: Attention with Linear
Biases Enables Input Length Extrapolation" (ICLR 2022).
https://arxiv.org/abs/2108.12409

Attribution: Algorithm adapted from the original paper. Implementation
references both the original alibi-transformers repo and HuggingFace's
integration in `bloom` / `mntptr` projects.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def get_alibi_slopes(num_heads: int, max_slope: float = 8.0) -> torch.Tensor:
    """Compute ALiBi slopes for `num_heads` attention heads.

    v0.4 fix: use `max_slope` correctly (was hardcoded to 8.0 → log2(8)=3).
    v0.4 fix: non-power-of-2 head counts now pick the *closest* n slopes
              (standard ALiBi behavior), not "evenly spaced" (which was buggy).

    Args:
        num_heads: number of attention heads
        max_slope: steepest slope (controls decay). Default 8.0.

    Returns:
        slopes: tensor of shape [num_heads]
    """
    if num_heads <= 0:
        return torch.tensor([], dtype=torch.float32)

    log_max = math.log2(max_slope)  # e.g. log2(8)=3

    def _get_slopes_power_of_2(n: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - log_max)))
        return [start * (2.0 ** (-i)) for i in range(n)]

    if (num_heads & (num_heads - 1)) == 0:
        # Power of 2 — direct
        slopes = _get_slopes_power_of_2(num_heads)
    else:
        # Non-power-of-2: standard ALiBi picks the n closest slopes
        # by computing slopes for the nearest power of 2 >= n and
        # interleaving them, then taking the first n.
        base = 1
        while base < num_heads:
            base *= 2
        full = _get_slopes_power_of_2(base)
        # Interleave: take even-indexed first, then odd, to pick "closest" slopes
        interleaved = (
            [full[i] for i in range(0, base, 2)]
            + [full[i] for i in range(1, base, 2)]
        )
        slopes = interleaved[:num_heads]

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_tensor(
    num_heads: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    max_slope: float = 8.0,
) -> torch.Tensor:
    """Build the additive ALiBi bias tensor.

    Args:
        num_heads: number of attention heads
        seq_len: attention sequence length
        device: target device
        dtype: target dtype
        max_slope: maximum slope (controls decay)

    Returns:
        alibi: tensor of shape [1, num_heads, seq_len, seq_len]
               Ready to ADD to attention weights before softmax.
    """
    slopes = get_alibi_slopes(num_heads, max_slope=max_slope).to(device=device, dtype=dtype)
    # positions: [seq_len, seq_len], value = j - i (j is query, i is key)
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    relative_positions = positions[None, :] - positions[:, None]  # [T, T]
    # Mask future positions to -inf (handled by causal mask elsewhere, but be safe)
    relative_positions = relative_positions.clamp(min=0)
    # alibi: [num_heads, seq_len, seq_len] = -slope * relative_positions
    alibi = slopes.view(-1, 1, 1) * relative_positions.unsqueeze(0)
    alibi = -alibi  # bias is negative (decreases attention with distance)
    # Add batch dim
    alibi = alibi.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]
    return alibi.to(dtype=dtype)


class AlibiPositionBias(nn.Module):
    """Module wrapper for ALiBi bias — registered as buffer, recomputed if seq_len grows."""

    def __init__(self, num_heads: int, max_slope: float = 8.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_slope = max_slope
        slopes = get_alibi_slopes(num_heads, max_slope=max_slope)
        self.register_buffer("slopes", slopes, persistent=False)
        self._cached_seq_len = 0
        self._cached_bias: torch.Tensor | None = None

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return ALiBi bias of shape [1, num_heads, seq_len, seq_len]."""
        if self._cached_bias is None or seq_len > self._cached_seq_len:
            self._cached_bias = build_alibi_tensor(
                self.num_heads, seq_len, device=device, dtype=dtype, max_slope=self.max_slope,
            )
            self._cached_seq_len = seq_len
        bias = self._cached_bias.to(device=device, dtype=dtype)
        if bias.shape[-1] < seq_len:
            # Re-build for new length
            self._cached_bias = build_alibi_tensor(
                self.num_heads, seq_len, device=device, dtype=dtype, max_slope=self.max_slope,
            )
            self._cached_seq_len = seq_len
            bias = self._cached_bias
        return bias[:, :, :seq_len, :seq_len]

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, max_slope={self.max_slope}"
