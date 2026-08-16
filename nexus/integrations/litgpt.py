"""
litgpt-inspired utilities for Nexus Coder v0.3
==============================================
Ported & simplified from Lightning-AI/litgpt (Apache 2.0).

Adapted into Nexus Coder:
  - RoPE scaling strategies (linear / NTK-aware / YaRN) — see nexus/model/rope.py
  - FusedLinear: concatenate Q/K/V projections for one big matmul (this module)
  - `apply_rotary_pos_emb` helper signature — see nexus/model/rope.py
  - PyTorch SDPA backend selection — see nexus/model/flash_attention.py

Original attribution:
    LitGPT: Lightning AI's LLM training toolkit.
    Authors: Karpathy et al. (Lightning AI), 2023-2024.
    License: Apache 2.0
    Source:  https://github.com/Lightning-AI/litgpt
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FusedLinear(nn.Module):
    """Fused multi-linear: concatenate N separate projections into one.

    LitGPT pattern: Q/K/V projections for attention are computed as a single
    matmul of shape `[hidden, num_heads * head_dim * 3]`, then split.

    Saves one kernel launch per attention layer — meaningful at scale.

    Example:
        >>> fused = FusedLinear(2048, [2048, 512, 512, 2048])
        >>> q, k, v, o = fused(x)  # one matmul, 4 splits
    """

    def __init__(self, in_features: int, out_features_list: list[int], bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features_list = list(out_features_list)
        self.total_out = sum(self.out_features_list)
        self.weight = nn.Parameter(torch.empty(self.total_out, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.total_out))
        else:
            self.register_parameter("bias", None)
        # Init like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns tuple of tensors, one per output spec."""
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return tuple(out.split(self.out_features_list, dim=-1))

    def extra_repr(self) -> str:
        return f"in={self.in_features}, outs={self.out_features_list}, bias={self.bias is not None}"


def build_qkv_fused(hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int) -> FusedLinear:
    """Build a fused Q/K/V projection for GQA attention."""
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    return FusedLinear(hidden_size, [q_size, kv_size, kv_size], bias=False)


__all__ = ["FusedLinear", "build_qkv_fused"]
