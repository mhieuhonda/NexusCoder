"""
FlashAttention-2 wrapper for Nexus Coder v0.3
=============================================
Provides a unified interface for:
  1. PyTorch native SDPA (F.scaled_dot_product_attention) — always available
  2. FlashAttention-2 (flash_attn package) — optional, faster on Ampere+

If `flash_attn` is not installed, we silently fall back to SDPA.

Attribution: FlashAttention-2 algorithm from Dao et al. (2023).
Reference implementation: https://github.com/Dao-AILab/flash-attention
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Optional dependency — installed via: pip install flash-attn --no-build-isolation
    from flash_attn import flash_attn_func  # type: ignore
    _HAS_FLASH_ATTN_2 = True
except Exception:
    _HAS_FLASH_ATTN_2 = False


def has_flash_attention_2() -> bool:
    """Check whether the FlashAttention-2 package is available at runtime."""
    return _HAS_FLASH_ATTN_2


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    is_causal: bool = True,
    use_flash_attn_2: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Unified entry point for attention computation.

    Args:
        query_states:  [B, num_heads, T, head_dim]  (SDPA layout)
                       or [B, T, num_heads, head_dim] (FA2 layout, if use_flash_attn_2)
        key_states:    same layout as query_states
        value_states:  same layout as query_states
        attention_mask: optional additive mask (SDPA only). Ignored for FA2.
        dropout: attention dropout probability
        is_causal: whether to apply causal mask
        use_flash_attn_2: try to use FlashAttention-2 (falls back to SDPA if unavailable)
        softmax_scale: custom scale; default = head_dim ** -0.5

    Returns:
        attn_output: same layout as input
    """
    head_dim = query_states.shape[-1]
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    # === FlashAttention-2 path ===
    if use_flash_attn_2 and _HAS_FLASH_ATTN_2 and not attention_mask is not None:
        # FA2 expects [B, T, num_heads, head_dim]
        if query_states.dim() == 4 and query_states.shape[1] != query_states.shape[2]:
            # Likely [B, num_heads, T, head_dim] — transpose
            q = query_states.transpose(1, 2)
            k = key_states.transpose(1, 2)
            v = value_states.transpose(1, 2)
        else:
            q, k, v = query_states, key_states, value_states
        out = flash_attn_func(
            q, k, v,
            dropout_p=dropout if torch.is_grad_enabled() else 0.0,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
        # Convert back to [B, num_heads, T, head_dim]
        if out.shape[1] != query_states.shape[1] if query_states.dim() == 4 else True:
            out = out.transpose(1, 2)
        return out

    # === PyTorch SDPA path (always available) ===
    # SDPA supports attn_mask as additive bias
    try:
        out = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=dropout if torch.is_grad_enabled() else 0.0,
            is_causal=is_causal and attention_mask is None,
            scale=softmax_scale,
        )
        return out
    except Exception:
        # Manual fallback (very slow, for debugging only)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * softmax_scale
        if is_causal and attention_mask is None:
            T = attn_weights.shape[-2]
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=attn_weights.device, dtype=attn_weights.dtype),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask
        elif attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if dropout > 0 and torch.is_grad_enabled():
            attn_weights = F.dropout(attn_weights, p=dropout)
        return torch.matmul(attn_weights, value_states)


class FlashAttention(nn.Module):
    """Drop-in replacement for the manual attention in `nexus/model/attention.py`.

    Automatically picks the best available backend:
      - FlashAttention-2 if `use_flash_attn_2=True` and package is installed
      - F.scaled_dot_product_attention (SDPA) otherwise
      - Manual fallback as last resort
    """

    def __init__(
        self,
        use_flash_attn_2: bool = False,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
    ):
        super().__init__()
        self.use_flash_attn_2 = use_flash_attn_2 and _HAS_FLASH_ATTN_2
        self.dropout = dropout
        self.softmax_scale = softmax_scale

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        return flash_attention_forward(
            q, k, v,
            attention_mask=attention_mask,
            dropout=self.dropout,
            is_causal=is_causal,
            use_flash_attn_2=self.use_flash_attn_2,
            softmax_scale=self.softmax_scale,
        )

    def extra_repr(self) -> str:
        return f"flash_attn_2={self.use_flash_attn_2}, dropout={self.dropout}"
