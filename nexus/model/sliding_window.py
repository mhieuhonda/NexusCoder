"""
Sliding Window Attention for Nexus Coder v0.3
=============================================
Local attention within a window of `sliding_window_size` tokens.
Combined with global attention layers, this enables efficient long-context
training (e.g. 64k+ sequences) at a fraction of the compute cost.

Reference: Beltagy et al., "Longformer: The Long-Document Transformer" (2020).
Attribution: Concept from Longformer / Mistral-7B / Gemma.

This module exports a helper that builds the appropriate attention mask:
  - For SWA layers: causal + windowed (tokens outside the window are masked to -inf)
  - For global layers: causal only
"""
from __future__ import annotations

from typing import List, Optional

import torch


def build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    is_causal: bool = True,
) -> torch.Tensor:
    """Build a [seq_len, seq_len] additive mask for sliding-window attention.

    A token at position `i` can attend to positions `[max(0, i - window + 1), i]`
    (if causal) or `[i - window + 1, i + window - 1]` (non-causal).

    Returns:
        mask: tensor of shape [seq_len, seq_len], 0 where allowed and -inf where masked.
    """
    # Default: allow everything, then mask out
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

    if is_causal:
        # Causal: can only look at past + self
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        mask = mask + causal_mask

    # Sliding window: mask positions outside [i - window + 1, i] (causal) or
    #                                  [i - window + 1, i + window - 1] (non-causal)
    for i in range(seq_len):
        if is_causal:
            lo = max(0, i - window_size + 1)
            hi = i + 1
            # Mask everything outside [lo, hi]
            if lo > 0:
                mask[i, :lo] = float("-inf")
        else:
            lo = max(0, i - window_size + 1)
            hi = min(seq_len, i + window_size)
            if lo > 0:
                mask[i, :lo] = float("-inf")
            if hi < seq_len:
                mask[i, hi:] = float("-inf")

    return mask


def get_layer_attention_pattern(
    num_layers: int,
    use_sliding_window: bool,
    sliding_window_layers: Optional[List[int]] = None,
) -> List[str]:
    """Decide which layers use SWA vs global attention.

    Mistral-7B alternates: SWA on even layers, global on odd.
    We follow the same convention if `sliding_window_layers` is None.

    Returns:
        List of strings: "sliding_window" or "global", one per layer.
    """
    if not use_sliding_window:
        return ["global"] * num_layers
    if sliding_window_layers is not None:
        return [
            "sliding_window" if i in sliding_window_layers else "global"
            for i in range(num_layers)
        ]
    # Default: alternate SWA / global
    return [
        "sliding_window" if i % 2 == 0 else "global"
        for i in range(num_layers)
    ]


def apply_pattern_to_mask(
    seq_len: int,
    window_size: int,
    pattern: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the mask for a single layer based on its pattern."""
    if pattern == "sliding_window":
        return build_sliding_window_mask(
            seq_len=seq_len,
            window_size=window_size,
            device=device,
            dtype=dtype,
            is_causal=True,
        )
    # global: causal only
    causal = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return causal


class SlidingWindowMaskCache:
    """Caches sliding-window masks per layer pattern to avoid recompute."""

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._cache: dict[tuple[int, str, torch.device, torch.dtype], torch.Tensor] = {}

    def get(
        self,
        seq_len: int,
        pattern: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key = (seq_len, pattern, device, dtype)
        if key not in self._cache:
            self._cache[key] = apply_pattern_to_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                pattern=pattern,
                device=device,
                dtype=dtype,
            )
        return self._cache[key]
