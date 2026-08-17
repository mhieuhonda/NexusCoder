"""
Transformer Decoder Block v0.3
==============================
Kết hợp Attention (with SWA pattern) + MoE + RMSNorm với pre-norm structure.

v0.3 NEW:
- Per-layer attention pattern (sliding_window vs global)
- Gradient checkpointing hook (saves VRAM on long context)
- MoE layer accepts MLP-parallel experts
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .layers import RMSNorm
from .attention import Attention
from .moe import MixtureOfExperts
from .sliding_window import get_layer_attention_pattern


class NexusDecoderLayer(nn.Module):
    """Một decoder layer với: Attention → MoE, cả hai có residual + pre-norm."""

    def __init__(self, config, layer_idx: int = 0, attention_pattern: str = "global"):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.attention_pattern = attention_pattern

        # Pre-norm
        self.input_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Attention with layer pattern
        self.self_attn = Attention(
            config, layer_idx=layer_idx, attention_pattern=attention_pattern,
        )

        # MoE FFN
        self.moe = MixtureOfExperts(config)

        # Gradient checkpointing flag (set on the parent model)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        # Gradient checkpointing: recompute forward in backward pass to save VRAM
        if self.gradient_checkpointing and self.training:
            return self._forward_checkpoint(
                hidden_states, attention_mask, position_ids, past_key_value, use_cache,
            )
        return self._forward(
            hidden_states, attention_mask, position_ids, past_key_value, use_cache,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        residual = hidden_states

        # Pre-norm + Self-attention
        hidden_states = self.input_norm(hidden_states)
        attn_output, new_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # Pre-norm + MoE FFN (v0.4 fix: forward attention_mask for proper aux loss)
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        moe_output, aux_loss = self.moe(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + moe_output

        return hidden_states, new_kv, aux_loss

    def _forward_checkpoint(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Gradient checkpointing wrapper — recompute forward in backward pass."""
        def custom_forward(*inputs):
            return self._forward(*inputs)

        layers_outputs = torch.utils.checkpoint.checkpoint(
            custom_forward,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
            use_reentrant=False,
        )
        return layers_outputs
