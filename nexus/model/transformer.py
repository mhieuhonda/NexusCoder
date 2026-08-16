"""
Transformer Block cho Nexus Coder.
Kết hợp attention, MoE và RMSNorm với pre-norm structure.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .layers import RMSNorm
from .attention import Attention
from .moe import MixtureOfExperts


class NexusDecoderLayer(nn.Module):
    """Một decoder layer với: Attention → MoE, cả hai có residual + pre-norm."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Pre-norm
        self.input_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Attention
        self.self_attn = Attention(config)

        # MoE FFN
        self.moe = MixtureOfExperts(config)

        # Optional: dense FFN fallback (cho testing)
        self.use_moe = True

    def forward(
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

        # Pre-norm + MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_norm(hidden_states)
        moe_output, aux_loss = self.moe(hidden_states)
        hidden_states = residual + moe_output

        return hidden_states, new_kv, aux_loss
