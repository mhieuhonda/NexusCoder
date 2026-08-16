"""
Multi-Head Attention với Grouped Query Attention (GQA) và RoPE.
Hỗ trợ KV cache để inference nhanh.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class Attention(nn.Module):
    """Multi-Head Attention với GQA + RoPE."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rotary_emb_base,
        )

        self.attn_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # RoPE
        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat K, V cho GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        # Causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        else:
            # Tạo causal mask mặc định
            causal_mask = torch.triu(
                torch.full((q_len, q_len), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.attn_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value
