"""
Multi-Head Attention v0.3
=========================
Features:
  - Grouped Query Attention (GQA)
  - RoPE with optional NTK/YaRN scaling (long-context extension)
  - FlashAttention-2 backend (when available, falls back to SDPA)
  - ALiBi position bias (optional alternative to RoPE)
  - Sliding window attention (alternating with global layers)
  - QK-norm (RMSNorm on query/key for training stability)
  - KV cache quantization (int8/fp8 for memory-efficient inference)

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import RotaryEmbedding, apply_rotary_pos_emb
from .flash_attention import flash_attention_forward, has_flash_attention_2
from .alibi import AlibiPositionBias
from .sliding_window import SlidingWindowMaskCache


class QKNorm(nn.Module):
    """RMSNorm applied to query and key (Llama-3 style)."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm.to(x.dtype) * self.weight)


class Attention(nn.Module):
    """Multi-Head Attention with GQA + RoPE/ALiBi + FlashAttention + sliding window + QK-norm."""

    def __init__(self, config, layer_idx: int = 0, attention_pattern: str = "global"):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_pattern = attention_pattern  # "global" | "sliding_window"
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # === RoPE or ALiBi ===
        self.use_alibi = config.use_alibi
        if not self.use_alibi:
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rotary_emb_base,
                scaling_type=config.rope_scaling_type,
                scaling_factor=config.rope_scaling_factor,
                yarn_beta_fast=getattr(config, "yarn_beta_fast", 32.0),
                yarn_beta_slow=getattr(config, "yarn_beta_slow", 1.0),
            )
        else:
            self.alibi = AlibiPositionBias(
                num_heads=self.num_heads,
                max_slope=getattr(config, "alibi_max_slope", 8.0),
            )

        # === QK-norm (Llama-3 style) ===
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = QKNorm(self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = QKNorm(self.head_dim, eps=config.qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # === FlashAttention ===
        self.use_flash_attn_2 = config.use_flash_attention_2 and has_flash_attention_2()
        self.use_sdpa = config.use_flash_attention  # PyTorch SDPA (always available)
        self.attn_dropout = config.attention_dropout

        # === Sliding window mask cache ===
        self.use_sliding_window = (
            config.use_sliding_window and attention_pattern == "sliding_window"
        )
        self.sliding_window_size = config.sliding_window_size
        if self.use_sliding_window:
            self._swa_cache = SlidingWindowMaskCache(window_size=self.sliding_window_size)
        else:
            self._swa_cache = None

        # === KV cache quantization ===
        self.kv_cache_quantization = config.kv_cache_quantization
        self.kv_cache_bits = config.kv_cache_bits

    def _quantize_kv_cache(self, x: torch.Tensor):
        """Quantize KV cache tensor to int8/fp8 to save memory (only at inference).

        Returns:
            - For int8: (quantized_tensor_int8, scale_tensor)
            - For fp8:  (tensor_fp8, None)
            - None / float input: (x, None)
        """
        if self.kv_cache_quantization is None or not torch.is_floating_point(x):
            return x, None
        if self.kv_cache_quantization == "int8":
            # Symmetric int8 quantization, scale stored alongside (per-row)
            abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = abs_max / 127.0
            q = (x / scale).round().clamp(-128, 127).to(torch.int8)
            return q, scale
        elif self.kv_cache_quantization == "fp8":
            return x.to(torch.float8_e4m3fn), None
        return x, None

    def _dequantize_kv_cache(self, x, scale=None) -> torch.Tensor:
        """Dequantize KV cache back to float (no-op if already float)."""
        if self.kv_cache_quantization is None or torch.is_floating_point(x):
            return x
        if self.kv_cache_quantization == "int8":
            if scale is None:
                # Cannot recover without scale → return zeros (graceful degradation)
                return torch.zeros_like(x, dtype=torch.float32)
            return x.to(torch.float32) * scale
        elif self.kv_cache_quantization == "fp8":
            return x.to(torch.float32)
        return x

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
            bsz, q_len, self.num_heads, self.head_dim,
        ).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim,
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim,
        ).transpose(1, 2)

        # QK-norm
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Apply RoPE
        if not self.use_alibi:
            cos, sin = self.rotary_emb(value_states, seq_len=q_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids,
            )

        # KV cache
        if past_key_value is not None:
            # Unpack: past_key_value is (cached_k, cached_v, k_scale, v_scale) for int8
            if isinstance(past_key_value, tuple) and len(past_key_value) == 4:
                cached_k, cached_v, k_scale, v_scale = past_key_value
            else:
                cached_k, cached_v = past_key_value
                k_scale, v_scale = None, None
            # dequantize if needed
            cached_k = self._dequantize_kv_cache(cached_k, k_scale)
            cached_v = self._dequantize_kv_cache(cached_v, v_scale)
            key_states = torch.cat([cached_k, key_states], dim=2)
            value_states = torch.cat([cached_v, value_states], dim=2)
        past_key_value = None
        if use_cache:
            # Quantize for storage (scales preserved)
            k_cached, k_scale = self._quantize_kv_cache(key_states)
            v_cached, v_scale = self._quantize_kv_cache(value_states)
            # Always return 4-tuple so downstream code knows the layout
            past_key_value = (k_cached, v_cached, k_scale, v_scale)

        # Repeat K, V cho GQA
        if self.num_kv_groups > 1:
            key_states = key_states.repeat_interleave(self.num_kv_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_kv_groups, dim=1)

        # Build attention mask
        full_mask = None
        if self.use_sliding_window and self._swa_cache is not None:
            full_seq_len = key_states.shape[2]
            full_mask = self._swa_cache.get(
                seq_len=full_seq_len,
                pattern="sliding_window",
                device=hidden_states.device,
                dtype=query_states.dtype,
            )
            if attention_mask is not None:
                # attention_mask: [B, 1, 1, T] (0 = keep, -inf = mask)
                full_mask = full_mask + attention_mask
        elif attention_mask is not None:
            full_mask = attention_mask

        # ALiBi additive bias
        if self.use_alibi:
            full_seq_len = key_states.shape[2]
            alibi_bias = self.alibi(
                seq_len=full_seq_len,
                device=hidden_states.device,
                dtype=query_states.dtype,
            )
            # ALiBi is [1, num_heads, T, T]; broadcast
            if full_mask is None:
                full_mask = alibi_bias
            else:
                full_mask = full_mask + alibi_bias

        # YaRN temperature correction
        softmax_scale = None
        if not self.use_alibi and self.config.rope_scaling_type == "yarn":
            temperature = self.rotary_emb.get_attention_temperature()
            softmax_scale = (self.head_dim ** -0.5) / temperature

        # Compute attention
        if self.use_flash_attn_2:
            attn_output = flash_attention_forward(
                query_states, key_states, value_states,
                attention_mask=full_mask,
                dropout=self.attn_dropout,
                is_causal=True,
                use_flash_attn_2=True,
                softmax_scale=softmax_scale,
            )
        elif self.use_sdpa:
            try:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=full_mask,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                    is_causal=(full_mask is None),
                    scale=softmax_scale,
                )
            except Exception:
                # Manual fallback
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
                scale = softmax_scale or (self.head_dim ** -0.5)
                attn_weights = attn_weights * scale
                if full_mask is not None:
                    attn_weights = attn_weights + full_mask
                else:
                    causal_mask = torch.triu(
                        torch.full((q_len, q_len), float("-inf"),
                                   device=hidden_states.device, dtype=query_states.dtype),
                        diagonal=1,
                    )
                    attn_weights = attn_weights + causal_mask
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                if self.attn_dropout > 0 and self.training:
                    attn_weights = F.dropout(attn_weights, p=self.attn_dropout)
                attn_output = torch.matmul(attn_weights, value_states)
        else:
            # Manual attention (slow)
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            scale = softmax_scale or (self.head_dim ** -0.5)
            attn_weights = attn_weights * scale
            if full_mask is not None:
                attn_weights = attn_weights + full_mask
            else:
                causal_mask = torch.triu(
                    torch.full((q_len, q_len), float("-inf"),
                               device=hidden_states.device, dtype=query_states.dtype),
                    diagonal=1,
                )
                attn_weights = attn_weights + causal_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output, past_key_value

    def extra_repr(self) -> str:
        s = f"heads={self.num_heads} (kv={self.num_kv_heads}), head_dim={self.head_dim}"
        if self.use_alibi:
            s += ", alibi=ON"
        else:
            s += f", rope_scaling={self.config.rope_scaling_type or 'none'}"
        if self.use_qk_norm:
            s += ", qk_norm=ON"
        if self.use_flash_attn_2:
            s += ", fa2=ON"
        elif self.use_sdpa:
            s += ", sdpa=ON"
        if self.use_sliding_window:
            s += f", swa(window={self.sliding_window_size})"
        if self.kv_cache_quantization:
            s += f", kv_quant={self.kv_cache_quantization}"
        return s
