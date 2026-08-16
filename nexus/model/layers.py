"""
RMSNorm + SwiGLU layers v0.3
============================
- RMSNorm (Zhang & Sennrich, 2019) — unchanged
- SwiGLU — adds MLP-parallel variant (compute gate/up in parallel)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019).
    Hiệu quả hơn LayerNorm truyền thống, không có bias và không trừ mean.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation: SiLU(gate(x)) * up(x).

    v0.3: adds MLP-parallel variant — gate_proj and up_proj are computed
    as a single concatenated matmul (faster on modern GPUs).
    """

    def __init__(self, hidden_size: int, intermediate_size: int, parallel: bool = True):
        super().__init__()
        self.parallel = parallel
        if parallel:
            # Concatenated gate + up projection (mathematically identical, faster)
            self.gate_up_proj = nn.Linear(
                hidden_size, 2 * intermediate_size, bias=False,
            )
            self.gate_proj = None
            self.up_proj = None
        else:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.gate_up_proj = None
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.intermediate_size = intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.parallel:
            gate_up = self.gate_up_proj(x)
            gate, up = gate_up[..., : self.intermediate_size], gate_up[..., self.intermediate_size :]
            gate = F.silu(gate)
        else:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
        return self.down_proj(gate * up)


def _expand_token_ids_to_mask(token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Helper: chuyển token ids thành attention mask."""
    mask = torch.zeros(token_ids.shape[0], seq_len, device=token_ids.device)
    for i, ids in enumerate(token_ids):
        mask[i, : len(ids)] = 1
    return mask
