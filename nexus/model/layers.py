"""
RMSNorm và các module cơ bản cho Nexus Coder.
"""
import torch
import torch.nn as nn


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
    Dùng trong FFN của các model hiện đại như LLaMA, Mixtral.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def _expand_token_ids_to_mask(token_ids: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Helper: chuyển token ids thành attention mask."""
    mask = torch.zeros(token_ids.shape[0], seq_len, device=token_ids.device)
    for i, ids in enumerate(token_ids):
        mask[i, : len(ids)] = 1
    return mask
