"""
Mixture of Experts (MoE) Layer - Cốt lõi của Nexus Coder
=========================================================
24 chuyên gia (experts) tổng cộng, chỉ 3 chuyên gia được kích hoạt mỗi token.
Đạt được 10B tổng tham số với chỉ 1.5B tham số active.

Tính năng:
- Top-K routing với noise (load balancing)
- Aux loss cho load balancing giữa các expert
- Hỗ trợ SwiGLU experts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .layers import SwiGLU


class Expert(nn.Module):
    """Một chuyên gia (expert) - thực chất là một SwiGLU FFN.

    v0.3: hỗ trợ MLP-parallel (gate/up concat thành 1 matmul).
    """

    def __init__(self, hidden_size: int, intermediate_size: int, parallel: bool = True):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, intermediate_size, parallel=parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class Router(nn.Module):
    """Router/Gating network: quyết định token nào đi đến expert nào."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Tính auxiliary loss cho load balancing (Switch Transformer).

    attention_mask có thể là:
      - None:                        tất cả token đều valid
      - 2D bool [B, T]:              True = valid token
      - 2D int  [B, T]:             1 = valid, 0 = padding
      - 4D float [B, 1, 1, T]:      0 = valid, large_negative = padding
    """
    if gate_logits is None:
        # gate_logits is None → cannot compute; return 0 on proper device
        return torch.tensor(0.0)

    # Normalize attention_mask → 1D bool [N_valid]
    if attention_mask is None:
        tokens_per_expert = gate_logits.shape[0] * gate_logits.shape[1]
        # 2D shape: [B, T] already flattened by caller, so gate_logits.shape[0] is N
        if gate_logits.dim() == 2:
            tokens_per_expert = gate_logits.shape[0]
    else:
        # Convert 4D mask to 2D bool
        if attention_mask.dim() == 4:
            # [B, 1, 1, T] with 0 / -inf values
            mask_2d = attention_mask.squeeze(1).squeeze(1)  # [B, T]
            mask_bool = mask_2d > -1e9
        elif attention_mask.dim() == 3:
            mask_bool = attention_mask.squeeze(1) > 0
        elif attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                mask_bool = attention_mask
            else:
                # 0/1 or 0/-inf
                if attention_mask.dtype.is_floating_point:
                    mask_bool = attention_mask > -1e9
                else:
                    mask_bool = attention_mask > 0
        else:
            mask_bool = None

        if mask_bool is None:
            tokens_per_expert = gate_logits.shape[0]
        else:
            tokens_per_expert = mask_bool.sum().item()
            if tokens_per_expert < 1:
                tokens_per_expert = gate_logits.shape[0]

    routing_weights = F.softmax(gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    expert_mask = expert_mask.sum(dim=-2).float()

    tokens_per_expert_normalized = expert_mask.mean(dim=-2)
    router_prob_per_expert = routing_weights.mean(dim=-2)

    aux_loss = (
        num_experts * (tokens_per_expert_normalized * router_prob_per_expert).sum()
    ) / max(tokens_per_expert, 1)

    return aux_loss


class MixtureOfExperts(nn.Module):
    """MoE Layer với Top-K routing và load balancing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_active_experts
        self.router_jitter_noise = config.router_jitter_noise
        self.aux_loss_coef = config.router_aux_loss_coef

        # Router
        self.router = Router(config.hidden_size, self.num_experts)

        # Experts (v0.3: MLP-parallel by default)
        mlp_parallel = getattr(config, "mlp_parallel", True)
        self.experts = nn.ModuleList([
            Expert(config.hidden_size, config.intermediate_size, parallel=mlp_parallel)
            for _ in range(self.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden)  # [N, H]

        # Router logits
        router_logits = self.router(flat_hidden)  # [N, E]

        # Thêm noise trong training để encourage exploration
        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.router_jitter_noise

        # Top-K routing
        routing_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.num_active_experts, dim=-1
        )
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Dispatch tokens to experts
        final_hidden = torch.zeros_like(flat_hidden)

        # Vectorized: iterate through experts
        for expert_idx in range(self.num_experts):
            # Find tokens that go to this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [N]
            if not expert_mask.any():
                continue

            # Get token indices
            token_indices = expert_mask.nonzero(as_tuple=True)[0]

            # Get the corresponding weights
            expert_weights = top_k_weights[token_indices]  # [num_tokens, top_k]
            expert_weight_for_this = (top_k_indices[token_indices] == expert_idx).float() * expert_weights
            expert_weight_for_this = expert_weight_for_this.sum(dim=-1)  # [num_tokens]

            # Run expert
            expert_input = flat_hidden[token_indices]
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output * expert_weight_for_this.unsqueeze(-1)

            final_hidden[token_indices] += expert_output

        # Load balancing loss
        aux_loss = load_balancing_loss_func(
            router_logits,
            self.num_experts,
            self.num_active_experts,
            attention_mask,
        )

        final_hidden = final_hidden.view(bsz, seq_len, hidden)
        return final_hidden, aux_loss
