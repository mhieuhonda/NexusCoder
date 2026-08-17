"""
Adaptive Density Routing (ADR)
==============================
Kỹ thuật routing độc đáo của CyberGym — top-k active experts thay đổi
theo input complexity, thay vì cố định như MoE truyền thống.

Ý tưởng:
  - Input đơn giản (1+1=2) → chỉ cần top-2 experts (nhanh, ít VRAM)
  - Input phức tạp (debug distributed race condition) → top-8 experts
  - Đánh giá complexity qua entropy của router logits:
      H = -Σ p_i log p_i  (entropy cao = uncertain = phức tạp)
  - Threshold H → map sang [min_active, max_active]

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ADRConfig:
    """Cấu hình Adaptive Density Routing."""
    min_active_experts: int = 2
    max_active_experts: int = 8
    # Entropy threshold: below → simple, above → complex
    entropy_low_threshold: float = 0.5     # ≈ log(2)/2 — rất confident
    entropy_high_threshold: float = 2.5     # ≈ log(12) — rất uncertain
    # Smooth interpolation between min/max
    smooth: bool = True


def compute_router_entropy(router_logits: torch.Tensor) -> torch.Tensor:
    """Tính entropy của router logits per token.

    Args:
        router_logits: [N, E] (N tokens, E experts)
    Returns:
        entropy: [N] — entropy per token
    """
    probs = F.softmax(router_logits, dim=-1)
    log_probs = F.log_softmax(router_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [N]
    return entropy


def adaptive_top_k(
    router_logits: torch.Tensor,
    config: ADRConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute adaptive top-k cho mỗi token.

    Args:
        router_logits: [N, E]
        config: ADRConfig
    Returns:
        top_k_weights: [N, max_k] — padded với 0 cho k < max_k
        top_k_indices: [N, max_k] — padded với -1
        per_token_k: [N] — số expert active per token
    """
    n_tokens, n_experts = router_logits.shape
    max_k = min(config.max_active_experts, n_experts)
    min_k = min(config.min_active_experts, max_k)

    # Compute entropy per token
    entropy = compute_router_entropy(router_logits)  # [N]

    # Map entropy → k
    if config.smooth:
        # Linear interpolation: low entropy → min_k, high entropy → max_k
        normalized = (
            (entropy - config.entropy_low_threshold)
            / max(
                config.entropy_high_threshold - config.entropy_low_threshold,
                1e-6,
            )
        )
        normalized = normalized.clamp(0.0, 1.0)
        per_token_k_float = min_k + normalized * (max_k - min_k)
        per_token_k = per_token_k_float.round().clamp(min_k, max_k).long()
    else:
        # Step function: 3 buckets
        per_token_k = torch.where(
            entropy < config.entropy_low_threshold,
            torch.full_like(entropy, min_k, dtype=torch.long),
            torch.where(
                entropy > config.entropy_high_threshold,
                torch.full_like(entropy, max_k, dtype=torch.long),
                torch.full_like(entropy, (min_k + max_k) // 2, dtype=torch.long),
            ),
        )

    # Top max_k cho tất cả tokens (lấy nhiều hơn rồi mask)
    routing_weights = F.softmax(router_logits, dim=-1)
    top_k_weights, top_k_indices = torch.topk(
        routing_weights, max_k, dim=-1
    )

    # Mask out weights beyond per_token_k
    # Build mask: [N, max_k] where mask[i, j] = (j < per_token_k[i])
    arange_k = torch.arange(max_k, device=router_logits.device).unsqueeze(0)  # [1, max_k]
    keep_mask = arange_k < per_token_k.unsqueeze(-1)  # [N, max_k]

    # Renormalize kept weights
    top_k_weights = top_k_weights * keep_mask.float()
    norm_sum = top_k_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    top_k_weights = top_k_weights / norm_sum

    # Indices: -1 cho các expert không active (để caller nhận biết)
    top_k_indices = torch.where(
        keep_mask, top_k_indices, torch.full_like(top_k_indices, -1)
    )

    return top_k_weights, top_k_indices, per_token_k


class AdaptiveRouter(nn.Module):
    """Router với Adaptive Density Routing.

    Drop-in replacement cho Router truyền thống trong MoE.
    """

    def __init__(self, hidden_size: int, num_experts: int, config: Optional[ADRConfig] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.config = config or ADRConfig()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            hidden_states: [N, H]
        Returns:
            top_k_weights: [N, max_k]
            top_k_indices: [N, max_k] (with -1 for inactive)
            per_token_k: [N]
        """
        logits = self.gate(hidden_states)  # [N, E]
        return adaptive_top_k(logits, self.config)
