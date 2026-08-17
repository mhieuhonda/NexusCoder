"""LoRA - Low-Rank Adaptation cho efficient fine-tuning."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    """Config cho LoRA."""
    rank: int = 8              # LoRA rank (r)
    alpha: int = 16            # LoRA scaling factor (α)
    dropout: float = 0.0       # LoRA dropout
    # v0.4 fix: thay "gate_proj"+"up_proj" → "gate_up_proj" vì v0.3 SwiGLU(parallel=True)
    # fuses gate+up thành 1 matmul. Nếu không có gate_up_proj, có thể truyền cả 3.
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_up_proj", "down_proj",                  # FFN (MLP-parallel)
    ])
    bias: str = "none"         # "none", "all", "lora_only"
    modules_to_save: List[str] = field(default_factory=list)  # Full-finetune these
    fan_in_fan_out: bool = False

    @property
    def scaling(self) -> float:
        if self.rank <= 0:
            return 0.0
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Linear layer với LoRA adaptation.
    
    Adds low-rank matrices A and B such that:
        output = original(x) + scaling * B(A(x))
    
    Only A and B are trainable; original weights are frozen.
    """
    
    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original
        for param in self.original.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        in_features = original.in_features
        out_features = original.out_features
        
        # A: in_features × rank (init with kaiming)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # B: rank × out_features (init with zeros)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_out = self.original(x)
        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_out + lora_out
    
    def merge(self) -> nn.Linear:
        """Merge LoRA weights into original (for inference)."""
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original.weight.data += delta
        return self.original
    
    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
) -> nn.Module:
    """Apply LoRA to a model.
    
    Replaces target Linear modules with LoRALinear.
    Returns the modified model.
    
    Usage:
        config = LoRAConfig(rank=8, target_modules=["q_proj", "v_proj"])
        model = apply_lora(model, config)
        # Now only LoRA params are trainable
    """
    target_modules = set(config.target_modules)
    
    def _replace_recursive(module: nn.Module, prefix: str = ""):
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            # Check if this module should be LoRA-adapted
            short_name = name
            if short_name in target_modules and isinstance(child, nn.Linear):
                lora_layer = LoRALinear(
                    original=child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(module, name, lora_layer)
            else:
                _replace_recursive(child, full_name)
    
    _replace_recursive(model)
    
    # Make sure non-LoRA params are frozen
    for name, param in model.named_parameters():
        if "lora_" not in name and name not in config.modules_to_save:
            param.requires_grad = False
    
    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Get only LoRA params (for saving)."""
    return {
        name: param
        for name, param in model.named_parameters()
        if "lora_" in name and param.requires_grad
    }


def count_lora_params(model: nn.Module) -> Dict[str, int]:
    """Count trainable vs total params."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": trainable / total * 100 if total > 0 else 0,
    }
