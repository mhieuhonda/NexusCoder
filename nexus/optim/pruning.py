"""Pruning - Structured/unstructured pruning."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Config cho pruning."""
    method: str = "magnitude_unstructured"  # "magnitude_unstructured", "magnitude_structured", "random"
    amount: float = 0.2  # Fraction of weights to prune (0.0-1.0)
    target_modules: List[str] = None  # Default: all Linear
    dim: int = 0  # For structured: which dim to prune
    n_prune_steps: int = 1  # Iterative pruning steps


class Pruner:
    """Prune model weights để giảm params và inference cost.
    
    Methods:
    - magnitude_unstructured: Prune smallest-magnitude weights (set to 0)
    - magnitude_structured: Remove entire neurons/channels
    - random: Random pruning (baseline)
    
    Usage:
        pruner = Pruner(config=PruningConfig(amount=0.3))
        pruned_model = pruner.prune(model)
    """
    
    def __init__(self, config: PruningConfig = None):
        self.config = config or PruningConfig()
        if self.config.target_modules is None:
            self.config.target_modules = [nn.Linear]
    
    def prune(self, model: nn.Module) -> nn.Module:
        """Prune model in-place."""
        method = self.config.method
        
        if method == "magnitude_unstructured":
            return self._prune_magnitude_unstructured(model)
        elif method == "magnitude_structured":
            return self._prune_magnitude_structured(model)
        elif method == "random":
            return self._prune_random(model)
        else:
            raise ValueError(f"Unknown pruning method: {method}")
    
    def _prune_magnitude_unstructured(self, model: nn.Module) -> nn.Module:
        """Prune smallest-magnitude weights (set to 0)."""
        try:
            from torch.nn.utils import prune
        except ImportError:
            logger.error("torch.nn.utils.prune not available")
            return model
        
        amount = self.config.amount
        
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.config.target_modules)):
                prune.l1_unstructured(module, name="weight", amount=amount)
                # Make pruning permanent
                prune.remove(module, "weight")
        
        # Count sparsity
        sparsity = self._compute_sparsity(model)
        logger.info(f"Magnitude unstructured pruning: {sparsity*100:.1f}% weights pruned")
        return model
    
    def _prune_magnitude_structured(self, model: nn.Module) -> nn.Module:
        """Remove entire neurons/channels based on L2 norm."""
        try:
            from torch.nn.utils import prune
        except ImportError:
            logger.error("torch.nn.utils.prune not available")
            return model
        
        amount = self.config.amount
        dim = self.config.dim
        
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.config.target_modules)):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
                prune.remove(module, "weight")
        
        sparsity = self._compute_sparsity(model)
        logger.info(f"Magnitude structured pruning (dim={dim}): {sparsity*100:.1f}% pruned")
        return model
    
    def _prune_random(self, model: nn.Module) -> nn.Module:
        """Random pruning (baseline)."""
        try:
            from torch.nn.utils import prune
        except ImportError:
            return model
        
        amount = self.config.amount
        
        for name, module in model.named_modules():
            if isinstance(module, tuple(self.config.target_modules)):
                prune.random_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
        
        return model
    
    def _compute_sparsity(self, model: nn.Module) -> float:
        """Compute fraction of zero weights."""
        total = 0
        zeros = 0
        for param in model.parameters():
            total += param.numel()
            zeros += (param == 0).sum().item()
        return zeros / total if total > 0 else 0
    
    def iterative_prune(
        self,
        model: nn.Module,
        train_fn=None,
        steps: int = None,
    ) -> nn.Module:
        """Iterative pruning: prune, retrain, prune, retrain, ...
        
        Args:
            model: Model to prune
            train_fn: Function(model) to retrain after each prune step
            steps: Number of prune-retrain cycles (default: config.n_prune_steps)
        """
        steps = steps or self.config.n_prune_steps
        amount_per_step = self.config.amount / steps
        
        original_config = self.config.amount
        self.config.amount = amount_per_step
        
        for step in range(steps):
            logger.info(f"Iterative pruning step {step+1}/{steps}")
            self.prune(model)
            if train_fn:
                logger.info("Retraining after pruning...")
                train_fn(model)
        
        self.config.amount = original_config
        return model
    
    def stats(self, model: nn.Module) -> Dict[str, float]:
        """Get pruning stats."""
        sparsity = self._compute_sparsity(model)
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
        return {
            "total_params": total_params,
            "nonzero_params": nonzero_params,
            "zero_params": total_params - nonzero_params,
            "sparsity": sparsity,
            "compression_ratio": 1 / (1 - sparsity) if sparsity < 1 else float("inf"),
        }
