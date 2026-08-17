"""
CyberForge Mutation Pressure Training (MPT)
===========================================
Kỹ thuật train độc đáo của Nexus Coder v0.4 — lõi của CyberGym.

Ý tưởng:
  Gradient descent truyền thống hội tụ về local optima. MPT kết hợp:
    1. Gradient descent (local search, mạnh)
    2. Random mutation (global search, yếu nhưng tránh local optima)
    3. Selection pressure: chỉ giữ lại mutation có lợi (giảm val loss)

  Cứ mỗi K step:
    - Sample 1% weight ngẫu nhiên (mutation_rate)
    - Áp perturbation N(0, sigma^2) lên chúng
    - Đánh giá trên val set
    - Nếu val_loss giảm ≥ threshold: giữ lại (beneficial mutation)
    - Nếu val_loss tăng > threshold: revert + giảm sigma
    - Nếu |Δval_loss| < threshold: keep với prob = exp(-Δval_loss/T)

  Tổng quát hơn Sharpness-Aware Minimization (SAM) vì:
    - SAM chỉ minimize sharpness (1 chiều), MPT explore mọi hướng
    - MPT không cần second-order gradient (rẻ hơn)
    - MPT có "selection pressure" kiểu di truyền → tránh local optima

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MutationState:
    """Trạng thái của một lần mutation — để revert nếu cần."""
    param_name: str
    original_tensor: torch.Tensor      # snapshot trước khi mutate
    perturbation: torch.Tensor          = None                    # noise đã thêm
    applied: bool                       = False
    val_loss_before: float             = float("inf")
    val_loss_after: float              = float("inf")


@dataclass
class MPTConfig:
    """Cấu hình Mutation Pressure Training."""
    mutation_rate: float       = 0.01   # tỷ lệ weight bị mutate mỗi step
    mutation_sigma: float      = 1e-4   # độ lớn perturbation
    mutation_period: int       = 500    # K step giữa 2 lần mutate
    keep_ratio: float           = 0.7    # tỷ lệ mutation được giữ lại (selection pressure)
    sigma_adapt: float         = 1.1    # factor adapt sigma (1.1 → +10% hoặc -10%)
    sigma_min: float           = 1e-7
    sigma_max: float           = 1e-2
    acceptance_threshold: float = 0.0   # Δval_loss ≥ 0 → accept
    temperature: float          = 1.0   # softmax temp cho probabilistic acceptance
    # Layers ưu tiên mutate (thường là expert FFN — ít rủi ro, nhiều gain)
    target_substrings: List[str] = field(
        default_factory=lambda: ["moe.experts", "lm_head", "embed_tokens"]
    )
    # Layers tránh mutate (router, norm — quá nhạy cảm)
    skip_substrings: List[str] = field(
        default_factory=lambda: ["router", "norm", "layernorm", "rmsnorm"]
    )


class MutationPressureTraining:
    """CyberForge Mutation Pressure Training hook.

    Usage:
        mpt = MutationPressureTraining(model, config=MPTConfig())
        for step, batch in enumerate(loader):
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()

            if step % config.mutation_period == 0:
                mpt.maybe_mutate(val_loader, val_loss_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MPTConfig] = None,
        val_loss_fn: Optional[Callable[[nn.Module], float]] = None,
    ):
        self.model = model
        self.config = config or MPTConfig()
        self.val_loss_fn = val_loss_fn
        self._mutations: List[MutationState] = []
        self._step_count = 0
        self._stats = {
            "mutations_attempted": 0,
            "mutations_accepted": 0,
            "mutations_reverted": 0,
            "total_delta_val_loss": 0.0,
        }
        # Lưu current sigma (có thể adapt)
        self._current_sigma = self.config.mutation_sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Gọi mỗi train step. Tự động mutate khi đến period."""
        self._step_count += 1
        if self._step_count % self.config.mutation_period != 0:
            return {"mutated": False}
        return self.maybe_mutate()

    def maybe_mutate(self) -> Dict[str, Any]:
        """Thực hiện một lần mutation pressure."""
        if self.val_loss_fn is None:
            # Không có val_fn → dry-run: chỉ mutate, không decide keep/revert
            return self._dry_mutate()

        # 1. Snapshot val loss trước mutation
        val_before = float(self.val_loss_fn(self.model))

        # 2. Snapshot weight & apply perturbation
        targets = self._select_target_params()
        if not targets:
            return {"mutated": False, "reason": "no_target_params"}

        mutations: List[MutationState] = []
        for name, param in targets:
            if not param.requires_grad or not torch.is_floating_point(param.data):
                continue
            original = param.data.clone()
            noise = torch.randn_like(param.data) * self._current_sigma
            param.data.add_(noise)
            mutations.append(MutationState(
                param_name=name,
                original_tensor=original,
                perturbation=noise,
                applied=True,
                val_loss_before=val_before,
            ))

        # 3. Đánh giá val loss sau mutation
        val_after = float(self.val_loss_fn(self.model))
        delta = val_before - val_after  # >0 means improved

        # 4. Selection pressure
        kept = 0
        reverted = 0
        if delta >= self.config.acceptance_threshold:
            # Beneficial mutation → keep all
            kept = len(mutations)
            self._adapt_sigma(up=True)
        else:
            # Probabilistic acceptance (simulated annealing style)
            prob = math.exp(delta / max(self.config.temperature, 1e-8))
            if random.random() < prob and random.random() < self.config.keep_ratio:
                kept = len(mutations)
            else:
                # Revert
                for m in mutations:
                    param = self._get_param_by_name(m.param_name)
                    if param is not None:
                        param.data.copy_(m.original_tensor)
                reverted = len(mutations)
                self._adapt_sigma(up=False)

        # 5. Update stats
        self._stats["mutations_attempted"] += len(mutations)
        self._stats["mutations_accepted"] += kept
        self._stats["mutations_reverted"] += reverted
        self._stats["total_delta_val_loss"] += delta

        return {
            "mutated": True,
            "n_targets": len(mutations),
            "n_kept": kept,
            "n_reverted": reverted,
            "val_before": val_before,
            "val_after": val_after,
            "delta": delta,
            "current_sigma": self._current_sigma,
        }

    def stats(self) -> Dict[str, Any]:
        s = dict(self._stats)
        s["current_sigma"] = self._current_sigma
        s["acceptance_rate"] = (
            s["mutations_accepted"] / max(s["mutations_attempted"], 1)
        )
        s["mean_delta_val_loss"] = (
            s["total_delta_val_loss"] / max(s["mutations_attempted"], 1)
        )
        return s

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _select_target_params(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """Chọn các param để mutate theo config (target/skip substrings)."""
        targets: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not torch.is_floating_point(param.data):
                continue
            # Skip list ưu tiên
            if any(s in name.lower() for s in self.config.skip_substrings):
                continue
            # Target list (nếu rỗng → accept all non-skip)
            if self.config.target_substrings:
                if not any(s in name.lower() for s in self.config.target_substrings):
                    continue
            targets.append((name, param))

        # Sample mutation_rate fraction
        n_total = len(targets)
        n_mutate = max(1, int(n_total * self.config.mutation_rate))
        if n_mutate < n_total:
            targets = random.sample(targets, n_mutate)
        return targets

    def _get_param_by_name(self, name: str) -> Optional[torch.nn.Parameter]:
        for n, p in self.model.named_parameters():
            if n == name:
                return p
        return None

    def _adapt_sigma(self, up: bool) -> None:
        """Adaptive sigma: tăng nếu mutation có lợi, giảm nếu không."""
        if up:
            self._current_sigma = min(
                self._current_sigma * self.config.sigma_adapt,
                self.config.sigma_max,
            )
        else:
            self._current_sigma = max(
                self._current_sigma / self.config.sigma_adapt,
                self.config.sigma_min,
            )

    def _dry_mutate(self) -> Dict[str, Any]:
        """Mutation không có val_fn — chỉ perturb, không revert."""
        targets = self._select_target_params()
        for name, param in targets:
            if not torch.is_floating_point(param.data):
                continue
            noise = torch.randn_like(param.data) * self._current_sigma
            param.data.add_(noise)
        self._stats["mutations_attempted"] += len(targets)
        self._stats["mutations_accepted"] += len(targets)
        return {
            "mutated": True,
            "dry_run": True,
            "n_targets": len(targets),
            "current_sigma": self._current_sigma,
        }


def apply_mpt_to_model(
    model: nn.Module,
    config: Optional[MPTConfig] = None,
    val_loss_fn: Optional[Callable[[nn.Module], float]] = None,
) -> MutationPressureTraining:
    """Helper: khởi tạo MPT hook cho model."""
    return MutationPressureTraining(model, config=config, val_loss_fn=val_loss_fn)
