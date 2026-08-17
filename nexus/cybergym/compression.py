"""
Recursive Self-Compression (RSC)
================================
Kỹ thuật self-distillation độc đáo của CyberGym — model tự distill
periodically để tìm biểu diễn effient hơn.

Ý tưởng:
  - Cứ mỗi N step, model ghi log output của chính nó trên subset data
  - So sánh output của step hiện tại vs. logged output (mô hình "teacher")
  - Tiny KL divergence loss → encourage student (current model) match teacher
  - Nhưng teacher = self at earlier step → student phải "compress" knowledge
  - Kết quả: weight pruning-friendly, structure co-adaptation tốt hơn

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSCConfig:
    """Cấu hình Recursive Self-Compression."""
    compress_period: int = 2000      # mỗi 2000 step, snapshot teacher
    kl_temperature: float = 2.0      # KL temp
    kl_weight: float = 0.1            # weight của KL loss trong total loss
    teacher_decay: float = 0.99       # EMA decay cho teacher weights
    max_teacher_snapshots: int = 3   # giữ 3 snapshot gần nhất


class RecursiveSelfCompression:
    """Hook áp dụng recursive self-compression trong training.

    Usage:
        rsc = RecursiveSelfCompression(model, config=RSCConfig())
        for step, batch in enumerate(loader):
            student_logits = model(batch.input_ids)
            ce_loss = F.cross_entropy(student_logits, batch.labels)

            if rsc.has_teacher():
                teacher_logits = rsc.get_teacher_logits(batch.input_ids)
                kl_loss = rsc.compute_kl_loss(student_logits, teacher_logits)
                total_loss = ce_loss + rsc.config.kl_weight * kl_loss
            else:
                total_loss = ce_loss

            total_loss.backward()
            optimizer.step()
            rsc.maybe_snapshot(step)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[RSCConfig] = None,
    ):
        self.model = model
        self.config = config or RSCConfig()
        self._teacher: Optional[nn.Module] = None
        self._step_count = 0
        self._stats = {
            "snapshots_taken": 0,
            "kl_loss_total": 0.0,
            "kl_loss_calls": 0,
        }

    def maybe_snapshot(self, step: int) -> bool:
        """Snapshot model làm teacher nếu đến period."""
        self._step_count = step
        if step % self.config.compress_period != 0:
            return False
        self._take_snapshot()
        return True

    def has_teacher(self) -> bool:
        return self._teacher is not None

    def get_teacher_logits(self, *args, **kwargs) -> Optional[torch.Tensor]:
        """Forward pass qua teacher (no_grad)."""
        if self._teacher is None:
            return None
        self._teacher.eval()
        with torch.no_grad():
            out = self._teacher(*args, **kwargs)
            if isinstance(out, dict):
                return out.get("logits")
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL(student || teacher) — encourage student match teacher's compression."""
        # Align shapes if needed
        if student_logits.shape != teacher_logits.shape:
            min_len = min(student_logits.shape[-2], teacher_logits.shape[-2])
            student_logits = student_logits[..., :min_len, :]
            teacher_logits = teacher_logits[..., :min_len, :]

        T = self.config.kl_temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        # Scale by T² (standard distillation trick)
        kl_scaled = kl * (T * T)

        self._stats["kl_loss_total"] += float(kl_scaled)
        self._stats["kl_loss_calls"] += 1
        return kl_scaled

    def stats(self) -> Dict[str, Any]:
        s = dict(self._stats)
        s["mean_kl_loss"] = (
            s["kl_loss_total"] / max(s["kl_loss_calls"], 1)
        )
        return s

    def _take_snapshot(self) -> None:
        """Take EMA snapshot của model làm teacher."""
        if self._teacher is None:
            try:
                self._teacher = copy.deepcopy(self.model)
            except Exception:
                self._teacher = None
                return
            for p in self._teacher.parameters():
                p.requires_grad = False
        else:
            # EMA update
            with torch.no_grad():
                teacher_params = dict(self._teacher.named_parameters())
                model_params = dict(self.model.named_parameters())
                decay = self.config.teacher_decay
                for name, p_model in model_params.items():
                    if name in teacher_params:
                        p_teacher = teacher_params[name]
                        p_teacher.data.mul_(decay).add_(
                            p_model.data, alpha=(1.0 - decay)
                        )
        self._stats["snapshots_taken"] += 1
