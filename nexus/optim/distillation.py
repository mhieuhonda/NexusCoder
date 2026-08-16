"""Knowledge Distillation - Train small model từ large teacher."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Config cho knowledge distillation."""
    temperature: float = 2.0      # Softmax temperature
    alpha: float = 0.5            # Weight for distillation loss (1-alpha for hard labels)
    hard_label_loss: str = "ce"   # "ce", "focal", "label_smoothing"
    label_smoothing: float = 0.1
    teacher_temp: Optional[float] = None  # Defaults to temperature


class Distiller:
    """Knowledge distillation: train student model from teacher.
    
    Loss = α * KL(teacher_soft || student_soft) * T² 
         + (1-α) * CE(student_hard, labels)
    
    Usage:
        distiller = Distiller(config=DistillationConfig(temperature=4.0))
        for batch in dataloader:
            loss = distiller.compute_loss(
                student_logits=student(batch),
                teacher_logits=teacher(batch),  # no_grad
                labels=batch_labels,
            )
            loss.backward()
    """
    
    def __init__(self, config: DistillationConfig = None):
        self.config = config or DistillationConfig()
    
    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss.
        
        Args:
            student_logits: [B, V] logits from student model
            teacher_logits: [B, V] logits from teacher model (should be no_grad)
            labels: [B] ground truth labels (optional, for hard label loss)
        
        Returns:
            Dict with 'loss', 'distill_loss', 'hard_loss' tensors
        """
        cfg = self.config
        T = cfg.temperature
        teacher_T = cfg.teacher_temp or T
        
        # Distillation loss: KL divergence between soft predictions
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / teacher_T, dim=-1)
        
        # KL(teacher || student) = sum(teacher * log(teacher/student))
        # = sum(teacher * log(teacher)) - sum(teacher * log(student))
        # We only need the second term (first is constant w.r.t. student)
        kl_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        # Scale by T² (per Hinton et al.)
        distill_loss = kl_loss * (T ** 2)
        
        # Hard label loss
        hard_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None:
            if cfg.hard_label_loss == "ce":
                hard_loss = F.cross_entropy(student_logits, labels)
            elif cfg.hard_label_loss == "focal":
                # Focal loss
                ce = F.cross_entropy(student_logits, labels, reduction="none")
                pt = torch.exp(-ce)
                hard_loss = ((1 - pt) ** 2 * ce).mean()
            elif cfg.hard_label_loss == "label_smoothing":
                hard_loss = F.cross_entropy(
                    student_logits, labels,
                    label_smoothing=cfg.label_smoothing,
                )
        
        # Total loss
        total_loss = cfg.alpha * distill_loss + (1 - cfg.alpha) * hard_loss
        
        return {
            "loss": total_loss,
            "distill_loss": distill_loss,
            "hard_loss": hard_loss,
        }
    
    def train_step(
        self,
        student: nn.Module,
        teacher: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """One distillation training step.
        
        Args:
            student: Student model (trainable)
            teacher: Teacher model (will be set to eval, no_grad)
            batch: Dict with 'input_ids', 'attention_mask', 'labels'
            optimizer: Optimizer for student
        
        Returns:
            Dict of loss values
        """
        teacher.eval()
        
        with torch.no_grad():
            teacher_outputs = teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            teacher_logits = teacher_outputs["logits"] if isinstance(teacher_outputs, dict) else teacher_outputs
        
        student.train()
        student_outputs = student(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        student_logits = student_outputs["logits"] if isinstance(student_outputs, dict) else student_outputs
        
        losses = self.compute_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=batch.get("labels"),
        )
        
        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
