"""ML Training Skill - Sinh training loop PyTorch hoàn chỉnh.

Tạo template training loop production-ready với optimizer, scheduler,
gradient accumulation, mixed precision (AMP), và checkpointing.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLTrainingSkill(Skill):
    """Sinh PyTorch training loop với AMP, grad accumulation, checkpointing."""

    category = SkillCategory.ML
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "train", "training", "pytorch", "tensorflow", "jax",
        "epoch", "loss", "optimizer", "scheduler", "backprop",
        "gradient accumulation", "mixed precision", "amp",
        "checkpoint", "finetune", "fine-tune", "distributed",
    ]
    examples = [
        "Viết training loop PyTorch với mixed precision",
        "Train a transformer model with gradient accumulation",
        "Setup DDP training across 4 GPUs",
    ]

    @property
    def name(self) -> str:
        return "ml_training"

    @property
    def description(self) -> str:
        return (
            "Sinh PyTorch training loop production-ready: optimizer (AdamW), "
            "LR scheduler (cosine/linear warmup), mixed precision (AMP), "
            "gradient accumulation, gradient clipping, periodic checkpointing, "
            "và tùy chọn DistributedDataParallel (DDP)."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        if context and context.metadata.get("framework") == "pytorch":
            score += 0.3
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        epochs = int(context.metadata.get("epochs", 3))
        grad_accum = int(context.metadata.get("grad_accum", 1))
        use_amp = bool(context.metadata.get("amp", True))
        distributed = bool(context.metadata.get("distributed", False))

        return SkillResult(
            success=True,
            output=(
                f"[MLTraining] Ready to train: epochs={epochs}, "
                f"grad_accum={grad_accum}, amp={use_amp}, ddp={distributed}"
            ),
            artifacts=[{"path": "train.py", "content": _TRAINING_LOOP}],
            metadata={
                "skill": self.name,
                "framework": "pytorch",
                "epochs": epochs,
                "grad_accum": grad_accum,
                "use_amp": use_amp,
                "distributed": distributed,
                "optimizer": "AdamW",
                "scheduler": "cosine_with_warmup",
                "defaults": {
                    "lr": 2e-5,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.03,
                    "max_grad_norm": 1.0,
                },
                "best_practices": [
                    "Use torch.amp for mixed precision (bf16 on Ampere+)",
                    "Checkpoint every N steps to S3/GCS, never only local",
                    "Log metrics to W&B/MLflow for reproducibility",
                    "Set seeds for torch, numpy, random for determinism",
                ],
            },
            suggestions=[
                "Pin CUDA seed: torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)",
                "Use torch.compile() for graph optimization on PyTorch 2.x",
                "Enable activation checkpointing to trade speed for memory",
                "Profile with torch.profiler to find bottlenecks",
            ],
        )


_TRAINING_LOOP = '''"""Production PyTorch training loop with AMP + grad accumulation + checkpointing."""
import math, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    DDP_AVAILABLE = True
except ImportError:
    DDP_AVAILABLE = False

SEED = 42
CKPT_DIR = "checkpoints"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, num_warmup, num_training):
    def lr_lambda(current):
        if current < num_warmup:
            return float(current) / float(max(1, num_warmup))
        progress = float(current - num_warmup) / float(max(1, num_training - num_warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    ckpt_every: int = 500,
    device: str = "cuda",
) -> None:
    set_seed(SEED)
    os.makedirs(CKPT_DIR, exist_ok=True)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) * epochs) // grad_accum_steps
    warmup = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = GradScaler(enabled=use_amp)
    model.train()
    global_step = 0

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(dtype=torch.bfloat16, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % 50 == 0:
                    print(f"epoch={epoch} step={global_step} "
                          f"loss={loss.item()*grad_accum_steps:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

                if global_step % ckpt_every == 0:
                    ckpt_path = os.path.join(CKPT_DIR, f"step-{global_step}.pt")
                    torch.save({"model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "step": global_step}, ckpt_path)
                    print(f"saved checkpoint -> {ckpt_path}")
'''
