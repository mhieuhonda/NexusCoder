"""
Nexus Trainer - Training loop cho Nexus Coder
==============================================
Hỗ trợ:
- Training với kiến trúc MoE
- Auxiliary loss (load balancing)
- Checkpointing
- Logging
- Mixed precision (fp16/bf16)
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Callable
from tqdm.auto import tqdm

from ..model.nexus_coder import NexusCoderForCausalLM
from ..config import NexusConfig
from .dataset import NexusDataset, AUTHOR_TRAINING_DATA


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Cosine LR schedule với warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * num_cycles * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class NexusTrainer:
    """Trainer cho Nexus Coder."""

    def __init__(
        self,
        model: NexusCoderForCausalLM,
        config: NexusConfig,
        train_dataset: NexusDataset,
        output_dir: str = "./checkpoints",
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 5000,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 10,
        save_steps: int = 500,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # DataLoader
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=per_device_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Scheduler
        total_steps = max_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if use_amp and torch.cuda.is_available() else None

        # Logging
        os.makedirs(output_dir, exist_ok=True)
        self.log_history = []

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict:
        """Bắt đầu training."""
        print("=" * 60)
        print(f"  Nexus Coder Training")
        print(f"  Tác giả: {self.config.author}")
        print(f"  Device:  {self.device}")
        print(f"  Steps:   {self.max_steps}")
        print("=" * 60)

        global_step = 0
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            global_step = self._load_checkpoint(resume_from_checkpoint)

        self.model.train()
        start_time = time.time()

        # Training loop
        dataloader_iter = iter(self.dataloader)
        accumulated_loss = 0.0

        progress_bar = tqdm(range(global_step, self.max_steps), desc="Training")
        for step in progress_bar:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                batch = next(dataloader_iter)

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward
            if self.use_amp and torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"] / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()

            accumulated_loss += loss.item()

            # Optimizer step
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp and torch.cuda.is_available():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                global_step += 1

                # Logging
                if global_step % self.logging_steps == 0:
                    avg_loss = accumulated_loss / self.logging_steps
                    elapsed = time.time() - start_time
                    lr = self.scheduler.get_last_lr()[0]
                    log_entry = {
                        "step": global_step,
                        "loss": avg_loss,
                        "learning_rate": lr,
                        "elapsed_seconds": elapsed,
                    }
                    self.log_history.append(log_entry)
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                    })
                    accumulated_loss = 0.0

                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self._save_checkpoint(global_step)

            if global_step >= self.max_steps:
                break

        # Final save
        self._save_checkpoint(global_step, final=True)

        # Save log
        self._save_logs()

        elapsed = time.time() - start_time
        print(f"\n✅ Training hoàn thành trong {elapsed:.1f}s")
        return {"global_step": global_step, "elapsed": elapsed}

    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """Lưu checkpoint (v0.4: include AMP scaler state for safe resume)."""
        suffix = "final" if final else f"step-{step}"
        path = os.path.join(self.output_dir, f"nexus_coder-{suffix}.pt")
        # v0.4 fix: persist GradScaler state so AMP can resume safely without
        # scale-factor NaNs on first few steps.
        scaler_state = None
        scaler = getattr(self, "scaler", None)
        if scaler is not None and hasattr(scaler, "state_dict"):
            try:
                scaler_state = scaler.state_dict()
            except Exception:
                scaler_state = None
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": scaler_state,
            "step": step,
            "config": self.config.__dict__,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str) -> int:
        """Load checkpoint (v0.4: also restore AMP scaler if present)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # v0.4 fix: restore scaler state if present
        scaler_state = checkpoint.get("scaler_state_dict")
        scaler = getattr(self, "scaler", None)
        if scaler_state is not None and scaler is not None and hasattr(scaler, "load_state_dict"):
            try:
                scaler.load_state_dict(scaler_state)
            except Exception:
                pass
        step = checkpoint.get("step", 0)
        print(f"  Resumed from checkpoint at step {step}")
        return step

    def _save_logs(self) -> None:
        """Lưu training logs."""
        log_path = os.path.join(self.output_dir, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_history, f, ensure_ascii=False, indent=2)
        print(f"  📝 Saved training log: {log_path}")
