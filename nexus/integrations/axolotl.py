"""
axolotl-inspired training config schema for Nexus Coder v0.3
============================================================
Ported & simplified from axolotl-ai-cloud/axolotl (Apache 2.0).

Axolotl uses a single YAML file to configure the entire training pipeline
(dataset, model, lora, deepspeed, distributed, etc.). We adapt this idea
into a typed dataclass that Nexus Coder's `scripts/train.py` will accept.

This is a SCHEMA/CONFIG class only — the actual training loop lives in
`nexus.training.trainer`. axolotl itself is NOT required at runtime.

Original attribution:
    Axolotl: a simple tool for fine-tuning LLMs.
    Authors: winglian + axolotl-ai-cloud contributors.
    License: Apache 2.0
    Source: https://github.com/axolotl-ai-cloud/axolotl
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


@dataclass
class AxolotlStyleConfig:
    """Axolotl-style training config adapted for Nexus Coder.

    Most fields are optional — defaults match Nexus Coder's 10B config.
    Use `AxolotlStyleConfig.from_dict(yaml_dict)` to load from a YAML file.
    """
    # === Model ===
    base_model: str = "nexus-coder-10b"          # variant name or HF repo
    base_model_config: Optional[str] = None       # path to NexusConfig YAML
    model_type: str = "moe_transformer"
    tokenizer_type: str = "bpe"

    # === Datasets ===
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {path, type, format, split, field}
    test_datasets: List[Dict[str, Any]] = field(default_factory=list)
    dataset_prepared_path: Optional[str] = None

    # === Sequence ===
    sequence_len: int = 4096
    max_samples: Optional[int] = None
    sample_packing: bool = True
    pad_to_sequence_len: bool = True

    # === LoRA / QLoRA ===
    adapter: Optional[str] = None          # None | "lora" | "qlora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_target_linear: bool = True
    peft_use_dora: bool = False

    # === Optimizer / LR ===
    optimizer: str = "adamw_torch"
    lr_scheduler: str = "cosine"          # cosine | linear | constant | warmup_stable_decay
    learning_rate: float = 5.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    warmup_ratio: Optional[float] = None
    max_steps: int = 5000
    num_epochs: int = 1
    gradient_accumulation_steps: int = 4

    # === Batch / precision ===
    micro_batch_size: int = 4
    batch_size: Optional[int] = None      # auto = micro * grad_accum
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    gradient_checkpointing: bool = False

    # === Distributed ===
    deepspeed: Optional[str] = None       # path to deepspeed config JSON
    fsdp: List[str] = field(default_factory=list)
    fsdp_config: Optional[Dict] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1

    # === Eval ===
    eval_steps: int = 500
    eval_table_size: int = 0
    save_steps: int = 500
    save_total_limit: int = 4
    early_stopping_patience: int = 0

    # === Logging ===
    logging_steps: int = 10
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None

    # === Inference (post-training) ===
    output_dir: str = "./checkpoints"
    inference: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AxolotlStyleConfig":
        """Build from a parsed YAML/JSON dict. Unknown keys are ignored."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def validate(self) -> List[str]:
        """Validate config. Returns list of error messages (empty = OK)."""
        errors = []
        if self.bf16 and self.fp16:
            errors.append("Cannot enable both bf16 and fp16")
        if self.adapter and self.adapter not in ("lora", "qlora"):
            errors.append(f"Unknown adapter: {self.adapter}")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.sequence_len < 64:
            errors.append("sequence_len must be >= 64")
        if self.batch_size and self.batch_size < self.micro_batch_size:
            errors.append("batch_size cannot be smaller than micro_batch_size")
        if self.deepspeed and self.fsdp:
            errors.append("Cannot use both deepspeed and fsdp")
        return errors

    def summary(self) -> str:
        """Human-readable one-line summary."""
        adapter_str = f" + {self.adapter.upper()}(r={self.lora_r})" if self.adapter else ""
        ds_str = " + DeepSpeed" if self.deepspeed else " + FSDP" if self.fsdp else ""
        return (
            f"{self.base_model}{adapter_str}{ds_str} | "
            f"lr={self.learning_rate:.1e} | "
            f"seq={self.sequence_len} | "
            f"bs={self.micro_batch_size}×{self.gradient_accumulation_steps} | "
            f"steps={self.max_steps}"
        )


__all__ = ["AxolotlStyleConfig"]
