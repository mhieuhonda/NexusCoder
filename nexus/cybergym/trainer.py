"""
CyberForge Trainer — Orchestrator
=================================
Tổng hợp toàn bộ CyberGym training methodology:
  1. Code Genome Initialization (CGI)
  2. Expert Speciation Curriculum (ESC)
  3. Mutation Pressure Training (MPT)
  4. Recursive Self-Compression (RSC)
  5. Context Expansion Protocol (CEP)
  6. Adaptive Density Routing (ADR)

Pipeline (không chạy — chỉ define):
  Stage 0: Genome Init
    - apply_genome_init(model)
  Stage 1: Speciation (30% train steps)
    - Đóng băng 90% expert routing theo domain
    - Train mỗi expert trên domain của nó
    - Context 32k (CEP stage 0)
  Stage 2: Hybridization (30% train steps)
    - Router học cách mix experts
    - Mix domain data
    - Context 131k → 524k (CEP stage 1-2)
  Stage 3: Generalization (40% train steps)
    - Mở full router + adaptive routing
    - Inject adversarial samples
    - Context 1M → 3M (CEP stage 3-5)
  Throughout:
    - MPT mỗi 500 step (mutation pressure)
    - RSC mỗi 2000 step (self-compression snapshot)
    - ADR enable từ stage 2

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .mutation import MutationPressureTraining, MPTConfig
from .genome import CodeGenomeInitializer, GenomeConfig
from .speciation import SpeciationCurriculum, SpeciationConfig, CurriculumPhase
from .compression import RecursiveSelfCompression, RSCConfig
from .context_expansion import ContextExpansionProtocol, CEPConfig
from .adaptive_routing import ADRConfig


@dataclass
class CyberForgeConfig:
    """Cấu hình tổng hợp CyberForge training."""
    # Component configs
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    speciation: SpeciationConfig = field(default_factory=SpeciationConfig)
    mpt: MPTConfig = field(default_factory=MPTConfig)
    rsc: RSCConfig = field(default_factory=RSCConfig)
    cep: CEPConfig = field(default_factory=CEPConfig)
    adr: ADRConfig = field(default_factory=ADRConfig)

    # Total schedule
    total_steps: int = 100_000
    warmup_steps: int = 1_000
    # Phase ratios (override speciation defaults nếu cần)
    speciation_ratio: float = 0.30
    hybridization_ratio: float = 0.30
    generalization_ratio: float = 0.40

    # Hardware
    use_amp: bool = True
    use_deepspeed: bool = False
    gradient_clip: float = 1.0

    # Checkpoint
    checkpoint_dir: str = "./checkpoints"
    checkpoint_period: int = 5_000
    log_period: int = 100


class CyberForgeTrainer:
    """Orchestrator cho toàn bộ CyberGym training.

    Lưu ý: Trainer này KHÔNG chạy trong môi trường sandbox.
    Nó define toàn bộ pipeline dưới dạng code, để user chạy trên cluster riêng.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[CyberForgeConfig] = None,
        train_loader: Optional[Any] = None,
        val_loader: Optional[Any] = None,
        val_loss_fn: Optional[Callable[[nn.Module], float]] = None,
    ):
        self.model = model
        self.config = config or CyberForgeConfig()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_loss_fn = val_loss_fn

        # Sub-components
        self.genome = CodeGenomeInitializer(self.config.genome)
        self.speciation = SpeciationCurriculum(
            self.config.speciation,
            total_steps=self.config.total_steps,
        )
        self.mpt = MutationPressureTraining(
            model,
            config=self.config.mpt,
            val_loss_fn=val_loss_fn,
        )
        self.rsc = RecursiveSelfCompression(model, config=self.config.rsc)
        self.cep = ContextExpansionProtocol(self.config.cep)

        # Stats
        self._step = 0
        self._stage_stats: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Stage 0: Genome Initialization
    # ------------------------------------------------------------------

    def stage_genome_init(self) -> Dict[str, int]:
        """Stage 0: Apply Code Genome Init to model weights."""
        stats = self.genome.apply_to(self.model)
        self._stage_stats.append({"stage": "genome_init", **stats})
        return stats

    # ------------------------------------------------------------------
    # CEP: Apply stage-th context expansion
    # ------------------------------------------------------------------

    def apply_cep_stage(self, stage_idx: int) -> Dict[str, Any]:
        """Apply CEP stage-th vào model config."""
        schedule = self.cep.get_schedule()
        if stage_idx < 0 or stage_idx >= len(schedule):
            return {"error": "invalid stage_idx"}
        stage = schedule[stage_idx]
        self.cep.apply_stage_to_config(self.model.config, stage_idx)
        return stage

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def train_step(self, batch: Any) -> Dict[str, Any]:
        """One training step — orchestrates all CyberGym components.

        Args:
            batch: dict with input_ids, attention_mask, labels, (optional) domain
        Returns:
            dict with loss, phase, mpt_stats, rsc_stats, cep_stage
        """
        if self.train_loader is None and batch is None:
            return {"error": "no batch"}

        # Determine current phase
        phase = self.speciation.get_phase_at_step(self._step)
        cep_stage = self._cep_stage_for_step(self._step)
        cep_info = self.cep.get_schedule()[cep_stage] if cep_stage < len(self.cep.get_schedule()) else None

        # Forward pass
        # (Actual forward/backward should be done by caller; here we just dispatch)
        self._step += 1

        # MPT
        mpt_stats = self.mpt.step()

        # RSC snapshot
        rsc_snapshot = self.rsc.maybe_snapshot(self._step)

        return {
            "step": self._step,
            "phase": phase.value,
            "cep_stage": cep_stage,
            "cep_info": cep_info,
            "mpt": mpt_stats,
            "rsc_snapshot_taken": rsc_snapshot,
        }

    def _cep_stage_for_step(self, step: int) -> int:
        """Map step → CEP stage."""
        n_stages = len(self.cep.config.stages)
        spec_end = int(self.config.total_steps * self.config.speciation_ratio)
        hyb_end = int(self.config.total_steps * (self.config.speciation_ratio + self.config.hybridization_ratio))
        if step < spec_end:
            return 0  # 32k
        if step < hyb_end:
            progress = (step - spec_end) / max(hyb_end - spec_end, 1)
            return min(n_stages - 1, 1 + int(progress * 2))  # stage 1-2
        progress = (step - hyb_end) / max(self.config.total_steps - hyb_end, 1)
        return min(n_stages - 1, 3 + int(progress * (n_stages - 3)))  # stage 3+

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "total_steps": self.config.total_steps,
            "phases": {
                "speciation_end": int(self.config.total_steps * self.config.speciation_ratio),
                "hybridization_end": int(self.config.total_steps * (self.config.speciation_ratio + self.config.hybridization_ratio)),
            },
            "genome": self.genome.get_genome_summary(),
            "speciation": self.speciation.summary(),
            "cep": self.cep.summary(),
            "mpt_stats": self.mpt.stats(),
            "rsc_stats": self.rsc.stats(),
            "adr": {
                "min_active": self.config.adr.min_active_experts,
                "max_active": self.config.adr.max_active_experts,
            },
            "stage_history": self._stage_stats,
        }

    def print_summary(self) -> None:
        """In tóm tắt pipeline."""
        s = self.summary()
        print("=" * 72)
        print("  CyberForge Training Pipeline Summary")
        print("=" * 72)
        print(f"  Total steps:          {s['total_steps']:,}")
        print(f"  Speciation phase end: {s['phases']['speciation_end']:,}")
        print(f"  Hybridization end:   {s['phases']['hybridization_end']:,}")
        print("-" * 72)
        print(f"  Genome motifs:        {s['genome']['num_motifs']}")
        print(f"  Genome inject layers: {s['genome']['injection_layers']}")
        print("-" * 72)
        print(f"  CEP stages:           {len(s['cep']['stages'])}")
        print(f"  CEP growth:           {s['cep']['total_context_growth']}")
        print(f"  CEP growth factor:    {s['cep']['growth_factor']:.0f}x")
        print("-" * 72)
        print(f"  ADR active experts:   {s['adr']['min_active']}..{s['adr']['max_active']}")
        print(f"  MPT acceptance rate:  {s['mpt_stats'].get('acceptance_rate', 0):.1%}")
        print(f"  RSC snapshots:        {s['rsc_stats'].get('snapshots_taken', 0)}")
        print("=" * 72)
