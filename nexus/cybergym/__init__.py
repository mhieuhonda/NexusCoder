"""
Nexus Coder CyberGym Module - v0.4 NEW
=======================================
CyberForge training methodology: kỹ thuật train độc đáo khiến 423B params
strong hơn 1000B+ models trained conventionally.

Components:
  1. Mutation Pressure Training (MPT) — mutation.py
     Periodic random perturbation + selection pressure → escape local optima.

  2. Code Genome Initialization (CGI) — genome.py
     Khởi tạo weight theo code motifs → prior knowledge.

  3. Adaptive Density Routing (ADR) — adaptive_routing.py
     Top-k active experts thay đổi theo input complexity.

  4. Expert Speciation Curriculum (ESC) — speciation.py
     48 experts → 48 "species" (Python/JS/Rust/...).

  5. Recursive Self-Compression (RSC) — compression.py
     Self-distillation để encourage efficient representations.

  6. Context Expansion Protocol (CEP) — context_expansion.py
     Progressive context extension 32k → 3M.

  7. CyberForgeTrainer — trainer.py
     Orchestrator cho toàn bộ pipeline.

Tác giả: Hieu Louis (2026)
"""
from .mutation import (
    MutationPressureTraining,
    MPTConfig,
    MutationState,
    apply_mpt_to_model,
)
from .genome import (
    CodeGenomeInitializer,
    GenomeConfig,
    apply_genome_init,
    DEFAULT_CODE_MOTIFS,
)
from .adaptive_routing import (
    AdaptiveRouter,
    ADRConfig,
    adaptive_top_k,
    compute_router_entropy,
)
from .speciation import (
    SpeciationCurriculum,
    SpeciationConfig,
    CurriculumPhase,
    DEFAULT_EXPERT_DOMAIN_MAP,
)
from .compression import (
    RecursiveSelfCompression,
    RSCConfig,
)
from .context_expansion import (
    ContextExpansionProtocol,
    CEPConfig,
    chunked_attention_mask,
)
from .trainer import (
    CyberForgeTrainer,
    CyberForgeConfig,
)

__all__ = [
    # Mutation Pressure Training
    "MutationPressureTraining",
    "MPTConfig",
    "MutationState",
    "apply_mpt_to_model",
    # Code Genome Init
    "CodeGenomeInitializer",
    "GenomeConfig",
    "apply_genome_init",
    "DEFAULT_CODE_MOTIFS",
    # Adaptive Density Routing
    "AdaptiveRouter",
    "ADRConfig",
    "adaptive_top_k",
    "compute_router_entropy",
    # Expert Speciation Curriculum
    "SpeciationCurriculum",
    "SpeciationConfig",
    "CurriculumPhase",
    "DEFAULT_EXPERT_DOMAIN_MAP",
    # Recursive Self-Compression
    "RecursiveSelfCompression",
    "RSCConfig",
    # Context Expansion Protocol
    "ContextExpansionProtocol",
    "CEPConfig",
    "chunked_attention_mask",
    # Orchestrator
    "CyberForgeTrainer",
    "CyberForgeConfig",
]
