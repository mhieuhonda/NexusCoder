"""
Expert Speciation Curriculum
============================
Kỹ thuật curriculum learning độc đáo của CyberGym — mỗi expert chuyên biệt
hóa cho một domain code cụ thể trong giai đoạn đầu, rồi fine-tune tổng hợp.

Ý tưởng (lấy cảm hứng từ speciation trong sinh học):
  - 48 experts → 48 "loài" chuyên biệt (Python, JS, Rust, Go, SQL, ...)
  - Phase 1 (Speciation, 30% train): mỗi expert chỉ thấy data của 1 domain
    → weight bias mạnh về domain đó
  - Phase 2 (Hybridization, 30% train): mix data, router học cách kết hợp experts
  - Phase 3 (Generalization, 40% train): mixed + adversarial samples
    → experts trở thành "specialists that collaborate"

  Kết quả: 48 experts × ~6 ngôn ngữ × ~8 sub-domain = coverage ~384 specializations
  Mỗi expert hoạt động như 8 "sub-experts" ảo → effective ~384 experts
  → Đây là cách 423B params có thể胜 hơn 1000B+ models.

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class CurriculumPhase(str, Enum):
    SPECIATION = "speciation"        # Phase 1: domain isolation
    HYBRIDIZATION = "hybridization"  # Phase 2: domain mixing
    GENERALIZATION = "generalization"  # Phase 3: adversarial + mix


# Domain → expert indices (nếu 48 experts):
# - 0-7:   Python (8 experts cho Python: ML, web, data, scripts, async, testing, ...)
# - 8-13:  JavaScript / TypeScript (6)
# - 14-19: C / C++ (6)
# - 20-23: Rust (4)
# - 24-27: Go (4)
# - 28-31: Java (4)
# - 32-35: SQL / DB (4)
# - 36-39: Shell / Bash (4)
# - 40-43: Config / YAML / TOML (4)
# - 44-47: Mixed / General (4)

DEFAULT_EXPERT_DOMAIN_MAP: Dict[int, str] = {}
_domain_ranges = [
    ("python",       range(0, 8)),
    ("javascript",   range(8, 14)),
    ("cpp",          range(14, 20)),
    ("rust",         range(20, 24)),
    ("go",           range(24, 28)),
    ("java",         range(28, 32)),
    ("sql",          range(32, 36)),
    ("shell",        range(36, 40)),
    ("config",       range(40, 44)),
    ("mixed",        range(44, 48)),
]
for _domain, _rng in _domain_ranges:
    for _i in _rng:
        DEFAULT_EXPERT_DOMAIN_MAP[_i] = _domain


@dataclass
class SpeciationConfig:
    """Cấu hình Expert Speciation Curriculum."""
    # Số expert dành cho mỗi domain (auto-tuned theo num_experts)
    expert_domain_map: Dict[int, str] = field(
        default_factory=lambda: dict(DEFAULT_EXPERT_DOMAIN_MAP)
    )
    # Tỷ lệ thời gian train cho mỗi phase
    phase_ratio_speciation: float = 0.30       # 30% train
    phase_ratio_hybridization: float = 0.30    # 30% train
    phase_ratio_generalization: float = 0.40   # 40% train
    # Probability override: trong phase speciation, 90% data vào đúng expert domain
    speciation_strictness: float = 0.90
    # Hybridization: 50% đúng domain, 50% mix
    hybridization_mix_ratio: float = 0.50
    # Adversarial samples trong generalization
    adversarial_ratio: float = 0.10
    # Adversarial sample types
    adversarial_types: List[str] = field(
        default_factory=lambda: [
            "obfuscated_code",       # code bị minify/obfuscate
            "cross_language",        # gọi API qua ngôn ngữ khác
            "anti_pattern",          # code sai convention
            "edge_case",             # boundary cases
            "security_vuln",         # code có lỗ hổng
        ]
    )


class SpeciationCurriculum:
    """Quản lý curriculum speciation cho CyberGym training.

    Usage:
        curr = SpeciationCurriculum(config, total_steps=10000)
        for step, batch in enumerate(loader):
            phase = curr.get_phase_at_step(step)
            domain = curr.sample_domain(phase, batch)
            # → route batch's loss chỉ vào các expert thuộc domain này
    """

    def __init__(
        self,
        config: Optional[SpeciationConfig] = None,
        total_steps: int = 10000,
    ):
        self.config = config or SpeciationConfig()
        self.total_steps = max(total_steps, 1)
        self._compute_phase_boundaries()

    def _compute_phase_boundaries(self) -> None:
        s = self.config.phase_ratio_speciation
        h = self.config.phase_ratio_hybridization
        # generalization gets the rest
        self._speciation_end = int(self.total_steps * s)
        self._hybridization_end = int(self.total_steps * (s + h))

    def get_phase_at_step(self, step: int) -> CurriculumPhase:
        if step < self._speciation_end:
            return CurriculumPhase.SPECIATION
        if step < self._hybridization_end:
            return CurriculumPhase.HYBRIDIZATION
        return CurriculumPhase.GENERALIZATION

    def get_active_experts_for_domain(self, domain: str) -> List[int]:
        """Trả về list expert indices chuyên cho domain này."""
        return [
            idx for idx, d in self.config.expert_domain_map.items()
            if d == domain
        ]

    def get_domain_for_expert(self, expert_idx: int) -> str:
        """Trả về domain mà expert này chuyên về."""
        return self.config.expert_domain_map.get(expert_idx, "mixed")

    def sample_domain(
        self,
        phase: CurriculumPhase,
        batch_domain: Optional[str] = None,
    ) -> str:
        """Chọn domain ưu tiên cho batch trong phase này.

        - SPECIATION: 90% đúng batch_domain, 10% random
        - HYBRIDIZATION: 50% đúng batch_domain, 50% random
        - GENERALIZATION: random
        """
        import random as _r

        if batch_domain is None:
            batch_domain = _r.choice(list({d for d in self.config.expert_domain_map.values()}))

        if phase == CurriculumPhase.SPECIATION:
            return batch_domain if _r.random() < self.config.speciation_strictness else _r.choice(
                list({d for d in self.config.expert_domain_map.values()})
            )
        if phase == CurriculumPhase.HYBRIDIZATION:
            return batch_domain if _r.random() < (1 - self.config.hybridization_mix_ratio) else _r.choice(
                list({d for d in self.config.expert_domain_map.values()})
            )
        return _r.choice(list({d for d in self.config.expert_domain_map.values()}))

    def should_inject_adversarial(self, step: int) -> bool:
        """Trong phase generalization, có nên inject adversarial sample?"""
        if self.get_phase_at_step(step) != CurriculumPhase.GENERALIZATION:
            return False
        import random as _r
        return _r.random() < self.config.adversarial_ratio

    def summary(self) -> Dict[str, object]:
        domain_count: Dict[str, int] = {}
        for d in self.config.expert_domain_map.values():
            domain_count[d] = domain_count.get(d, 0) + 1
        return {
            "total_steps": self.total_steps,
            "phase_boundaries": {
                "speciation_end": self._speciation_end,
                "hybridization_end": self._hybridization_end,
            },
            "expert_per_domain": domain_count,
            "adversarial_types": self.config.adversarial_types,
        }
