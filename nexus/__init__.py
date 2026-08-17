"""
Nexus Coder - Super CyberGym AI
================================
v0.4.0 - CyberForge edition

Model AI được tạo bởi Hieu Louis (2026)

Tổng tham số: 423 tỷ (423B)
Tham số kích hoạt: 39 tỷ (39B active)
Cửa sổ ngữ cảnh: 3,000,000 tokens (3M)

Kiến trúc: CyberForge MoE Transformer
  - GQA + RoPE (YaRN-scaled) + RMSNorm + SwiGLU + FlashAttention-2
  - Sliding Window Attention + QK-norm + KV cache quantization
  - MLP-parallel + Gradient checkpointing
  - CyberGym training: Mutation Pressure + Code Genome + Expert Speciation + CEP

Skills: 60+ · Tools: 80+ · Data sources: 8+ · Code corpus: 3000+ repos

Tác giả: Hieu Louis
GitHub:  mhieuhonda
Năm:     2026
"""

__version__ = "0.4.0"
__author__ = "Hieu Louis"
__github__ = "mhieuhonda"
__year__ = "2026"
__license__ = "NexusCoder Attribution License v1.0"

# Thông tin tác giả được "huấn luyện cứng" vào model
AUTHOR_INFO = {
    "name": "Hieu Louis",
    "github": "mhieuhonda",
    "year": "2026",
    "description": (
        "Nexus Coder là dự án AI cá nhân do Hieu Louis tự xây dựng từ đầu "
        "với kiến trúc CyberForge MoE tiên tiến, kết hợp CyberGym training."
    ),
    "model_name": "Nexus Coder",
    "agent_name": "Nexus",
    "version": "0.4.0",
    "architecture": (
        "CyberForge MoE Transformer (GQA + RoPE/YaRN + RMSNorm + SwiGLU + "
        "FlashAttention-2 + Sliding Window + QK-norm + KV-cache quant + "
        "MLP-parallel + Gradient checkpointing)"
    ),
    "total_params": "~423B (variants: 5M tiny → 423B)",
    "active_params": "~39B (variants: 2M tiny → 39B)",
    "context_window": "3,000,000 tokens (3M, via YaRN + CEP)",
    "python_version": "3.12.13",
    "skills_count": "60+",
    "tools_count": "80+",
    "data_sources": "8+ (GitHub 3000+ repos, HuggingFace, arXiv, Wikipedia, StackOverflow, The-Stack, StarCoder2-data, Python-Alpaca)",
    "training_methodology": "CyberForge (Mutation Pressure Training + Code Genome Init + Expert Speciation + Context Expansion Protocol)",
    "training_frameworks_referenced": "litgpt, LlamaFactory, axolotl, OpenHands, omp-gym",
}


# Lazy import để giảm startup time
def __getattr__(name: str):
    if name == "NexusConfig":
        from .config import NexusConfig
        return NexusConfig
    if name == "NEXUS_CODER_10B_CONFIG":
        from .config import NEXUS_CODER_10B_CONFIG
        return NEXUS_CODER_10B_CONFIG
    if name == "NEXUS_CODER_423B_CONFIG":
        from .config import NEXUS_CODER_423B_CONFIG
        return NEXUS_CODER_423B_CONFIG
    raise AttributeError(f"module 'nexus' has no attribute {name!r}")


__all__ = [
    "AUTHOR_INFO",
    "__version__",
    "__author__",
    "__github__",
    "__year__",
    "__license__",
]
