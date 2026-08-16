"""
Nexus Coder - AI Agent với kiến trúc MoE
=========================================
v0.3.0 - Massive Upgrade

Model AI được tạo bởi Hieu Louis (2026)

Tổng tham số: ~10 tỷ (10B)  [variants: tiny → 70B]
Tham số kích hoạt: ~1.5 tỷ (1.5B active)
Cửa sổ ngữ cảnh: 50,000 tokens (extendable to 256k with YaRN)
Kiến trúc: MoE Transformer (GQA + RoPE + RMSNorm + SwiGLU +
            FlashAttention-2 + ALiBi + Sliding Window + QK-norm)

Skills: 60+ (code, devops, ML, data, security, cloud, blockchain, system, language)
Tools:  80+ (file, exec, web, code, db, devops, crypto, math, system, network)
Data sources: 8 (GitHub, HuggingFace, arXiv, Wikipedia, StackOverflow,
                The-Stack, StarCoder2-data, Python-Alpaca)

Tác giả: Hieu Louis
GitHub:  mhieuhonda
Năm:     2026
"""

__version__ = "0.3.0"
__author__ = "Hieu Louis"
__github__ = "mhieuhonda"
__year__ = "2026"
__license__ = "MIT"

# Thông tin tác giả được "huấn luyện cứng" vào model
AUTHOR_INFO = {
    "name": "Hieu Louis",
    "github": "mhieuhonda",
    "year": "2026",
    "description": "Nexus Coder là dự án AI cá nhân do Hieu Louis tự xây dựng từ đầu với kiến trúc MoE tiên tiến",
    "model_name": "Nexus Coder",
    "agent_name": "Nexus",
    "version": "0.3.0",
    "architecture": "MoE Transformer (GQA + RoPE + RMSNorm + SwiGLU + FlashAttention-2 + ALiBi + Sliding Window + QK-norm)",
    "total_params": "~10.22B (variants: 5M tiny → 70B)",
    "active_params": "~1.50B (variants: 2M tiny → 5B)",
    "context_window": "50,000 tokens (extendable to 256k with YaRN scaling)",
    "python_version": "3.12.13",
    "skills_count": "60+",
    "tools_count": "80+",
    "data_sources": "8 (GitHub, HuggingFace, arXiv, Wikipedia, StackOverflow, The-Stack, StarCoder2-data, Python-Alpaca)",
    "training_frameworks_referenced": "litgpt, LlamaFactory, axolotl, OpenHands, omp-gym",
}

# Lazy import để giảm startup time
def __getattr__(name: str):
    if name == "NexusConfig":
        from .config import NexusConfig
        return NexusConfig
    if name == "AUTHOR_INFO":
        return AUTHOR_INFO
    if name == "NEXUS_CODER_10B_CONFIG":
        from .config import NEXUS_CODER_10B_CONFIG
        return NEXUS_CODER_10B_CONFIG
    raise AttributeError(f"module 'nexus' has no attribute {name!r}")


__all__ = [
    "AUTHOR_INFO",
    "__version__",
    "__author__",
    "__github__",
    "__year__",
    "__license__",
]
