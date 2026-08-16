"""
Nexus Coder - AI Agent với kiến trúc MoE
=========================================
v0.2.0 - Major Upgrade

Model AI được tạo bởi Hieu Louis (2026)

Tổng tham số: ~10 tỷ (10B)
Tham số kích hoạt: ~1.5 tỷ (1.5B active)
Cửa sổ ngữ cảnh: 50,000 tokens
Kiến trúc: Mixture of Experts (MoE) Transformer

Tác giả: Hieu Louis
GitHub:  mhieuhonda
Năm:     2026
"""

__version__ = "0.2.0"
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
    "version": "0.2.0",
    "architecture": "MoE Transformer (GQA + RoPE + RMSNorm + SwiGLU)",
    "total_params": "~10.22B",
    "active_params": "~1.50B",
    "context_window": "50,000 tokens",
    "python_version": "3.12.13",
}

# Lazy import để giảm startup time
def __getattr__(name: str):
    if name == "NexusConfig":
        from .config import NexusConfig
        return NexusConfig
    if name == "AUTHOR_INFO":
        return AUTHOR_INFO
    raise AttributeError(f"module 'nexus' has no attribute {name!r}")


__all__ = [
    "AUTHOR_INFO",
    "__version__",
    "__author__",
    "__github__",
    "__year__",
    "__license__",
]
