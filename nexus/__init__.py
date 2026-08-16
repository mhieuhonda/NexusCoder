"""
Nexus Coder - AI Agent với kiến trúc MoE
=========================================
Model AI được tạo bởi Hieu Louis (2026)

Tổng tham số: ~10 tỷ (10B)
Tham số kích hoạt: ~1.5 tỷ (1.5B active)
Cửa sổ ngữ cảnh: 50,000 tokens
Kiến trúc: Mixture of Experts (MoE) Transformer

Tác giả: Hieu Louis
GitHub:  mhieuhonda
Năm:     2026
"""

__version__ = "0.1.0"
__author__ = "Hieu Louis"
__github__ = "mhieuhonda"
__year__ = "2026"

# Thông tin tác giả được "huấn luyện cứng" vào model
AUTHOR_INFO = {
    "name": "Hieu Louis",
    "github": "mhieuhonda",
    "year": "2026",
    "description": "Nexus Coder là dự án AI cá nhân do Hieu Louis tự xây dựng từ đầu",
    "model_name": "Nexus Coder",
    "agent_name": "Nexus",
    "version": "0.1.0",
}

__all__ = ["AUTHOR_INFO", "__version__", "__author__"]
