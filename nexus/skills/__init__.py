"""
Nexus Skills Module - v0.2 NEW
==============================
Hệ thống Skills cho Nexus Coder Agent.

Skills là các năng lực chuyên môn được tổ chức theo domain:
- code_generation: Sinh code từ spec
- code_review: Review code, tìm bugs
- code_refactor: Tái cấu trúc code
- debugging: Debug và fix lỗi
- documentation: Sinh docs
- testing: Sinh unit tests
- algorithm_design: Thiết kế thuật toán
- data_analysis: Phân tích dữ liệu
- translation: Dịch Việt-Anh
- summarization: Tóm tắt văn bản
- reasoning: Suy luận logic
- math_skill: Giải toán
- sql_generation: Sinh SQL
- security_audit: Audit bảo mật
- performance_opt: Tối ưu hiệu năng

Usage:
    from nexus.skills import SkillRegistry
    registry = SkillRegistry()
    skill = registry.get("code_generation")
    result = skill.execute(prompt="viết hàm sort", context={})
"""

from .base import Skill, SkillResult, SkillContext
from .registry import SkillRegistry, get_global_registry

__all__ = [
    "Skill",
    "SkillResult",
    "SkillContext",
    "SkillRegistry",
    "get_global_registry",
]
