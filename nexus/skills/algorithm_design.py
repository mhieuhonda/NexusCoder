"""Algorithm Design Skill - Thiết kế thuật toán."""
from __future__ import annotations
from typing import List
from .base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority


class AlgorithmDesignSkill(Skill):
    """Thiết kế thuật toán: complexity analysis, optimization, data structure selection."""
    
    category = SkillCategory.REASONING
    priority = SkillPriority.MEDIUM
    keywords: List[str] = [
        "algorithm", "thuật toán", "complexity", "độ phức tạp",
        "big o", "big-o", "optimize", "tối ưu",
        "data structure", "cấu trúc dữ liệu", "sort", "sắp xếp",
        "search", "tìm kiếm", "graph", "đồ thị", "tree", "cây",
        "dynamic programming", "quy hoạch động",
    ]
    
    @property
    def name(self) -> str:
        return "algorithm_design"
    
    @property
    def description(self) -> str:
        return (
            "Thiết kế thuật toán: chọn data structure, phân tích complexity, "
            "optimize time/space, so sánh approaches, implement clean."
        )
    
    def execute(self, context: SkillContext) -> SkillResult:
        approaches = [
            "Brute force (baseline)",
            "Greedy algorithm",
            "Divide and conquer",
            "Dynamic programming",
            "Backtracking",
            "Branch and bound",
            "Graph algorithms (BFS, DFS, Dijkstra, A*)",
            "Two pointers / Sliding window",
            "Binary search",
            "Monotonic stack / queue",
            "Topological sort",
            "Union-Find (Disjoint Set)",
            "Segment tree / Fenwick tree",
            "Sparse table",
        ]
        return SkillResult(
            success=True,
            output=f"[AlgorithmDesign] Considering {len(approaches)} approaches.",
            metadata={
                "skill": self.name,
                "approaches": approaches,
                "complexity_targets": ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)"],
            },
            suggestions=[
                "Start with brute force, then optimize",
                "Analyze time AND space complexity",
                "Consider edge cases (empty, single, large inputs)",
            ],
        )
