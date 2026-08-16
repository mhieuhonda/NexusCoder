# Skills Documentation

Nexus Coder v0.2 có 15 skills chuyên môn, được tổ chức theo 5 categories.

## Categories

| Category | Skills |
|----------|--------|
| CODE | code_generation, code_review, code_refactor, debugging, documentation, testing |
| REASONING | algorithm_design, reasoning, math_skill |
| LANGUAGE | translation, summarization |
| DATA | data_analysis, sql_generation |
| SECURITY | security_audit |
| DEVOPS | performance_optimization |

## Skill List

### 1. code_generation
- **Category**: CODE
- **Priority**: HIGH
- **Description**: Sinh code từ mô tả tự nhiên
- **Languages**: Python, JavaScript, TypeScript, Go, Rust, C++, Java, SQL
- **Example**: "Viết hàm Python tính fibonacci"

### 2. code_review
- **Category**: CODE
- **Priority**: HIGH
- **Description**: Review code toàn diện
- **Checks**: bugs, security, performance, style, error handling, type safety
- **Example**: "Review đoạn code này giúp tôi"

### 3. code_refactor
- **Category**: CODE
- **Priority**: MEDIUM
- **Description**: Refactor code an toàn
- **Patterns**: Extract Method/Class, Rename, Move, Replace Conditional, etc.
- **Example**: "Refactor hàm này cho clean hơn"

### 4. debugging
- **Category**: CODE
- **Priority**: CRITICAL
- **Description**: Debug code với 7-step protocol
- **Supports**: Python, JavaScript, Java, Go, Rust, C++, Ruby
- **Example**: "Fix lỗi IndexError trong hàm này"

### 5. documentation
- **Category**: CODE
- **Priority**: MEDIUM
- **Description**: Sinh tài liệu tự động
- **Types**: Docstrings (Google/NumPy/Sphinx), README, API ref, tutorials
- **Example**: "Sinh docstring cho hàm này"

### 6. testing
- **Category**: CODE
- **Priority**: HIGH
- **Description**: Sinh tests
- **Types**: unit, integration, E2E, property-based, mutation, fuzz, snapshot
- **Frameworks**: pytest, unittest, jest, vitest, mocha, cargo test, JUnit
- **Example**: "Viết unit tests cho class User"

### 7. algorithm_design
- **Category**: REASONING
- **Priority**: MEDIUM
- **Description**: Thiết kế thuật toán
- **Approaches**: Brute force, Greedy, D&C, DP, Backtracking, Graph algorithms
- **Example**: "Tối ưu thuật toán này từ O(n²) xuống O(n log n)"

### 8. data_analysis
- **Category**: DATA
- **Priority**: MEDIUM
- **Description**: Phân tích dữ liệu
- **Steps**: Loading, cleaning, statistics, correlation, outliers, visualization
- **Libraries**: pandas, numpy, scipy, matplotlib, seaborn, plotly
- **Example**: "Phân tích dataset này và tìm insights"

### 9. translation
- **Category**: LANGUAGE
- **Priority**: MEDIUM
- **Description**: Dịch song ngữ Việt-Anh
- **Pairs**: vi↔en, vi↔zh, vi↔ja, vi↔ko, vi↔fr
- **Example**: "Dịch đoạn văn này sang tiếng Anh"

### 10. summarization
- **Category**: LANGUAGE
- **Priority**: MEDIUM
- **Description**: Tóm tắt văn bản
- **Methods**: extractive, abstractive, key phrase, topic modeling
- **Example**: "Tóm tắt bài viết này trong 3 câu"

### 11. reasoning
- **Category**: REASONING
- **Priority**: HIGH
- **Description**: Suy luận đa bước
- **Strategies**: CoT, ToT, Self-Consistency, Reflexion, ReAct, Least-to-Most
- **Example**: "Tại sao bầu trời màu xanh?"

### 12. math_skill
- **Category**: REASONING
- **Priority**: HIGH
- **Description**: Giải toán đa cấp
- **Domains**: arithmetic, algebra, calculus, linear algebra, probability, statistics
- **Tools**: sympy, numpy, scipy
- **Example**: "Tính đạo hàm của x³ + 2x²"

### 13. sql_generation
- **Category**: DATA
- **Priority**: HIGH
- **Description**: Sinh SQL queries
- **Dialects**: PostgreSQL, MySQL, SQLite, SQL Server, Oracle, BigQuery, Snowflake
- **Example**: "Viết SQL tìm top 10 khách hàng"

### 14. security_audit
- **Category**: SECURITY
- **Priority**: CRITICAL
- **Description**: Audit bảo mật
- **Standards**: OWASP Top 10, SAST, dependency vulnerabilities
- **Tools**: bandit, semgrep, safety, pip-audit, trufflehog
- **Example**: "Audit code này cho security issues"

### 15. performance_optimization
- **Category**: DEVOPS
- **Priority**: MEDIUM
- **Description**: Tối ưu hiệu năng
- **Categories**: algorithmic, memory, concurrency, caching, I/O, Python-specific
- **Tools**: cProfile, line_profiler, memory_profiler, py-spy
- **Example**: "Tối ưu hàm này đang chạy chậm"

## Usage

```python
from nexus.skills import get_global_registry
from nexus.skills.base import SkillContext

registry = get_global_registry()

# List all skills
print(registry.list_skills())

# Route prompt to best skill
skill = registry.route("Viết hàm Python tính giai thừa")
print(f"Selected: {skill.name}")

# Execute skill
context = SkillContext(prompt="Viết hàm Python tính giai thừa")
result = skill.execute(context)
print(result.output)
```

## Custom Skills

Tạo skill tùy chỉnh:

```python
from nexus.skills.base import Skill, SkillResult, SkillContext, SkillCategory, SkillPriority

class MyCustomSkill(Skill):
    category = SkillCategory.CODE
    priority = SkillPriority.MEDIUM
    keywords = ["custom", "riêng"]
    
    @property
    def name(self) -> str:
        return "my_custom_skill"
    
    @property
    def description(self) -> str:
        return "My custom skill description"
    
    def execute(self, context: SkillContext) -> SkillResult:
        return SkillResult(success=True, output="Custom result")

# Register
from nexus.skills import get_global_registry
get_global_registry().register(MyCustomSkill())
```
