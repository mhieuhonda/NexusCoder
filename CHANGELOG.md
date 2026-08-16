# Thay đổi / Changelog

## v0.2.0 - 2026-08-16

### 🚀 Major Upgrade - Skills, Tools, và Data Pipeline

**Tác giả / Author**: Hieu Louis

#### ✨ Tính năng mới / New Features

##### 🎯 Skills System (15 skills)
- ✅ `code_generation` - Sinh code từ mô tả (Python, JS, Go, Rust, SQL, ...)
- ✅ `code_review` - Review code: bugs, security, performance
- ✅ `code_refactor` - Tái cấu trúc code (extract, rename, patterns)
- ✅ `debugging` - Debug đa ngôn ngữ với 7-step protocol
- ✅ `documentation` - Sinh docstrings, README, API docs
- ✅ `testing` - Unit/integration/E2E/property/mutation tests
- ✅ `algorithm_design` - Thiết kế thuật toán, complexity analysis
- ✅ `data_analysis` - EDA, statistics, visualization
- ✅ `translation` - Dịch song ngữ Việt-Anh
- ✅ `summarization` - Extractive + abstractive summarization
- ✅ `reasoning` - CoT, ToT, ReAct, self-consistency
- ✅ `math_skill` - Algebra, calculus, linear algebra, statistics
- ✅ `sql_generation` - SQL cho 7 dialects (Postgres, MySQL, ...)
- ✅ `security_audit` - OWASP Top 10, SAST, dependency scan
- ✅ `performance_optimization` - Profiling, bottleneck, optimization

##### 🔧 Tools System (15+ tools)
- ✅ `file_read` / `file_write` / `file_list` / `file_delete` - File operations
- ✅ `shell_exec` - Execute bash commands (sandboxed)
- ✅ `python_exec` - Execute Python code (restricted namespace)
- ✅ `git_ops` - Git commands với safety classification
- ✅ `http_request` - HTTP GET/POST/PUT/DELETE
- ✅ `web_fetch` - Fetch webpage, extract text
- ✅ `web_search` - Web search (Google/Bing/Brave API)
- ✅ `code_search` - Regex search trong code files
- ✅ `code_lint` / `code_format` - Lint & format code
- ✅ `calculator` - Safe math expression eval
- ✅ `json_parse` / `yaml_parse` / `csv_parse` - Data parsers
- ✅ `regex_search` - Regex search trong files
- ✅ `archive` - ZIP/TAR create/extract/list
- ✅ `hash` / `encrypt` - Hashing & AES-256-GCM encryption
- ✅ `datetime` - DateTime operations + timezone convert
- ✅ `dns_lookup` / `ping` - Network diagnostics

##### 📊 Training Data Pipeline
- ✅ `GitHubCollector` - Thu thập code từ 60+ curated GitHub repos
- ✅ `HuggingFaceCollector` - 20+ curated HF datasets (code, text, Vietnamese)
- ✅ `ArxivCollector` - Scientific papers từ arXiv API
- ✅ `WikipediaCollector` - Vietnamese + English Wikipedia
- ✅ `StackOverflowCollector` - Q&A từ StackOverflow API
- ✅ `TextCleaner` - HTML stripping, unicode normalize, whitespace cleanup
- ✅ `CodeFormatter` - Format code samples, detect language
- ✅ `Deduplicator` - MinHash LSH for near-duplicate detection
- ✅ `QualityFilter` - Quality scoring (length, diversity, repetition)
- ✅ `CurriculumLearning` - 4-stage curriculum (easy → expert)

##### 🧠 Agent Upgrades
- ✅ `SkillRegistry` - Auto-routing prompt → best skill
- ✅ `ToolRegistry` - Safety-gated tool execution + audit log
- ✅ `ConversationMemory` - Sliding window + summarization
- ✅ `TaskPlanner` - Multi-step task decomposition
- ✅ `ToolRouter` - Detect tool calls from natural language
- ✅ `SafetyFilter` - Content filter + PII detection
- ✅ `GuardrailManager` - Configurable safety rules

##### ⚡ Optimization Module
- ✅ `Quantizer` - INT8/INT4/FP8 quantization
- ✅ `LoRA` - Low-Rank Adaptation (rank 8/16/32/64)
- ✅ `Distiller` - Knowledge distillation (Hinton et al.)
- ✅ `Pruner` - Structured/unstructured pruning

##### 🛡️ Safety Module
- ✅ `ContentFilter` - Detect harmful content (violence, hate, self-harm)
- ✅ `PIIFilter` - Detect & mask PII (email, phone, SSN, API keys)
- ✅ `Guardrails` - Configurable rules with allow/warn/block/redact

##### 📈 Evaluation Module
- ✅ `BenchmarkSuite` - 8 benchmarks (HumanEval, MBPP, GSM8K, MMLU, BBH, ...)
- ✅ Metrics: Perplexity, BLEU, ROUGE, F1

##### 📚 Multi-Variant Configs
- ✅ `tiny` - ~5M params (CPU demo)
- ✅ `small` - ~125M params (1 GPU fine-tune)
- ✅ `medium` - ~1B params (4-8 GPU pretrain)
- ✅ `large` - 10B/1.5B (default, 32+ GPU)
- ✅ `xlarge` - ~30B/3B (research only)

##### 📝 Training Data Expansion
- ✅ Hardcoded data: 25 → 60+ examples
- ✅ Categories: author info, code, math, reasoning, Vietnamese culture, SQL, debugging, tools
- ✅ External data loading from JSONL files
- ✅ Combined dataset API

#### 🔧 Cải tiến / Improvements

- ✅ **Modular architecture**: 8 modules (model, tokenizer, training, inference, agent, skills, tools, data, optim, safety, eval)
- ✅ **Lazy imports**: Faster startup time
- ✅ **Type hints**: Full typing throughout
- ✅ **Docstrings**: Comprehensive documentation
- ✅ **Safety first**: All dangerous operations require confirmation
- ✅ **Audit logging**: All tool calls logged to JSONL
- ✅ **Configurable**: Every component has dataclass config

#### 📊 Thông số kỹ thuật / Technical Specs

| Thông số | Giá trị |
|----------|---------|
| Version | 0.2.0 |
| Skills | 15 |
| Tools | 18+ |
| Data sources | 5 (GitHub, HuggingFace, arXiv, Wikipedia, StackOverflow) |
| Curated repos | 60+ |
| Curated datasets | 20+ |
| Configs | 5 variants (tiny → xlarge) |
| Python version | 3.12.13 |
| PyTorch | >= 2.0 |

#### 📁 Cấu trúc thư mục v0.2

```
NexusCoder/
├── nexus/
│   ├── __init__.py
│   ├── config.py              # Multi-variant configs
│   ├── model/                 # MoE Transformer
│   ├── tokenizer/             # BPE Tokenizer
│   ├── training/              # Trainer + Dataset
│   ├── inference/             # Generator
│   ├── agent/                 # Agent + Memory + Planner + Router
│   ├── skills/                # 15 skills (NEW)
│   ├── tools/                 # 18+ tools (NEW)
│   ├── data/                  # Data pipeline (NEW)
│   │   ├── collectors/        # 5 collectors
│   │   └── processors/        # 4 processors
│   ├── optim/                 # Quantization, LoRA, Distill, Prune (NEW)
│   ├── safety/                # Filters, Guardrails (NEW)
│   ├── eval/                  # Benchmarks, Metrics (NEW)
│   └── utils/
├── configs/                   # YAML configs
├── scripts/                   # CLI scripts (7 scripts)
├── tests/
└── docs/
```

#### 🚀 Migration từ v0.1

v0.2 backward compatible với v0.1:
- `NexusConfig()` vẫn hoạt động (default = large 10B)
- `NexusAgent()` vẫn hoạt động (nhưng có thêm features)
- `AUTHOR_TRAINING_DATA` vẫn có (nhưng mở rộng)
- `scripts/train.py` vẫn hoạt động (nhưng có thêm flags)

---

## v0.1.0 - 2026-08-16

### 🎉 Initial Release - Foundation

**Tác giả / Author**: Hieu Louis

#### Thêm mới / Added

- ✅ Kiến trúc **Mixture of Experts (MoE)** với 24 experts, 3 active mỗi token
- ✅ Tổng **10 tỷ tham số (10B)** với chỉ **1.5 tỷ tham số active (1.5B)** mỗi token
- ✅ **Cửa sổ ngữ cảnh 50,000 tokens** với RoPE
- ✅ **Grouped Query Attention (GQA)** - 16 heads, 4 KV heads
- ✅ **RMSNorm** + **SwiGLU** activation
- ✅ **BPE Tokenizer** song ngữ Việt-Anh
- ✅ **Training script** với AdamW + cosine LR schedule
- ✅ **Inference engine** với top-k, top-p, temperature sampling
- ✅ **AI Agent wrapper** (Nexus Agent) với quản lý hội thoại
- ✅ **Hardcoded author info** - model luôn nhớ được tạo bởi Hieu Louis
- ✅ **Test suite** đầy đủ
- ✅ **Song ngữ Việt-Anh** trong README và giao tiếp
- ✅ **MIT License**
- ✅ Tương thích **Python 3.12.13**

#### Tác giả / Author

**Hieu Louis** · GitHub: [@mhieuhonda](https://github.com/mhieuhonda) · 2026
