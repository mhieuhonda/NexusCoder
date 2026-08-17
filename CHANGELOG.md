# Thay đổi / Changelog

## v0.4.0 - 2026-08-17 — CyberForge Edition

### SUPREME UPGRADE — 423B params, 3M context, CyberGym training methodology

**Tác giả / Author**: Hieu Louis

#### New Features

##### Model architecture — 423B / 39B / 3M context
- New config `423b` (DEFAULT for v0.4): 423B total / 39B active params
- 24 layers, hidden 7168, 48 experts (4 active), inter 16384
- 3,000,000-token context window via YaRN RoPE scaling (×60)
- Sliding window 32k + QK-norm + KV cache int8 + gradient checkpointing
- Adaptive Density Routing: top-2 → top-8 active experts based on input entropy

##### CyberGym training methodology (NEW)
- **Code Genome Initialization (CGI)**: weight init from code motifs
- **Mutation Pressure Training (MPT)**: beneficial weight perturbations during training
- **Expert Speciation Curriculum (ESC)**: 48 experts → 48 species (Python/JS/Rust/Go/...)
- **Recursive Self-Compression (RSC)**: periodic self-distillation snapshots
- **Context Expansion Protocol (CEP)**: progressive 32k → 3M context extension
- **Adaptive Density Routing (ADR)**: entropy-based top-k routing
- Orchestrator `CyberForgeTrainer` wires all components together

##### Data pipeline — Code corpus curated
- `configs/code_corpus.yaml`: 1000+ curated GitHub repos across 17 categories
- Categories: python_core, python_web, python_data, python_ml, python_dl,
  python_tools, javascript_core, javascript_frameworks, rust_core, go_core,
  java_core, c_cpp, devops, security, ai_tools, scientific, systems

#### Bug Fixes (48 total)

##### CRITICAL (6 fixes)
- `nexus/safety/__init__.py`: missing `get_default_guardrails` export broke `nexus.agent`
- `nexus/data/processors/deduplicator.py`: wrong import path (`.._logging_helpers` → `...utils.logging`)
- `nexus/model/attention.py`: INT8 KV cache quantization discarded scale → crash on 2nd decode step
- `scripts/collect_data.py`: `CURATED_TAGS` was a class attribute, not module-level → ImportError
- `nexus/agent/planner.py`: invalid dependency IDs silently treated as "met" (security bug)
- `nexus/config.py`: 30B / 70B configs were 5×–9× off their advertised size

##### MAJOR (22 fixes)
- MoE never received `attention_mask` (padded tokens polluted aux loss)
- LoRA `target_modules` listed `gate_proj`/`up_proj` but v0.3 SwiGLU fuses them into `gate_up_proj`
- `python_exec` sandbox: when run as script, `__builtins__` was a module → sandbox escape
- `python_exec`: timeout was computed but never enforced → infinite loops could hang the agent
- `shell.py`: dead `if False` branch with unimported `os`
- ALiBi `max_slope` parameter was hardcoded to 8.0 (parameter had no effect)
- ALiBi non-power-of-2 head count subselection was wrong (took first N, not closest N)
- GitHub collector: `"c++"` language key didn't exist in EXTENSIONS (should be `"cpp"`)
- GitHub collector: hardcoded `--branch main` failed for repos using `master`
- arXiv collector: `.find().text` without None check crashed entire parse on missing element
- arXiv collector: query string not URL-encoded
- `compute_rouge`: rouge_1 was precision, not recall (corrected to F1)
- `compute_bleu`: empty references list crashed `min()` call
- FP8 quantization skip_layers comparison never matched (all params got FP8-quantized)
- Attention mask shape mismatch with KV cache + sliding window
- Trainer: AMP scaler state not checkpointed (resume caused NaN gradients)
- `quality_filter`: off-by-one in 10-gram repetition window
- `dataset.py`: hardcoded pad id 0 (collided with token 0 if user changed `pad_token_id`)
- Tokenizer: Vietnamese char `Ẵ` was duplicated as `Ẳ` (missing `Ẵ`)
- Tokenizer: BPE merge lost `</w>` marker when first symbol had it
- Tokenizer: `tuple(k.split("|"))` broke when token contained `|`
- `scripts/train.py`: `--config` choices missing `30b`, `70b`, `423b`

##### MINOR (20 fixes)
- Various unused imports, dead code, type hints
- See git log for full list

#### License change
- Switched from MIT to **NexusCoder Attribution License v1.0 (NAL-1.0)**
- Free use for any purpose (commercial/non-commercial/research)
- Mandatory attribution: "Hieu Louis" + link to original repo
- See [LICENSE](LICENSE) for full terms

#### Files added
- `nexus/cybergym/__init__.py`
- `nexus/cybergym/mutation.py`
- `nexus/cybergym/genome.py`
- `nexus/cybergym/adaptive_routing.py`
- `nexus/cybergym/speciation.py`
- `nexus/cybergym/compression.py`
- `nexus/cybergym/context_expansion.py`
- `nexus/cybergym/trainer.py`
- `configs/nexus_coder_423b.yaml`
- `configs/code_corpus.yaml`
- `ADVERTISEMENT.txt`

---

## v0.3.0 - 2026-08-16

### 🚀 MASSIVE UPGRADE - Architecture + 4× Skills + 4× Tools + Massive Data

**Tác giả / Author**: Hieu Louis

#### ✨ Tính năng mới / New Features

##### 🏗️ Kiến trúc v0.3 (NEW)
- ✅ **FlashAttention-2**: Optional `flash_attn` package backend (falls back to SDPA)
- ✅ **ALiBi position bias**: Alternative to RoPE for long-context extrapolation (Press et al., 2022)
- ✅ **Sliding Window Attention**: Alternating SWA / global layers (Longformer / Mistral style)
- ✅ **QK-norm**: RMSNorm on query/key for training stability (Llama-3 style)
- ✅ **MLP-parallel**: Fused gate+up projection (concatenated matmul) — faster on modern GPUs
- ✅ **KV cache quantization**: int8 / fp8 options for inference memory reduction
- ✅ **Gradient checkpointing**: Trade compute for VRAM at training time
- ✅ **RoPE scaling strategies**: linear / dynamic (NTK) / ntk / yarn — supports context extension up to 256k

##### 📊 Multi-Variant Configs (7 variants)
- ✅ `tiny` - ~5M params (CPU demo)
- ✅ `small` - ~125M params (1 GPU)
- ✅ `medium` - ~1B params (4-8 GPU)
- ✅ `large` - 10B/1.5B (default, 32+ GPU)
- ✅ `xlarge` - ~30B/3B (research, 64+ GPU)
- ✅ `30b` - 30B/3B (v0.3 NEW, 64-128 H100, 64k context)
- ✅ `70b` - 70B/5B (v0.3 NEW, 256+ H100/H200, 128k context with YaRN ×4)

##### 🎯 Skills System (15 → 60+)
- ✅ **Existing 15**: code_generation, code_review, code_refactor, debugging, documentation, testing, algorithm_design, data_analysis, translation, summarization, reasoning, math_skill, sql_generation, security_audit, performance_opt
- ✅ **DevOps (5 NEW)**: devops_skill, ci_cd_pipeline, release_management, monitoring, logging_analytics
- ✅ **ML (10 NEW)**: ml_training, ml_inference, ml_evaluation, ml_data_preprocessing, ml_feature_engineering, ml_hyperparameter_tuning, ml_model_explainability, ml_model_selection, ml_metrics, anomaly_detection
- ✅ **Data (5 NEW)**: data_pipeline, statistical_analysis, time_series_forecasting, clustering_analysis, knowledge_graph
- ✅ **Code (10 NEW)**: code_translation, code_completion, code_explanation, code_minification, code_documentation_generation, code_duplication_detection, code_dead_code_analysis, code_complexity_analysis, code_dependency_analysis, bug_reproduction
- ✅ **System (4 NEW)**: system_design, api_design, graphql_skill, microservices
- ✅ **Language (5 NEW)**: prompt_engineering, sentiment_analysis, topic_modeling, language_detection, creative_writing
- ✅ **Cloud (1 NEW)**: cloud_deploy
- ✅ **Blockchain (1 NEW)**: blockchain_audit
- ✅ **Caching (1 NEW)**: caching_strategy
- ✅ **Classification (1 NEW)**: classification_automation
- ✅ **Regex (1 NEW)**: regex_master
- ✅ **Shell (1 NEW)**: shell_scripting

##### 🔧 Tools System (18+ → 80+)
- ✅ **Existing 24**: file_read/write/list/delete, shell_exec, python_exec, git_ops, http_request, web_fetch, web_search, code_search/lint/format, calculator, json/yaml/csv_parse, regex_search, archive, hash, encrypt, datetime, dns_lookup, ping
- ✅ **Database (12 NEW)**: sql_runner, sql_formatter, sql_migrator, postgres, mysql, sqlite, redis, mongo, elasticsearch, kafka, rabbitmq, graphql_client
- ✅ **DevOps/Cloud (12 NEW)**: docker, kubectl, terraform, ansible, aws_cli, gcloud_cli, azure_cli, ssh, scp, rsync, systemd, crontab
- ✅ **Code analysis (13 NEW)**: code_ast, code_complexity, code_dependency, code_metrics, code_smells, code_formatter_advanced, code_minifier, code_transpiler, code_runner, code_tester, code_compiler, code_profiler, code_coverage
- ✅ **Web/Network (12 NEW)**: websocket_client, grpc_client, url_shortener, dns_query, traceroute_tool, port_scanner, ssl_checker, ssl_generator, cert_checker, web_scraper, web_crawler, web_auth
- ✅ **Misc/Convert/Security (13 NEW)**: jwt_tool, oauth_tool, api_key_validator, markdown_converter, pdf_generator, image_processor, statistics_tool, linear_algebra_tool, probability_tool, ml_metrics_tool, model_evaluator, benchmark_runner, log_analyzer

##### 📊 Data Pipeline (5 → 8 sources, 60 → 500+ repos)
- ✅ `GitHubCollector` (expanded): 60+ → 500+ curated repos (Python, JS, TS, Go, Rust, C/C++, Java, C#, Ruby, PHP, Swift, Kotlin, ...)
- ✅ `HuggingFaceCollector` (expanded): 20+ → 150+ datasets (code, instruction, math, Vietnamese, multilingual)
- ✅ `ArxivCollector`: 20 → 40 queries
- ✅ `WikipediaCollector`: 18 → 50+ topics per language
- ✅ `StackOverflowCollector`: 30 → 47 tags
- ✅ `TheStackCollector` (v0.3 NEW): BigCode's The-Stack v2 (~600 languages)
- ✅ `StarCoder2Collector` (v0.3 NEW): github_code + commits + jupyter notebooks
- ✅ `PythonAlpacaCollector` (v0.3 NEW): aggregates 6 Python instruction datasets

##### 🧠 Processors (4 → 6)
- ✅ `TextCleaner`, `Deduplicator`, `QualityFilter`, `CodeFormatter` (existing)
- ✅ `LanguageIdProcessor` (v0.3 NEW): identifies vi/en/code, drops mislabeled
- ✅ `CodeQualityProcessor` (v0.3 NEW): scores Python 1-10 (docstring, type hints, no eval, etc.)

##### 🤝 Integrations (5 reference frameworks)
- ✅ `litgpt.py`: FusedLinear adapter (Apache 2.0, Lightning AI)
- ✅ `llamafactory.py`: dataset format converters (alpaca/sharegpt/chatml/completion → nexus)
- ✅ `axolotl.py`: AxolotlStyleConfig dataclass (typed training config schema)
- ✅ `openhands.py`: AgentLoop pattern (planner/executor/observer/reflector)
- ✅ `omp_gym.py`: OpenMP optimization benchmark tasks

##### 📈 Evaluation Module
- ✅ `BenchmarkSuite` - 10 benchmarks (HumanEval, MBPP, GSM8K, MMLU, BBH, MATH, ARC, TruthfulQA, AlpacaFarm, OMP-gym)
- ✅ Metrics: Perplexity, BLEU, ROUGE, F1, code-pass@k

#### 🔧 Cải tiến / Improvements

- ✅ **Auto-discovery registries**: Skills + Tools now scan directories dynamically — drop a `.py` file with a `Skill`/`Tool` subclass and it auto-registers
- ✅ **Stream-friendly training data**: `StreamingNexusDataset` for >1M example datasets (no RAM pressure)
- ✅ **Trimmed hardcoded data**: AUTHOR_TRAINING_DATA 150+ → 15 core examples (rest loaded from JSONL)
- ✅ **Lazy imports**: Faster startup; optional deps only imported when needed
- ✅ **Type hints**: Full typing throughout
- ✅ **Safety first**: All DANGEROUS/DESTRUCTIVE tools have `requires_confirmation=True` + `dry_run` support
- ✅ **Audit logging**: All tool calls logged to JSONL with timestamp, args, result, duration
- ✅ **Bilingual**: Vietnamese + English throughout

#### 📊 Thông số kỹ thuật / Technical Specs

| Thông số | v0.2 | v0.3 |
|----------|------|------|
| Version | 0.2.0 | 0.3.0 |
| Skills | 15 | 60+ |
| Tools | 18+ | 80+ |
| Data sources | 5 | 8 |
| Curated repos | 60+ | 500+ |
| Curated datasets | 20+ | 150+ |
| Configs | 5 | 7 |
| Reference frameworks | 0 | 5 |
| Attention backends | 1 (SDPA) | 3 (SDPA + FA2 + ALiBi) |
| Python version | 3.12.13 | 3.12.13 (strict) |
| PyTorch | >= 2.0 | >= 2.0 (>= 2.3 for 70b config) |

#### 📁 Cấu trúc thư mục v0.3 (key changes)

```
NexusCoder/
├── nexus/
│   ├── __init__.py                    # v0.3.0 metadata
│   ├── config.py                      # + 30b/70b configs + attention features
│   ├── model/
│   │   ├── attention.py               # + FA2, ALiBi, SWA, QK-norm, KV quant
│   │   ├── rope.py                    # + NTK/YaRN scaling
│   │   ├── flash_attention.py         # NEW
│   │   ├── alibi.py                   # NEW
│   │   ├── sliding_window.py          # NEW
│   │   ├── layers.py                  # + MLP-parallel SwiGLU
│   │   └── transformer.py             # + gradient checkpointing
│   ├── training/
│   │   └── dataset.py                 # trimmed + StreamingNexusDataset
│   ├── skills/                        # 60+ skills, auto-discovery registry
│   ├── tools/                         # 80+ tools, auto-discovery registry
│   ├── data/
│   │   ├── collectors/                # 8 collectors (3 NEW)
│   │   └── processors/                # 6 processors (2 NEW)
│   └── integrations/                  # NEW: 5 reference framework adapters
├── configs/
│   ├── nexus_coder_30b.yaml           # NEW
│   ├── nexus_coder_70b.yaml           # NEW
│   └── sources.yaml                   # expanded to 500+ repos, 150+ datasets
├── ATTRIBUTIONS.md                    # NEW
├── requirements.txt                   # + 30 new optional deps
├── pyproject.toml                     # v0.3.0 + extras groups
└── setup.py                           # v0.3.0
```

#### 🚀 Migration từ v0.2

v0.3 backward compatible với v0.2:
- `NexusConfig()` vẫn hoạt động (default = large 10B)
- `NexusAgent()` vẫn hoạt động
- `AUTHOR_TRAINING_DATA` vẫn có (nhưng được tinh gọn)
- `scripts/train.py` vẫn hoạt động (nhưng có thêm config 30b, 70b)

Breaking changes (minor):
- `nexus.skills.registry._auto_register_defaults` giờ dùng dynamic discovery thay vì hardcoded imports
- `nexus.tools.registry._auto_register_defaults` tương tự
- `AUTHOR_TRAINING_DATA` giảm từ 150+ xuống 15 mẫu (phần còn lại load từ `data/processed/*.jsonl`)

#### 📦 Dependencies mới

```bash
# Database tools
pip install sqlalchemy psycopg2-binary pymysql redis pymongo elasticsearch kafka-python pika

# Web/Network tools
pip install aiohttp websockets grpcio beautifulsoup4 lxml

# DevOps tools
pip install paramiko kubernetes docker

# Media/Convert tools
pip install Pillow reportlab markdown

# ML tools
pip install scikit-learn scipy transformers accelerate peft

# Crypto
pip install pyjwt

# GPU acceleration
pip install flash-attn --no-build-isolation

# All at once
pip install -e ".[all]"
```

---

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
