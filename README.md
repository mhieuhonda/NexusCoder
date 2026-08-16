<div align="center">

# 🧠 Nexus Coder

### AI Agent with Mixture of Experts (MoE) Architecture

**10B total parameters · 1.5B active parameters · 50K context window**
**15 skills · 18+ tools · 5 data sources · Multi-variant configs**

[![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen.svg)]()

**Created by [Hieu Louis](https://github.com/mhieuhonda)** · 2026

</div>

---

# 🇻🇳 Tiếng Việt

## Giới thiệu

**Nexus Coder v0.2** là bản nâng cấp lớn từ v0.1, bổ sung hệ thống Skills, Tools, Data Pipeline, và nhiều module chuyên nghiệp khác. Vẫn giữ nguyên kiến trúc **Mixture of Experts (MoE)** 10B/1.5B, nhưng nay mạnh mẽ hơn rất nhiều.

Được tạo ra bởi **Hieu Louis** vào năm 2026, Nexus Coder là dự án cá nhân đầy tham vọng, chứng minh khả năng xây dựng một mô hình AI hoàn chỉnh từ con số không.

## ✨ Tính năng chính v0.2

| Đặc điểm | Giá trị |
|----------|---------|
| **Tổng tham số** | ~10 tỷ (10B) |
| **Tham số kích hoạt** | ~1.5 tỷ (1.5B mỗi token) |
| **Cửa sổ ngữ cảnh** | 50,000 tokens |
| **Số chuyên gia (experts)** | 24 (chỉ 3 active mỗi token) |
| **Skills** | 15 (code, reasoning, math, language, data, security, devops) |
| **Tools** | 18+ (file, exec, web, code, math, parser, network, crypto) |
| **Data sources** | 5 (GitHub, HuggingFace, arXiv, Wikipedia, StackOverflow) |
| **Configs** | 5 variants (tiny → xlarge) |
| **Ngôn ngữ lập trình** | Python 3.12.13 |
| **Framework** | PyTorch |
| **Tính cách** | Hài hước, thân thiện |
| **Ngôn ngữ giao tiếp** | Song ngữ Việt - Anh |

## 🏗️ Kiến trúc v0.2

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEXUS CODER v0.2                               │
├─────────────────────────────────────────────────────────────────┤
│  Agent Layer                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Memory   │  │ Planner  │  │ Router   │  │ Safety   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  Skills (15)              │  Tools (18+)                         │
│  ┌─────────────────────┐  │  ┌──────────────────────────┐       │
│  │ Code Skills (6)     │  │  │ File Ops (4)             │       │
│  │ Reasoning (3)       │  │  │ Exec (3)                 │       │
│  │ Language (2)        │  │  │ Web (3)                  │       │
│  │ Data (2)            │  │  │ Code (4)                 │       │
│  │ Security (1)        │  │  │ Math/Parser/System (4+)  │       │
│  │ DevOps (1)          │  │  │ Network/Crypto (4)       │       │
│  └─────────────────────┘  │  └──────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│  Model (MoE Transformer)                                         │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Token Embedding (vocab=32K, hidden=2048)                  │ │
│  │  Decoder Layer × 12                                        │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  RMSNorm → Multi-Head Attention (GQA + RoPE)         │  │ │
│  │  │  RMSNorm → MoE Layer (24 experts, 3 active)          │  │ │
│  │  │  Residual connections                                 │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │  RMSNorm → LM Head (vocab=32K)                             │ │
│  └────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Data Pipeline                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Collect  │→ │ Process  │→ │  Train   │→ │ Evaluate │        │
│  │ (5 srcs) │  │ (4 stgs) │  │ (curric) │  │ (8 benc) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  Optimization & Safety                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Quantize │  │   LoRA   │  │ Distill  │  │  Pruner  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐                                    │
│  │ Safety   │  │Guardrails│                                    │
│  └──────────┘  └──────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Cài đặt

```bash
# Clone repository
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder

# Tạo môi trường Python 3.12.13
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 📖 Sử dụng

### 1. Multi-variant configs

```bash
# Tiny (CPU demo, ~5M params)
python scripts/train.py --config tiny --steps 100

# Small (1 GPU, ~125M params)
python scripts/train.py --config small --steps 1000

# Medium (4-8 GPU, ~1B params)
python scripts/train.py --config medium --steps 5000

# Large 10B (32+ GPU, default)
python scripts/train.py --config large --steps 5000 --use-amp

# XLarge 30B (research only)
python scripts/train.py --config xlarge --steps 10000 --use-amp
```

### 2. Collect training data

```bash
# Collect from all sources
python scripts/collect_data.py --source all --output ./data/raw

# Specific source
python scripts/collect_data.py --source github --max-repos 10
python scripts/collect_data.py --source huggingface --max-datasets 5
python scripts/collect_data.py --source arxiv --max-queries 5
python scripts/collect_data.py --source wikipedia --language vi
python scripts/collect_data.py --source stackoverflow --max-tags 5
```

### 3. Process dataset

```bash
python scripts/prepare_dataset.py --input ./data/raw --output ./data/processed
```

### 4. Train với external data

```bash
# Train với collected data
python scripts/train.py --config large --include-external --steps 5000

# LoRA fine-tune
python scripts/train.py --config large --lora --lora-rank 8 --steps 1000

# Resume từ checkpoint
python scripts/train.py --resume ./checkpoints/nexus_coder-step-1000.pt
```

### 5. Chat với Nexus Agent

```bash
python scripts/chat.py
```

Trong chat:
- `skills` - Liệt kê 15 skills
- `tools` - Liệt kê 18+ tools
- `stats` - Thống kê session
- `info` - Thông tin model
- `reset` - Xóa lịch sử

### 6. Quantize model

```bash
# INT8 (4x memory reduction)
python scripts/quantize_model.py --input model.pt --method int8 --output model_int8.pt

# INT4 (8x memory reduction)
python scripts/quantize_model.py --input model.pt --method int4 --output model_int4.pt
```

### 7. Evaluate

```bash
python scripts/evaluate.py --model model.pt --benchmarks humaneval,gsm8k
```

### 8. Python API

```python
from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.agent.agent import NexusAgent
from nexus.skills import get_global_registry as get_skills
from nexus.tools import get_global_registry as get_tools

# Khởi tạo
config = NexusConfig()
model = NexusCoderForCausalLM(config)

# Tạo Agent v0.2
agent = NexusAgent(config=config, name="Nexus")

# Skills & Tools ready
print(f"Skills: {len(get_skills())}")
print(f"Tools: {len(get_tools())}")

# Chat
agent.chat()
```

## 📁 Cấu trúc dự án v0.2

```
NexusCoder/
├── nexus/                          # Package chính
│   ├── __init__.py
│   ├── config.py                   # Multi-variant configs
│   ├── model/                      # Kiến trúc model
│   │   ├── attention.py
│   │   ├── moe.py
│   │   ├── transformer.py
│   │   ├── nexus_coder.py
│   │   ├── rope.py
│   │   └── layers.py
│   ├── tokenizer/                  # BPE Tokenizer
│   ├── training/                   # Training + Dataset
│   ├── inference/                  # Inference engine
│   ├── agent/                      # Agent + Memory + Planner + Router
│   │   ├── agent.py
│   │   ├── memory.py              # NEW
│   │   ├── planner.py             # NEW
│   │   └── router.py              # NEW
│   ├── skills/                     # 15 Skills (NEW)
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── code_generation.py
│   │   ├── code_review.py
│   │   ├── code_refactor.py
│   │   ├── debugging.py
│   │   ├── documentation.py
│   │   ├── testing.py
│   │   ├── algorithm_design.py
│   │   ├── data_analysis.py
│   │   ├── translation.py
│   │   ├── summarization.py
│   │   ├── reasoning.py
│   │   ├── math_skill.py
│   │   ├── sql_generation.py
│   │   ├── security_audit.py
│   │   └── performance_opt.py
│   ├── tools/                      # 18+ Tools (NEW)
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── file_ops.py
│   │   ├── shell.py
│   │   ├── python_exec.py
│   │   ├── git_ops.py
│   │   ├── web_tools.py
│   │   ├── code_tools.py
│   │   ├── calculator.py
│   │   ├── parsers.py
│   │   ├── search.py
│   │   ├── archive.py
│   │   ├── crypto.py
│   │   ├── datetime_tool.py
│   │   └── network.py
│   ├── data/                       # Data Pipeline (NEW)
│   │   ├── collectors/
│   │   │   ├── github_collector.py
│   │   │   ├── huggingface_collector.py
│   │   │   ├── arxiv_collector.py
│   │   │   ├── wikipedia_collector.py
│   │   │   └── stackoverflow_collector.py
│   │   ├── processors/
│   │   │   ├── cleaner.py
│   │   │   ├── deduplicator.py
│   │   │   ├── quality_filter.py
│   │   │   └── code_formatter.py
│   │   └── curriculum.py
│   ├── optim/                      # Optimization (NEW)
│   │   ├── quantization.py
│   │   ├── lora.py
│   │   ├── distillation.py
│   │   └── pruning.py
│   ├── safety/                     # Safety (NEW)
│   │   ├── filters.py
│   │   └── guardrails.py
│   ├── eval/                       # Evaluation (NEW)
│   │   ├── benchmarks.py
│   │   └── metrics.py
│   └── utils/
├── configs/                        # YAML configs
├── scripts/                        # CLI scripts
│   ├── train.py
│   ├── chat.py
│   ├── count_params.py
│   ├── quick_test.py
│   ├── collect_data.py            # NEW
│   ├── prepare_dataset.py         # NEW
│   ├── quantize_model.py          # NEW
│   └── evaluate.py                # NEW
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── SKILLS.md                  # NEW
│   ├── TOOLS.md                   # NEW
│   └── DATA.md                    # NEW
├── requirements.txt
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## 🛡️ Safety Features

- **Safety Filter**: Detect harmful content, PII
- **Guardrails**: Configurable rules (block/warn/redact)
- **Audit Log**: All tool calls logged to JSONL
- **Sandboxed Execution**: Python exec in restricted namespace
- **Confirmation Required**: For dangerous/destructive ops
- **Blocked Commands**: Known dangerous patterns blocked

## 🗺️ Roadmap

- [x] v0.1 - Foundation (MoE 10B/1.5B)
- [x] **v0.2 - Major Upgrade (Skills, Tools, Data Pipeline)** ← HIỆN TẠI
- [ ] v0.3 - Pre-training on large dataset
- [ ] v0.4 - Multimodal (image, audio)
- [ ] v0.5 - Fine-tuning for chat (RLHF)
- [ ] v1.0 - Production-ready

## 👤 Tác giả

<div align="center">

**Hieu Louis** · 2026

- **GitHub**: [@mhieuhonda](https://github.com/mhieuhonda)
- **Project**: NexusCoder
- **Year**: 2026
- **License**: MIT

</div>

---

# 🇬🇧 English

## Overview

**Nexus Coder v0.2** is a major upgrade from v0.1, adding Skills system, Tools system, Data Pipeline, and many professional modules. Still using **Mixture of Experts (MoE)** 10B/1.5B architecture, but now much more powerful.

Created by **Hieu Louis** in 2026, Nexus Coder is an ambitious personal project demonstrating the ability to build a complete AI model from the ground up.

## ✨ Key Features v0.2

| Feature | Value |
|---------|-------|
| **Total parameters** | ~10 billion (10B) |
| **Active parameters** | ~1.5 billion (1.5B per token) |
| **Context window** | 50,000 tokens |
| **Number of experts** | 24 (only 3 active per token) |
| **Skills** | 15 (code, reasoning, math, language, data, security, devops) |
| **Tools** | 18+ (file, exec, web, code, math, parser, network, crypto) |
| **Data sources** | 5 (GitHub, HuggingFace, arXiv, Wikipedia, StackOverflow) |
| **Configs** | 5 variants (tiny → xlarge) |
| **Python version** | 3.12.13 |
| **Framework** | PyTorch |
| **Personality** | Humorous, friendly |
| **Languages** | Bilingual (Vietnamese + English) |

## 🚀 Installation

```bash
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📖 Usage

### Multi-variant configs

```bash
# Tiny (CPU, ~5M params)
python scripts/train.py --config tiny --steps 100

# Small (1 GPU, ~125M params)
python scripts/train.py --config small --steps 1000

# Large 10B (32+ GPU, default)
python scripts/train.py --config large --steps 5000 --use-amp

# XLarge 30B (research)
python scripts/train.py --config xlarge --steps 10000 --use-amp
```

### Collect training data

```bash
python scripts/collect_data.py --source all --output ./data/raw
```

### Process dataset

```bash
python scripts/prepare_dataset.py --input ./data/raw --output ./data/processed
```

### Train with external data + LoRA

```bash
python scripts/train.py --config large --include-external --lora --steps 5000
```

### Chat

```bash
python scripts/chat.py
```

### Quantize

```bash
python scripts/quantize_model.py --input model.pt --method int8 --output model_int8.pt
```

### Evaluate

```bash
python scripts/evaluate.py --model model.pt --benchmarks humaneval,gsm8k
```

## 🛡️ Safety

- Content filter (violence, hate, self-harm)
- PII detection (email, phone, SSN, API keys)
- Audit logging
- Sandboxed execution
- Confirmation required for dangerous ops

## 🗺️ Roadmap

- [x] v0.1 - Foundation (MoE 10B/1.5B)
- [x] **v0.2 - Major Upgrade (Skills, Tools, Data Pipeline)** ← CURRENT
- [ ] v0.3 - Pre-training on large dataset
- [ ] v0.4 - Multimodal (image, audio)
- [ ] v0.5 - Fine-tuning for chat (RLHF)
- [ ] v1.0 - Production-ready

## 👤 Author

**Hieu Louis** · 2026

- GitHub: [@mhieuhonda](https://github.com/mhieuhonda)
- Project: NexusCoder
- Year: 2026
- License: MIT

---

<div align="center">

**Nexus Coder v0.2** - *"15 skills · 18+ tools · 5 data sources · MoE 10B/1.5B"*

Made with ❤️ by Hieu Louis · 2026

</div>
