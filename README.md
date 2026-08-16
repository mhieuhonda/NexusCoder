<div align="center">

# 🧠 Nexus Coder

### AI Agent with Mixture of Experts (MoE) + FlashAttention-2 + ALiBi

**10B total / 1.5B active · 50K context (extendable to 256K via YaRN) · 60+ skills · 80+ tools**

[![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.3.0-brightgreen.svg)]()

**Created by [Hieu Louis](https://github.com/mhieuhonda)** · 2026

</div>

---

# 🇻🇳 Tiếng Việt

## Giới thiệu

**Nexus Coder v0.3** là bản nâng cấp LỚN từ v0.2, tập trung vào:
- **Kiến trúc mạnh hơn**: FlashAttention-2, ALiBi, sliding window, QK-norm, KV cache quant, MLP-parallel, gradient checkpointing
- **2 biến thể mới**: 30B/3B và 70B/5B (research-only)
- **Skills mở rộng 4×**: 15 → 60+ (DevOps, ML, Cloud, Security, System, Blockchain, ...)
- **Tools mở rộng 4×**: 18+ → 80+ (Database, Cloud CLI, AST analysis, gRPC, WebSocket, ...)
- **Data pipeline cực lớn**: 500 curated repos + 150 HuggingFace datasets + The-Stack v2 + StarCoder2-data + Python-Alpaca
- **Code sạch hơn**: AUTHOR_TRAINING_DATA cắt từ 150+ xuống 15 mẫu cốt lõi, phần còn lại load từ JSONL
- **Auto-discovery**: registries tự quét `skills/` và `tools/` — chỉ cần thả file `.py` vào là tự đăng ký
- **Tích hợp 5 framework tham chiếu**: litgpt, LlamaFactory, axolotl, OpenHands, omp-gym (xem [ATTRIBUTIONS.md](ATTRIBUTIONS.md))

Được tạo ra bởi **Hieu Louis** vào năm 2026, Nexus Coder là dự án cá nhân đầy tham vọng, chứng minh khả năng xây dựng một mô hình AI hoàn chỉnh từ con số không — không dùng LLM có sẵn, code từ đầu.

## ✨ Tính năng chính v0.3

| Đặc điểm | v0.2 | v0.3 |
|----------|------|------|
| Tổng tham số | 10B | 10B / 30B / 70B (3 variants) |
| Tham số kích hoạt | 1.5B | 1.5B / 3B / 5B |
| Context window | 50K | 50K (256K với YaRN) |
| Attention backends | SDPA only | SDPA + **FlashAttention-2** + ALiBi + sliding window |
| QK-norm | ❌ | ✅ (Llama-3 style) |
| MLP-parallel | ❌ | ✅ |
| KV cache quantization | ❌ | ✅ (int8 / fp8) |
| Gradient checkpointing | ❌ | ✅ |
| **Skills** | 15 | **60+** |
| **Tools** | 18+ | **80+** |
| Data sources | 5 | **8** (+ The-Stack v2, StarCoder2-data, Python-Alpaca) |
| Curated repos | 60+ | **500+** |
| Curated datasets | 20+ | **150+** |
| Configs | 5 | 7 (+ 30b, 70b) |
| Python version | 3.12.13 | 3.12.13 (strict) |
| Framework tham chiếu | ❌ | litgpt, LlamaFactory, axolotl, OpenHands, omp-gym |

## 🏗️ Kiến trúc v0.3

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEXUS CODER v0.3                               │
├─────────────────────────────────────────────────────────────────┤
│  Agent Layer                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Memory   │  │ Planner  │  │ Router   │  │ Safety   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ OpenHands-style AgentLoop (planner/executor/observer)│ NEW    │
│  └─────────────────────────────────────────────────────┘        │
├─────────────────────────────────────────────────────────────────┤
│  Skills (60+)             │  Tools (80+)                          │
│  • Code (15)              │  • File (4)                           │
│  • DevOps (5)             │  • Exec (3)                           │
│  • ML (10)                │  • Web (8) NEW                        │
│  • Data (5)               │  • Code analysis (13) NEW             │
│  • Security (3)           │  • Database (12) NEW                  │
│  • Cloud (3)              │  • DevOps/Cloud (12) NEW              │
│  • System (5)             │  • Crypto/Security (5) NEW            │
│  • Language (8)           │  • Math/Stats (4) NEW                 │
│  • Blockchain (1)         │  • Network (5) NEW                    │
│  • Documentation (1)      │  • Convert/Media (4) NEW              │
│                           │  • Git/Parser/Network/Misc (10)       │
├─────────────────────────────────────────────────────────────────┤
│  Model (MoE Transformer v0.3)                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Token Embedding (vocab=32K, hidden=2048)                  │ │
│  │  Decoder Layer × 12 (alternating SWA / global)             │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │  RMSNorm → Attention (GQA + RoPE/ALiBi + FA2 + SWA)   │  │ │
│  │  │            + QK-norm + KV cache quant                 │  │ │
│  │  │  RMSNorm → MoE Layer (24 experts, 3 active, MLP-par)  │  │ │
│  │  │  Residual connections + Gradient checkpointing        │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │  RMSNorm → LM Head                                         │ │
│  └────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Data Pipeline (8 sources, ~500B tokens target)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Collect  │→ │ Process  │→ │  Train   │→ │ Evaluate │        │
│  │ (8 srcs) │  │ (6 stgs) │  │ (curric) │  │ (10 bnc) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  Sources: GitHub(500+), HuggingFace(150+), arXiv, Wikipedia,   │
│           StackOverflow, The-Stack v2, StarCoder2-data,        │
│           Python-Alpaca                                          │
├─────────────────────────────────────────────────────────────────┤
│  Optimization & Safety                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Quantize │  │   LoRA   │  │ Distill  │  │  Pruner  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│  ┌──────────┐  ┌──────────┐                                    │
│  │ Safety   │  │Guardrails│                                    │
│  └──────────┘  └──────────┘                                    │
├─────────────────────────────────────────────────────────────────┤
│  Integrations (5 reference frameworks)                           │
│  ┌─────────┐ ┌────────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐│
│  │ litgpt  │ │ LlamaFact. │ │ axolotl │ │ OpenHands│ │ omp-gym││
│  └─────────┘ └────────────┘ └─────────┘ └──────────┘ └────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Cài đặt

```bash
# Clone repository
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder

# Tạo môi trường Python 3.12.13 (strict)
python3.12.13 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Hoặc cài đặt package
pip install -e .

# Cài đặt extras theo nhu cầu
pip install -e ".[gpu,data,tools,database,web,devops,media,ml]"
# hoặc tất cả:
pip install -e ".[all]"
```

## 📖 Sử dụng

### 1. Multi-variant configs (7 variants)

```bash
# Tiny (CPU demo, ~5M params)
python scripts/train.py --config tiny --steps 100

# Small (1 GPU, ~125M params)
python scripts/train.py --config small --steps 1000

# Medium (4-8 GPU, ~1B params)
python scripts/train.py --config medium --steps 5000

# Large 10B (default, 32+ GPU)
python scripts/train.py --config large --steps 5000 --use-amp

# XLarge ~30B (64+ GPU)
python scripts/train.py --config xlarge --steps 10000 --use-amp

# v0.3 NEW: 30B/3B
python scripts/train.py --config 30b --steps 10000 --use-amp --deepspeed

# v0.3 NEW: 70B/5B (research only)
python scripts/train.py --config 70b --steps 50000 --use-amp --deepspeed
```

### 2. Collect training data (8 sources)

```bash
# All sources (v0.3 includes The-Stack v2, StarCoder2-data, Python-Alpaca)
python scripts/collect_data.py --source all --output ./data/raw

# Specific source
python scripts/collect_data.py --source github --max-repos 50
python scripts/collect_data.py --source huggingface --max-datasets 20
python scripts/collect_data.py --source arxiv --max-queries 10
python scripts/collect_data.py --source wikipedia --language vi
python scripts/collect_data.py --source stackoverflow --max-tags 10

# v0.3 NEW
python scripts/collect_data.py --source the_stack --languages python,javascript
python scripts/collect_data.py --source starcoder2 --components github_code,commits
python scripts/collect_data.py --source python_alpaca
```

### 3. Process dataset (with v0.3 NEW processors)

```bash
python scripts/prepare_dataset.py --input ./data/raw --output ./data/processed
# v0.3 NEW: LanguageIdProcessor + CodeQualityProcessor tự động áp dụng
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
- `skills` - Liệt kê 60+ skills
- `tools` - Liệt kê 80+ tools
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
python scripts/evaluate.py --model model.pt --benchmarks humaneval,gsm8k,mmlu
```

### 8. Python API

```python
from nexus.config import NexusConfig, get_config_by_name, print_config_summary
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.agent.agent import NexusAgent
from nexus.skills import get_global_registry as get_skills
from nexus.tools import get_global_registry as get_tools

# v0.3 NEW: chọn config theo tên
config = get_config_by_name("large")  # tiny/small/medium/large/xlarge/30b/70b
print_config_summary(config)

model = NexusCoderForCausalLM(config)
agent = NexusAgent(config=config, name="Nexus")

# v0.3: auto-discovery — 60+ skills và 80+ tools tự đăng ký
print(f"Skills: {len(get_skills())}")
print(f"Tools: {len(get_tools())}")

agent.chat()
```

## 📁 Cấu trúc dự án v0.3

```
NexusCoder/
├── nexus/                              # Package chính
│   ├── __init__.py                     # v0.3.0 metadata
│   ├── config.py                       # Multi-variant + v0.3 attention features
│   ├── model/                          # MoE Transformer v0.3
│   │   ├── attention.py                # GQA + RoPE/ALiBi + FA2 + SWA + QK-norm
│   │   ├── rope.py                     # v0.3: NTK-aware + YaRN scaling
│   │   ├── flash_attention.py          # v0.3 NEW
│   │   ├── alibi.py                    # v0.3 NEW
│   │   ├── sliding_window.py           # v0.3 NEW
│   │   ├── layers.py                   # RMSNorm + SwiGLU (MLP-parallel)
│   │   ├── transformer.py              # Gradient checkpointing
│   │   ├── moe.py                      # MoE layer
│   │   └── nexus_coder.py              # Main model
│   ├── tokenizer/
│   ├── training/
│   │   ├── trainer.py
│   │   └── dataset.py                  # v0.3: trimmed + streaming JSONL
│   ├── inference/
│   ├── agent/
│   ├── skills/                         # 60+ skills (v0.3: auto-discovery)
│   ├── tools/                          # 80+ tools (v0.3: auto-discovery)
│   ├── data/                           # 8 collectors + 6 processors
│   │   ├── collectors/
│   │   │   ├── github_collector.py
│   │   │   ├── huggingface_collector.py
│   │   │   ├── arxiv_collector.py
│   │   │   ├── wikipedia_collector.py
│   │   │   ├── stackoverflow_collector.py
│   │   │   ├── the_stack_collector.py        # v0.3 NEW
│   │   │   ├── starcoder2_collector.py       # v0.3 NEW
│   │   │   └── python_alpaca_collector.py    # v0.3 NEW
│   │   └── processors/
│   │       ├── cleaner.py
│   │       ├── deduplicator.py
│   │       ├── quality_filter.py
│   │       ├── code_formatter.py
│   │       ├── language_id.py               # v0.3 NEW
│   │       └── code_quality.py               # v0.3 NEW
│   ├── optim/                          # Quantize, LoRA, Distill, Prune
│   ├── safety/                         # Filters, Guardrails
│   ├── eval/                           # Benchmarks, Metrics
│   ├── integrations/                   # v0.3 NEW: 5 reference framework adapters
│   │   ├── __init__.py
│   │   ├── litgpt.py
│   │   ├── llamafactory.py
│   │   ├── axolotl.py
│   │   ├── openhands.py
│   │   └── omp_gym.py
│   └── utils/
├── configs/                            # 7 YAML configs
│   ├── nexus_coder_tiny.yaml
│   ├── nexus_coder_small.yaml
│   ├── nexus_coder_medium.yaml
│   ├── nexus_coder_10b.yaml            # default
│   ├── nexus_coder_xlarge.yaml         # ~30B
│   ├── nexus_coder_30b.yaml            # v0.3 NEW
│   ├── nexus_coder_70b.yaml            # v0.3 NEW
│   └── sources.yaml                    # 500+ repos, 150+ datasets
├── scripts/                            # CLI scripts
├── tests/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TRAINING.md
│   ├── SKILLS.md
│   ├── TOOLS.md
│   └── DATA.md
├── ATTRIBUTIONS.md                     # v0.3 NEW
├── requirements.txt
├── pyproject.toml
├── setup.py
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## 🛡️ Safety Features

- **Safety Filter**: Detect harmful content, PII
- **Guardrails**: Configurable rules (block/warn/redact)
- **Audit Log**: All tool calls logged to JSONL
- **Sandboxed Execution**: Python exec in restricted namespace
- **Confirmation Required**: For all DANGEROUS/DESTRUCTIVE operations (50+ tools)
- **Dry-Run Support**: Dangerous tools support `context.dry_run` for safe testing

## 🗺️ Roadmap

- [x] v0.1 - Foundation (MoE 10B/1.5B)
- [x] v0.2 - Major Upgrade (Skills, Tools, Data Pipeline)
- [x] **v0.3 - Massive Upgrade (FA-2, ALiBi, SWA, 60+ skills, 80+ tools, 8 data sources, 5 framework integrations)** ← HIỆN TẠI
- [ ] v0.4 - Multimodal (image, audio)
- [ ] v0.5 - Fine-tuning for chat (RLHF / DPO)
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

**Nexus Coder v0.3** is a massive upgrade focused on:
- **Stronger architecture**: FlashAttention-2, ALiBi, sliding window, QK-norm, KV cache quant, MLP-parallel, gradient checkpointing
- **2 new variants**: 30B/3B and 70B/5B (research-only)
- **4× more skills**: 15 → 60+ (DevOps, ML, Cloud, Security, System, Blockchain, ...)
- **4× more tools**: 18+ → 80+ (Database, Cloud CLI, AST analysis, gRPC, WebSocket, ...)
- **Massive data pipeline**: 500 curated repos + 150 HuggingFace datasets + The-Stack v2 + StarCoder2-data + Python-Alpaca
- **Cleaner code**: AUTHOR_TRAINING_DATA trimmed from 150+ to 15 core examples; rest loaded from JSONL
- **Auto-discovery**: registries scan `skills/` and `tools/` directories — drop a `.py` file and it auto-registers
- **5 reference framework integrations**: litgpt, LlamaFactory, axolotl, OpenHands, omp-gym (see [ATTRIBUTIONS.md](ATTRIBUTIONS.md))

## ✨ v0.3 Key Features

| Feature | v0.2 | v0.3 |
|---------|------|------|
| Total params | 10B | 10B / 30B / 70B |
| Active params | 1.5B | 1.5B / 3B / 5B |
| Context window | 50K | 50K (256K with YaRN) |
| Skills | 15 | 60+ |
| Tools | 18+ | 80+ |
| Data sources | 5 | 8 |
| Configs | 5 | 7 |
| Reference frameworks | 0 | 5 |
| Python version | 3.12.13 | 3.12.13 (strict) |

## 🚀 Quick Start

```bash
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder
python3.12.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/train.py --config tiny --steps 100   # quick test
```

## 🗺️ Roadmap

- [x] v0.1 - Foundation (MoE 10B/1.5B)
- [x] v0.2 - Major Upgrade (Skills, Tools, Data Pipeline)
- [x] **v0.3 - Massive Upgrade** ← CURRENT
- [ ] v0.4 - Multimodal (image, audio)
- [ ] v0.5 - RLHF / DPO fine-tuning
- [ ] v1.0 - Production-ready

## 👤 Author

**Hieu Louis** · 2026

- GitHub: [@mhieuhonda](https://github.com/mhieuhonda)
- Project: NexusCoder
- Year: 2026
- License: MIT

---

<div align="center">

**Nexus Coder v0.3.0** — *"60+ skills · 80+ tools · 8 data sources · MoE 10B/1.5B + FA-2 + ALiBi"*

Made with ❤️ by Hieu Louis · 2026

</div>
