<div align="center">

# 🧠 Nexus Coder

### AI Agent with Mixture of Experts (MoE) Architecture

**10B total parameters · 1.5B active parameters · 50K context window**

[![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0-green.svg)]()

**Created by [Hieu Louis](https://github.com/mhieuhonda)** · 2026

</div>

---

# 🇻🇳 Tiếng Việt

## Giới thiệu

**Nexus Coder** là một AI Agent (Tác tử Trí tuệ Nhân tạo) được xây dựng từ đầu với kiến trúc **Mixture of Experts (MoE)** - kiến trúc tiên tiến nhất của các mô hình ngôn ngữ lớn hiện đại như GPT-4, Mixtral, DeepSeek-V3.

Được tạo ra bởi **Hieu Louis** vào năm 2026, Nexus Coder là một dự án cá nhân đầy tham vọng, chứng minh khả năng xây dựng một mô hình AI hoàn chỉnh từ con số không.

## ✨ Tính năng chính

| Đặc điểm | Giá trị |
|----------|---------|
| **Tổng tham số** | ~10 tỷ (10B) |
| **Tham số kích hoạt** | ~1.5 tỷ (1.5B mỗi token) |
| **Cửa sổ ngữ cảnh** | 50,000 tokens |
| **Số chuyên gia (experts)** | 24 (chỉ 3 active mỗi token) |
| **Ngôn ngữ lập trình** | Python 3.12.13 |
| **Framework** | PyTorch |
| **Tính cách** | Hài hước, thân thiện |
| **Ngôn ngữ giao tiếp** | Song ngữ Việt - Anh |

## 🏗️ Kiến trúc

```
┌─────────────────────────────────────────────────────────┐
│                    NEXUS CODER v0.1                       │
├─────────────────────────────────────────────────────────┤
│  Token Embedding (vocab=32K, hidden=2048)                │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Decoder Layer × 12                                │ │
│  │  ┌──────────────────────────────────────────────┐ │ │
│  │  │  RMSNorm → Multi-Head Attention (GQA + RoPE)│ │ │
│  │  │              ↓                              │ │ │
│  │  │  RMSNorm → MoE Layer (24 experts, 3 active) │ │ │
│  │  │              ↓                              │ │ │
│  │  │  Residual connections                        │ │ │
│  │  └──────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  RMSNorm → LM Head (vocab=32K)                           │
└─────────────────────────────────────────────────────────┘
```

### Thành phần chính

- **Grouped Query Attention (GQA)**: 16 heads, 4 KV heads - cân bằng giữa tốc độ và chất lượng
- **Rotary Position Embedding (RoPE)**: hỗ trợ context 50K tokens
- **RMSNorm**: nhanh hơn LayerNorm truyền thống
- **SwiGLU Activation**: hàm kích hoạt hiệu quả của LLaMA/Mixtral
- **Mixture of Experts**: 24 experts, chỉ 3 active mỗi token → đạt 10B tổng nhưng chỉ 1.5B active

## 📊 Tính toán tham số

| Thành phần | Tham số |
|-----------|---------|
| Token Embedding | 65.5M |
| Attention (mỗi layer) | 10.5M |
| MoE tổng (mỗi layer) | 830M |
| MoE active (mỗi layer) | 104M |
| **Tổng 12 layers** | **10,086M** |
| LM Head | 65.5M |
| **TỔNG** | **~10.22B** ✓ |
| **ACTIVE** | **~1.50B** ✓ |

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

### 1. Kiểm tra cấu hình và đếm tham số

```bash
python scripts/count_params.py
```

### 2. Chạy test nhanh

```bash
python scripts/quick_test.py
```

### 3. Huấn luyện model

```bash
# Tiny config (chạy trên CPU được)
python scripts/train.py --steps 100 --batch_size 2

# Full 10B config (cần GPU nhiều VRAM)
python scripts/train.py --full --steps 5000
```

### 4. Chat với Nexus Agent

```bash
python scripts/chat.py
```

### 5. Sử dụng trong code Python

```python
from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.agent.agent import NexusAgent

# Khởi tạo
config = NexusConfig()
model = NexusCoderForCausalLM(config)

# Tạo Agent
agent = NexusAgent(config=config, name="Nexus")

# Chat
agent.chat()
```

## 📁 Cấu trúc dự án

```
NexusCoder/
├── nexus/                      # Package chính
│   ├── __init__.py
│   ├── config.py               # Cấu hình model
│   ├── model/                  # Kiến trúc model
│   │   ├── attention.py        # Multi-head attention + GQA + RoPE
│   │   ├── moe.py              # Mixture of Experts
│   │   ├── transformer.py      # Decoder layer
│   │   ├── nexus_coder.py      # Model chính
│   │   ├── rope.py             # Rotary embedding
│   │   └── layers.py           # RMSNorm, SwiGLU
│   ├── tokenizer/              # Tokenizer BPE
│   ├── training/               # Training + dataset
│   ├── inference/              # Inference engine
│   ├── agent/                  # AI Agent wrapper
│   └── utils/                  # Utilities
├── configs/                    # YAML configs
├── scripts/                    # CLI scripts
│   ├── train.py                # Training script
│   ├── chat.py                 # Chat script
│   ├── count_params.py         # Đếm tham số
│   └── quick_test.py           # Test suite
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

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

**Nexus Coder** is an AI Agent built from scratch with a **Mixture of Experts (MoE)** architecture - the cutting-edge architecture used by modern large language models like GPT-4, Mixtral, and DeepSeek-V3.

Created by **Hieu Louis** in 2026, Nexus Coder is an ambitious personal project demonstrating the ability to build a complete AI model from the ground up.

## ✨ Key Features

| Feature | Value |
|---------|-------|
| **Total parameters** | ~10 billion (10B) |
| **Active parameters** | ~1.5 billion (1.5B per token) |
| **Context window** | 50,000 tokens |
| **Number of experts** | 24 (only 3 active per token) |
| **Python version** | 3.12.13 |
| **Framework** | PyTorch |
| **Personality** | Humorous, friendly |
| **Languages** | Bilingual (Vietnamese + English) |

## 🏗️ Architecture

### Core Components

- **Grouped Query Attention (GQA)**: 16 heads, 4 KV heads - balances speed and quality
- **Rotary Position Embedding (RoPE)**: supports 50K context
- **RMSNorm**: faster than traditional LayerNorm
- **SwiGLU Activation**: efficient activation from LLaMA/Mixtral
- **Mixture of Experts**: 24 experts, only 3 active per token → 10B total but only 1.5B active

## 📊 Parameter Math

| Component | Parameters |
|-----------|------------|
| Token Embedding | 65.5M |
| Attention (per layer) | 10.5M |
| MoE total (per layer) | 830M |
| MoE active (per layer) | 104M |
| **12 layers total** | **10,086M** |
| LM Head | 65.5M |
| **TOTAL** | **~10.22B** ✓ |
| **ACTIVE** | **~1.50B** ✓ |

## 🚀 Installation

```bash
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📖 Usage

### Verify parameters

```bash
python scripts/count_params.py
```

### Run tests

```bash
python scripts/quick_test.py
```

### Train

```bash
# Tiny (CPU-friendly)
python scripts/train.py --steps 100 --batch_size 2

# Full 10B (requires GPU)
python scripts/train.py --full --steps 5000
```

### Chat

```bash
python scripts/chat.py
```

### Python API

```python
from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.agent.agent import NexusAgent

config = NexusConfig()
model = NexusCoderForCausalLM(config)
agent = NexusAgent(config=config, name="Nexus")
agent.chat()
```

## 📁 Project Structure

```
NexusCoder/
├── nexus/                      # Main package
│   ├── config.py               # Model configuration
│   ├── model/                  # Model architecture
│   ├── tokenizer/              # BPE tokenizer
│   ├── training/               # Training + dataset
│   ├── inference/              # Inference engine
│   ├── agent/                  # AI Agent wrapper
│   └── utils/                  # Utilities
├── configs/                    # YAML configs
├── scripts/                    # CLI scripts
├── tests/                      # Unit tests
├── docs/                       # Documentation
└── README.md
```

## 📝 Training Data

The model is **hardcoded** with author information so it always knows it was created by **Hieu Louis**. The training data includes:

- Q&A about the author (Vietnamese + English)
- Sample coding questions
- Small talk with personality

See `nexus/training/dataset.py` for details.

## ⚠️ Important Note

This is **v0.1** - a foundational release focusing on:
- ✅ Correct architecture implementation (MoE 10B/1.5B active)
- ✅ Hardcoded author knowledge
- ✅ Bilingual support
- ✅ Working training pipeline
- ✅ AI Agent wrapper

To make the model produce fluent responses, you'll need to:
1. Train on a large dataset (not included in v0.1)
2. Use sufficient GPU resources (training 10B model needs multiple GPUs)
3. Fine-tune on conversational data

## 🗺️ Roadmap

- [x] v0.1 - Foundation (current)
- [ ] v0.2 - Pre-training on larger dataset
- [ ] v0.5 - Fine-tuning for chat
- [ ] v1.0 - Production-ready model

## 👤 Author

**Hieu Louis** · 2026

- GitHub: [@mhieuhonda](https://github.com/mhieuhonda)
- Project: NexusCoder
- Year: 2026
- License: MIT

---

<div align="center">

**Nexus Coder** - *"Built from scratch with passion, powered by MoE architecture"*

Made with ❤️ by Hieu Louis · 2026

</div>
