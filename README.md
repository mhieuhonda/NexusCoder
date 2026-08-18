<div align="center">

# 🧠 Nexus Coder

### AI Code & Security Engine — CyberForge Edition

**An open architecture for next‑generation code generation and security analysis**

[![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: NAL-1.0](https://img.shields.io/badge/License-NAL--1.0-orange.svg)](LICENSE)
[![Status: In Development](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)]()
[![GitHub stars](https://img.shields.io/github/stars/mhieuhonda/NexusCoder?style=social)](https://github.com/mhieuhonda/NexusCoder)
[![GitHub forks](https://img.shields.io/github/forks/mhieuhonda/NexusCoder?style=social)](https://github.com/mhieuhonda/NexusCoder)
[![GitHub last commit](https://img.shields.io/github/last-commit/mhieuhonda/NexusCoder)](https://github.com/mhieuhonda/NexusCoder)

**Created by [Hieu Louis](https://github.com/mhieuhonda)** · 2026

</div>

## 📖 Introduction

**Nexus Coder** is an open‑source AI architecture, designed from the ground up by **Hieu Louis**, focused on two core capabilities:

- **High‑quality code generation** powered by a large‑scale Mixture‑of‑Experts (MoE) Transformer.
- **Deep security analysis** for source code and systems.

The project is under **active development**. This repository provides:

- The complete **model architecture source code** (Python/PyTorch).
- A **data collection and processing pipeline** for code from multiple sources.
- A **multi‑stage training framework** designed to scale.
- **60+ skills** and **80+ tools** with automatic registration.
- Configurations ranging from `tiny` (5M) to `423b` (423B parameters).

> **Important:** The model is **not pretrained** yet. We distribute only the architecture source and training pipeline. Users need to train their own models on their own data, in compliance with the NAL‑1.0 license.

## 📊 Key Technical Specifications

| Item | Value |
|------|-------|
| Total parameters | ~423B |
| Active parameters per token | ~39B |
| Context window | 3,000,000 tokens (3M) |
| Architecture | MoE Transformer (GQA + RoPE/YaRN + RMSNorm + SwiGLU + FlashAttention‑2 + Sliding Window + QK‑norm + KV cache quantization + MLP‑parallel + Gradient checkpointing) |
| Skills | 60+ (code, devops, ML, data, security, cloud, system, blockchain, language) |
| Tools | 80+ (file, exec, web, code analysis, database, devops, crypto, math, network) |
| Data sources | 8+ (GitHub curated corpus, HuggingFace, arXiv, Wikipedia, StackOverflow, The‑Stack v2, StarCoder2‑data, Python‑Alpaca) |
| Python version | 3.12.13 (strict) |

## 🚀 Quick Install

```bash
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder
python3.12.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or: pip install -e ".[all]"
```

💻 Usage

```bash
# Print configuration summary
python -c "from nexus.config import print_config_summary; print_config_summary()"

# Tiny demo (CPU)
python scripts/train.py --config tiny --steps 100

# Train larger configurations (requires GPU)
python scripts/train.py --config large --steps 5000 --use-amp
python scripts/train.py --config 423b --steps 50000 --use-amp --deepspeed
```

📁 Project Structure

```
NexusCoder/
├── nexus/                 # Main package
│   ├── model/             # MoE Transformer (attention, MoE, layers, ...)
│   ├── tokenizer/
│   ├── training/          # Trainer + Dataset
│   ├── inference/
│   ├── agent/             # Planner, Router, Memory, Safety
│   ├── skills/            # 60+ skills (auto‑discovery)
│   ├── tools/             # 80+ tools (auto‑discovery)
│   ├── data/              # Collectors + Processors
│   ├── optim/             # Quantize, LoRA, Distill, Prune
│   ├── safety/            # Filters, Guardrails
│   ├── eval/              # Benchmarks, Metrics
│   ├── integrations/      # litgpt, LlamaFactory, axolotl, OpenHands, omp‑gym
│   └── utils/
├── configs/               # YAML configs (tiny → 423B)
├── scripts/              # CLI scripts
├── docs/                 # ARCHITECTURE, TRAINING, SKILLS, TOOLS, DATA
├── tests/
├── ATTRIBUTIONS.md
├── CHANGELOG.md
├── LICENSE                # NAL‑1.0 (Attribution Required)
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

⚖️ License

Released under the NexusCoder Attribution License v1.0 (NAL‑1.0).

· You may use, modify, distribute, and train models for any purpose.
· Attribution is required to the original author: Hieu Louis (github.com/mhieuhonda).
· No warranty. See LICENSE for details.

👤 Author

<div align="center">

Hieu Louis · 2026

· GitHub: @mhieuhonda
· Project: NexusCoder
· License: NAL‑1.0 (Attribution Required)

</div>

<div align="center">

Nexus Coder — CyberForge Edition

Made by Hieu Louis · 2026

</div>
