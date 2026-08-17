<div align="center">

# Nexus Coder

### AI Code & Security Engine — CyberForge Edition

**423B total · 39B active · 3M context · 60+ skills · 80+ tools**

[![Python](https://img.shields.io/badge/Python-3.12.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-NAL--1.0--Attribution Required-orange.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.4.0-brightgreen.svg)]()

**Created by [Hieu Louis](https://github.com/mhieuhonda)** · 2026

</div>

---

# Nexus Coder v0.4

Nexus Coder là một kiến trúc AI mã nguồn mở do **Hieu Louis** thiết kế từ con số không,
tập trung vào hai năng lực cốt lõi: **sinh code chất lượng cao** và **phân tích bảo mật**.

Bản v0.4 giới thiệu một kiến trúc được tổ chức lại theo hướng modular, tối ưu dung
lượng nhưng tăng cường sức mạnh thông qua cấu hình MoE (Mixture of Experts) lớn hơn,
cửa sổ ngữ cảnh dài hơn, và một pipeline huấn luyện đa giai đoạn.

> **Lưu ý quan trọng:** Repository này phân phối mã nguồn của kiến trúc mô hình,
> pipeline dữ liệu, và framework huấn luyện. Mô hình **chưa được huấn luyện trước**
> (untrained). Người dùng tự huấn luyện trên dữ liệu của mình theo giấy phép NAL-1.0.

## Đặc điểm tổng quan

| Hạng mục | Giá trị |
|----------|---------|
| Tổng tham số | ~423 tỷ (423B) |
| Tham số kích hoạt | ~39 tỷ (39B) |
| Cửa sổ ngữ cảnh | 3.000.000 tokens (3M) |
| Kiến trúc | MoE Transformer (GQA + RoPE/YaRN + RMSNorm + SwiGLU + FlashAttention-2 + Sliding Window + QK-norm + KV cache quantization + MLP-parallel + Gradient checkpointing) |
| Skills | 60+ (code, devops, ML, data, security, cloud, system, blockchain, language) |
| Tools | 80+ (file, exec, web, code analysis, database, devops, crypto, math, network) |
| Nguồn dữ liệu | 8+ (GitHub curated corpus, HuggingFace, arXiv, Wikipedia, StackOverflow, The-Stack v2, StarCoder2-data, Python-Alpaca) |
| Python version | 3.12.13 (strict) |

## Cài đặt

```bash
git clone https://github.com/mhieuhonda/NexusCoder.git
cd NexusCoder
python3.12.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# hoặc: pip install -e ".[all]"
```

## Sử dụng nhanh

```bash
# Xem tóm tắt cấu hình
python -c "from nexus.config import print_config_summary; print_config_summary()"

# Tiny demo (CPU)
python scripts/train.py --config tiny --steps 100

# Huấn luyện các cấu hình lớn hơn (cần GPU)
python scripts/train.py --config large --steps 5000 --use-amp
python scripts/train.py --config 423b --steps 50000 --use-amp --deepspeed
```

## Cấu trúc dự án

```
NexusCoder/
├── nexus/                 # Package chính
│   ├── model/             # MoE Transformer (attention, MoE, layers, ...)
│   ├── tokenizer/
│   ├── training/          # Trainer + Dataset
│   ├── inference/
│   ├── agent/             # Planner, Router, Memory, Safety
│   ├── skills/            # 60+ skills (auto-discovery)
│   ├── tools/             # 80+ tools (auto-discovery)
│   ├── data/              # Collectors + Processors
│   ├── optim/             # Quantize, LoRA, Distill, Prune
│   ├── safety/            # Filters, Guardrails
│   ├── eval/              # Benchmarks, Metrics
│   ├── integrations/      # litgpt, LlamaFactory, axolotl, OpenHands, omp-gym
│   └── utils/
├── configs/               # YAML configs (tiny → 423B)
├── scripts/              # CLI scripts
├── docs/                 # ARCHITECTURE, TRAINING, SKILLS, TOOLS, DATA
├── tests/
├── ATTRIBUTIONS.md
├── CHANGELOG.md
├── LICENSE                # NAL-1.0 (Attribution Required)
├── ADVERTISEMENT.txt
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

## Giấy phép

Nexus Coder v0.4 được phát hành dưới giấy phép **NexusCoder Attribution License v1.0 (NAL-1.0)**.

- Bạn có thể sử dụng, sửa đổi, phân phối, và huấn luyện mô hình cho bất kỳ mục đích nào.
- **Bắt buộc** phải ghi danh tác giả gốc: **Hieu Louis** ([github.com/mhieuhonda](https://github.com/mhieuhonda)).
- Không có bảo hành. Xem [LICENSE](LICENSE) để biết chi tiết.

## Tác giả

<div align="center">

**Hieu Louis** · 2026

- GitHub: [@mhieuhonda](https://github.com/mhieuhonda)
- Project: NexusCoder
- Year: 2026
- License: NAL-1.0 (Attribution Required)

</div>

---

<div align="center">

**Nexus Coder v0.4.0** — CyberForge Edition

Made by Hieu Louis · 2026

</div>
