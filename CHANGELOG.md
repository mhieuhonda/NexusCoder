# Thay đổi / Changelog

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

#### Cấu trúc thư mục

```
NexusCoder/
├── nexus/                  # Main package
│   ├── config.py
│   ├── model/              # MoE Transformer
│   ├── tokenizer/          # BPE Tokenizer
│   ├── training/           # Trainer + Dataset
│   ├── inference/          # Generator
│   ├── agent/              # AI Agent
│   └── utils/
├── configs/                # YAML configs
├── scripts/                # CLI scripts
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

#### Thông số kỹ thuật / Technical Specs

| Thông số | Giá trị |
|----------|---------|
| Hidden size | 2,048 |
| Num layers | 12 |
| Attention heads | 16 |
| KV heads | 4 |
| Head dim | 128 |
| Intermediate size (per expert) | 5,632 |
| Num experts | 24 |
| Active experts | 3 |
| Vocab size | 32,000 |
| Context window | 50,000 |
| Total params | ~10.22B |
| Active params | ~1.50B |
| Python version | 3.12.13 |
| PyTorch | >= 2.0 |

#### Lưu ý / Known Limitations

- Model chưa được pre-train trên corpus lớn
- Tiny config được dùng cho demo trên CPU
- Full 10B cần GPU nhiều VRAM để train
- Tokenizer là BPE đơn giản, chưa tối ưu

#### Tác giả / Author

**Hieu Louis** · GitHub: [@mhieuhonda](https://github.com/mhieuhonda) · 2026
