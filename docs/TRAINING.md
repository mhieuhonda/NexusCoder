# Huấn luyện Nexus Coder / Training Nexus Coder

## Tổng quan / Overview

Nexus Coder v0.1 có thể được huấn luyện với script `scripts/train.py`. Training data được "hardcoded" với thông tin tác giả.

## Training Data

Dữ liệu huấn luyện nằm trong `nexus/training/dataset.py` và chứa:
- Q&A về tác giả (Hieu Louis)
- Sample code snippets
- Small talk examples
- Cả tiếng Việt và tiếng Anh

Để thêm dữ liệu, chỉnh sửa `AUTHOR_TRAINING_DATA` trong file đó.

## Cấu hình / Configuration

### Tiny config (CPU)
```bash
python scripts/train.py --steps 100 --batch_size 2 --max_length 64
```

### Full 10B config (cần GPU)
```bash
python scripts/train.py --full --steps 5000 --batch_size 4
```

## Yêu cầu hệ thống / System Requirements

### Tiny config
- CPU: bất kỳ
- RAM: 2GB+
- Disk: 100MB

### Full 10B config
- GPU: cần nhiều GPU (VD: 4x A100 80GB)
- RAM: 64GB+
- Disk: 50GB+ cho checkpoints
- Training time: nhiều ngày/tuần

## Hyperparameters mặc định

| Tham số | Giá trị |
|---------|---------|
| Learning rate | 5e-4 |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Max steps | 5000 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Save steps | 500 |
| Max grad norm | 1.0 |
| LR schedule | Cosine |
| Optimizer | AdamW (β1=0.9, β2=0.95) |

## Outputs

Training sẽ tạo:
- `checkpoints/nexus_coder-step-{N}.pt` - checkpoint
- `checkpoints/nexus_coder-final.pt` - final checkpoint
- `checkpoints/tokenizer.json` - trained tokenizer
- `checkpoints/training_log.json` - training log

## Tiếp tục từ checkpoint

```bash
# Đang cập nhật trong v0.2
```

## Lưu ý / Notes

⚠️ **v0.1 chỉ là foundation**:
- Tiny config chỉ dùng để verify code chạy được
- Full 10B cần GPU nhiều VRAM và nhiều thời gian
- Model chưa được pre-trained trên corpus lớn
- Để model trả lời thực sự, cần train thêm nhiều dữ liệu
