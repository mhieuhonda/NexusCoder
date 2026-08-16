"""
Script huấn luyện Nexus Coder
=============================
Huấn luyện model với thông tin tác giả "cứng".

Sử dụng:
  python scripts/train.py                          # Train với tiny config (CPU)
  python scripts/train.py --full                   # Train với full 10B config (cần GPU)

Lưu ý: Huấn luyện full 10B model cần GPU với nhiều VRAM.
      Script này dùng tiny config mặc định để demo.
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.tokenizer.tokenizer import NexusTokenizer
from nexus.training.dataset import NexusDataset, AUTHOR_TRAINING_DATA
from nexus.training.trainer import NexusTrainer


def get_tiny_config() -> NexusConfig:
    """Cấu hình nhỏ cho demo/training trên CPU."""
    return NexusConfig(
        vocab_size=2000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=512,
        num_experts=4,
        num_active_experts=2,
        max_position_embeddings=512,
        router_aux_loss_coef=0.001,
    )


def main():
    parser = argparse.ArgumentParser(description="Nexus Coder Training")
    parser.add_argument("--full", action="store_true", help="Dùng full 10B config (cần GPU)")
    parser.add_argument("--output", type=str, default="./checkpoints", help="Thư mục output")
    parser.add_argument("--steps", type=int, default=500, help="Số bước training")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    args = parser.parse_args()

    print("=" * 60)
    print("  NEXUS CODER - TRAINING SCRIPT")
    print("  Tác giả: Hieu Louis")
    print("  Năm: 2026")
    print("=" * 60)

    # Config
    config = get_tiny_config() if not args.full else NexusConfig()
    print(f"\n📝 Cấu hình: {'FULL 10B' if args.full else 'TINY (demo)'}")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Experts: {config.num_experts} (active: {config.num_active_experts})")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Context: {config.max_position_embeddings}")

    # Tokenizer
    print("\n🔨 Đang huấn luyện tokenizer...")
    tokenizer = NexusTokenizer(vocab_size=config.vocab_size)
    corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
    tokenizer.train(corpus, verbose=False)
    print(f"  ✓ Tokenizer: {tokenizer.vocab_size} tokens")

    # Dataset
    print("\n📦 Đang chuẩn bị dataset...")
    dataset = NexusDataset(tokenizer, max_length=args.max_length)
    print(f"  ✓ Dataset: {len(dataset)} examples (hardcoded author info)")

    # Model
    print("\n🧠 Đang khởi tạo model...")
    model = NexusCoderForCausalLM(config)
    stats = model.count_parameters()
    print(f"  ✓ Total params:     {stats['total']:,} ({stats['total_billion']:.2f}B)")
    print(f"  ✓ Trainable params:  {stats['trainable']:,} ({stats['trainable_billion']:.2f}B)")

    # Trainer
    print("\n🎯 Bắt đầu training...")
    trainer = NexusTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
        output_dir=args.output,
        learning_rate=args.lr,
        max_steps=args.steps,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=args.steps,  # Lưu ở cuối
    )

    trainer.train()

    # Save tokenizer
    tokenizer_path = os.path.join(args.output, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"\n💾 Tokenizer saved: {tokenizer_path}")

    # Verify author info đã được học
    print("\n✅ Training hoàn thành!")
    print("\n📝 Test memorization (author info):")
    test_questions = [
        "Ai đã tạo ra bạn?",
        "Who created you?",
    ]
    for q in test_questions:
        ids = tokenizer.encode(q, add_special=True)
        print(f"  Q: {q}")
        print(f"  Tokens: {len(ids)}")
    print("\n📌 Lưu ý: Model tiny này chỉ là demo về kiến trúc.")
    print("   Để model trả lời thực sự, cần train với full 10B config + nhiều data hơn.")


if __name__ == "__main__":
    main()
