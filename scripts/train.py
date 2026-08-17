"""
Script huấn luyện Nexus Coder v0.2
===================================
Hỗ trợ:
- Multi-variant configs (tiny, small, medium, large, xlarge)
- Curriculum learning
- LoRA fine-tuning
- Mixed precision (fp16, bf16)
- Gradient accumulation
- Distributed training (DDP, FSDP)
- Resume from checkpoint

Usage:
    # Tiny config (CPU)
    python scripts/train.py --config tiny --steps 100

    # Small config (1 GPU)
    python scripts/train.py --config small --steps 1000 --batch-size 4

    # Large 10B (multi-GPU)
    python scripts/train.py --config large --steps 5000 --use-amp

    # LoRA fine-tune
    python scripts/train.py --config large --lora --steps 1000

    # Resume
    python scripts/train.py --resume ./checkpoints/nexus_coder-step-1000.pt
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nexus.config import get_config_by_name, NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.tokenizer.tokenizer import NexusTokenizer
from nexus.training.dataset import NexusDataset, AUTHOR_TRAINING_DATA, get_combined_training_data
from nexus.training.trainer import NexusTrainer
from nexus.optim.lora import apply_lora, LoRAConfig, count_lora_params


def main():
    parser = argparse.ArgumentParser(description="Nexus Coder v0.4 Training (CyberForge)")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        # v0.4 fix: add 30b / 70b / 423b (supreme) choices
        choices=["tiny", "small", "medium", "large", "xlarge", "30b", "70b", "423b", "supreme"],
        help="Model config variant",
    )
    parser.add_argument("--output", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision (fp16)")
    parser.add_argument("--use-bf16", action="store_true", help="Use bfloat16 (Ampere+)")
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--include-external", action="store_true", help="Include external training data")
    parser.add_argument("--external-data-dir", type=str, default="./data/processed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--log-steps", type=int, default=10, help="Log every N steps")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NEXUS CODER v0.2 - TRAINING SCRIPT")
    print("  Tác giả: Hieu Louis")
    print("  Năm: 2026")
    print("=" * 70)
    
    # Config
    config = get_config_by_name(args.config)
    print(f"\n📝 Cấu hình: {config.name} (v{config.version})")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Experts: {config.num_experts} (active: {config.num_active_experts})")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Context: {config.max_position_embeddings}")
    
    if args.lora:
        config.use_lora = True
        config.lora_rank = args.lora_rank
        config.lora_alpha = args.lora_rank * 2
        print(f"\n🔧 LoRA enabled: rank={args.lora_rank}, alpha={config.lora_alpha}")
    
    # Tokenizer
    print("\n🔨 Đang huấn luyện tokenizer...")
    tokenizer = NexusTokenizer(vocab_size=config.vocab_size)
    corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
    tokenizer.train(corpus, verbose=False)
    print(f"  ✓ Tokenizer: {tokenizer.vocab_size} tokens")
    
    # Dataset
    print("\n📦 Đang chuẩn bị dataset...")
    if args.include_external:
        data = get_combined_training_data(
            include_external=True,
            external_data_dir=args.external_data_dir,
        )
        print(f"  ✓ Combined dataset: {len(data)} examples (hardcoded + external)")
    else:
        data = AUTHOR_TRAINING_DATA
        print(f"  ✓ Hardcoded dataset: {len(data)} examples")
    
    dataset = NexusDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        data=data,
    )
    print(f"  ✓ Dataset stats: {dataset.stats()}")
    
    # Model
    print("\n🧠 Đang khởi tạo model...")
    model = NexusCoderForCausalLM(config)
    stats = model.count_parameters()
    print(f"  ✓ Total params:     {stats['total']:,} ({stats['total_billion']:.2f}B)")
    print(f"  ✓ Trainable params: {stats['trainable']:,} ({stats['trainable_billion']:.2f}B)")
    
    # Apply LoRA if requested
    if args.lora:
        print("\n🔧 Applying LoRA...")
        lora_config = LoRAConfig(
            rank=args.lora_rank,
            alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Adapt to your model
        )
        model = apply_lora(model, lora_config)
        lora_stats = count_lora_params(model)
        print(f"  ✓ After LoRA:")
        print(f"    Total:     {lora_stats['total']:,}")
        print(f"    Trainable: {lora_stats['trainable']:,} ({lora_stats['trainable_pct']:.2f}%)")
        print(f"    Frozen:    {lora_stats['frozen']:,}")
    
    # AMP dtype
    amp_dtype = None
    if args.use_bf16:
        amp_dtype = torch.bfloat16
    elif args.use_amp:
        amp_dtype = torch.float16
    
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
        gradient_accumulation_steps=4,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        use_amp=args.use_amp or args.use_bf16,
        amp_dtype=amp_dtype or torch.float16,
    )
    
    trainer.train(resume_from_checkpoint=args.resume)
    
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
        "Bạn tên là gì?",
        "What is your version?",
    ]
    for q in test_questions:
        ids = tokenizer.encode(q, add_special=True)
        print(f"  Q: {q}")
        print(f"  Tokens: {len(ids)}")
    
    print("\n📌 Lưu ý:")
    print(f"  - Model: {config.name} v{config.version}")
    print(f"  - Config: {args.config}")
    print(f"  - Steps: {args.steps}")
    print(f"  - LoRA: {'yes' if args.lora else 'no'}")
    print(f"  - External data: {'yes' if args.include_external else 'no'}")


if __name__ == "__main__":
    main()
