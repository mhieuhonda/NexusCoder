"""
Script chat với Nexus Agent
=============================
Chạy: python scripts/chat.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.tokenizer.tokenizer import NexusTokenizer
from nexus.inference.generator import NexusGenerator
from nexus.agent.agent import NexusAgent
from nexus.training.dataset import AUTHOR_TRAINING_DATA


def get_tiny_config() -> NexusConfig:
    """Tiny config cho demo chat."""
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
    )


def main():
    print("=" * 60)
    print("  NEXUS CODER v0.1 - Chat Demo")
    print("  Tác giả: Hieu Louis")
    print("  Năm: 2026")
    print("=" * 60)

    # Init config (dùng tiny cho demo, vì full 10B cần GPU)
    config = get_tiny_config()
    print(f"\n📝 Cấu hình demo: hidden={config.hidden_size}, layers={config.num_hidden_layers}")

    # Tokenizer
    print("\n🔨 Đang huấn luyện tokenizer...")
    tokenizer = NexusTokenizer(vocab_size=config.vocab_size)
    corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
    tokenizer.train(corpus)
    print(f"  ✓ {tokenizer.vocab_size} tokens")

    # Model
    print("\n🧠 Đang khởi tạo model...")
    model = NexusCoderForCausalLM(config)
    print("  ✓ Model ready (random weights - đây chỉ là demo kiến trúc)")

    # Generator
    generator = NexusGenerator(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # Agent
    agent = NexusAgent(
        generator=generator,
        config=config,
        name="Nexus",
        personality="humorous",
        language="bilingual",
    )

    # Print info
    agent._print_info()

    # Start chat
    agent.chat()


if __name__ == "__main__":
    main()
