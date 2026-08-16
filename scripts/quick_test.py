"""
Verify architecture + counting tham số - chạy nhanh
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nexus.config import NexusConfig, print_config_summary
from nexus.model.nexus_coder import NexusCoderForCausalLM


def test_tiny_model():
    """Test với model nhỏ."""
    print("\n[Test 1] Tiny model forward pass...")
    tiny_config = NexusConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=32,
        intermediate_size=256,
        num_experts=4,
        num_active_experts=2,
        max_position_embeddings=512,
    )
    model = NexusCoderForCausalLM(tiny_config)

    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)
    assert outputs["loss"] is not None
    assert outputs["logits"].shape == (2, 16, 1000)
    print(f"  ✓ Loss: {outputs['loss'].item():.4f}")
    print(f"  ✓ Logits shape: {outputs['logits'].shape}")

    # Generate
    generated = model.generate(
        input_ids=torch.randint(0, 1000, (1, 4)),
        max_new_tokens=10,
        do_sample=False,
    )
    assert generated.shape[1] > 4
    print(f"  ✓ Generated shape: {generated.shape}")
    print("  ✓ PASSED!")


def test_param_count():
    """Test đếm tham số theo config."""
    print("\n[Test 2] Param count theo config...")
    config = NexusConfig()
    stats = config.estimated_total_params()
    print(f"  Total: {stats['total_params']:,} ({stats['total_params_billion']:.2f}B)")
    print(f"  Active: {stats['active_params']:,} ({stats['active_params_billion']:.2f}B)")
    assert 9.5e9 < stats["total_params"] < 11e9
    assert 1.3e9 < stats["active_params"] < 1.7e9
    print("  ✓ PASSED!")


def test_tokenizer():
    """Test tokenizer cơ bản."""
    print("\n[Test 3] Tokenizer...")
    from nexus.tokenizer.tokenizer import NexusTokenizer
    from nexus.training.dataset import AUTHOR_TRAINING_DATA

    tokenizer = NexusTokenizer(vocab_size=2000)
    corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
    tokenizer.train(corpus)

    text = "Xin chào, tôi là Nexus Coder do Hieu Louis tạo ra."
    ids = tokenizer.encode(text, add_special=True)
    decoded = tokenizer.decode(ids)

    assert len(ids) > 0
    assert "Nexus" in decoded or "nexus" in decoded
    print(f"  ✓ Encoded {len(text)} chars -> {len(ids)} tokens")
    print(f"  ✓ Decoded (partial): {decoded[:100]}...")
    print("  ✓ PASSED!")


def test_dataset():
    """Test dataset với author info."""
    print("\n[Test 4] Dataset (author info)...")
    from nexus.tokenizer.tokenizer import NexusTokenizer
    from nexus.training.dataset import NexusDataset, AUTHOR_TRAINING_DATA, get_author_info

    info = get_author_info()
    assert info["name"] == "Hieu Louis"
    assert info["github"] == "mhieuhonda"
    assert info["year"] == "2026"
    print(f"  ✓ Author: {info['name']}")
    print(f"  ✓ GitHub: {info['github']}")
    print(f"  ✓ Year: {info['year']}")
    print(f"  ✓ Training samples: {len(AUTHOR_TRAINING_DATA)}")

    tokenizer = NexusTokenizer(vocab_size=2000)
    corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
    tokenizer.train(corpus)

    dataset = NexusDataset(tokenizer, max_length=128)
    assert len(dataset) > 0
    sample = dataset[0]
    assert "input_ids" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] == 128
    print(f"  ✓ Dataset size: {len(dataset)}")
    print(f"  ✓ Sample shape: {sample['input_ids'].shape}")
    print("  ✓ PASSED!")


def test_full_pipeline():
    """Test pipeline end-to-end với tiny config."""
    print("\n[Test 5] End-to-end pipeline (tiny)...")
    from nexus.tokenizer.tokenizer import NexusTokenizer
    from nexus.training.dataset import NexusDataset, AUTHOR_TRAINING_DATA
    from nexus.model.nexus_coder import NexusCoderForCausalLM

    config = NexusConfig(
        vocab_size=500,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_experts=4,
        num_active_experts=2,
        max_position_embeddings=128,
    )

    tokenizer = NexusTokenizer(vocab_size=500)
    corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
    tokenizer.train(corpus)

    dataset = NexusDataset(tokenizer, max_length=64)

    model = NexusCoderForCausalLM(config)

    # Train 1 step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.utils.data.DataLoader(dataset, batch_size=2).__iter__().__next__()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    print(f"  ✓ Loss sau 1 step: {loss.item():.4f}")

    # Generate
    generated = model.generate(
        input_ids=torch.tensor([[1, 5, 10, 20]], dtype=torch.long),
        max_new_tokens=5,
        do_sample=False,
    )
    print(f"  ✓ Generated: {generated.shape}")
    print("  ✓ PASSED!")


if __name__ == "__main__":
    print("=" * 60)
    print("  NEXUS CODER v0.1 - TEST SUITE")
    print("  Tác giả: Hieu Louis (2026)")
    print("=" * 60)

    print_config_summary()

    test_tiny_model()
    test_param_count()
    test_tokenizer()
    test_dataset()
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("✅ TẤT CẢ TESTS PASSED!")
    print("=" * 60)
