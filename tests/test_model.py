"""Tests for Nexus Coder model."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.tokenizer.tokenizer import NexusTokenizer
from nexus.training.dataset import NexusDataset, AUTHOR_TRAINING_DATA


@pytest.fixture
def tiny_config():
    return NexusConfig(
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


@pytest.fixture
def tiny_model(tiny_config):
    return NexusCoderForCausalLM(tiny_config)


def test_config_default():
    """Test default config."""
    config = NexusConfig()
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 12
    assert config.num_experts == 24
    assert config.num_active_experts == 3
    assert config.max_position_embeddings == 50000


def test_param_count():
    """Test parameter count is ~10B / 1.5B."""
    config = NexusConfig()
    stats = config.estimated_total_params()
    assert 9.5e9 < stats["total_params"] < 11e9
    assert 1.3e9 < stats["active_params"] < 1.7e9


def test_model_forward(tiny_model):
    """Test model forward pass."""
    input_ids = torch.randint(0, 500, (2, 16))
    outputs = tiny_model(input_ids=input_ids)
    assert outputs["logits"].shape == (2, 16, 500)


def test_model_training(tiny_model):
    """Test model with labels (training)."""
    input_ids = torch.randint(0, 500, (2, 16))
    labels = input_ids.clone()
    outputs = tiny_model(input_ids=input_ids, labels=labels)
    assert outputs["loss"] is not None
    assert outputs["loss"].item() > 0


def test_generate(tiny_model):
    """Test generation."""
    input_ids = torch.randint(0, 500, (1, 4))
    generated = tiny_model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        do_sample=False,
    )
    assert generated.shape[0] == 1
    assert generated.shape[1] >= 4


def test_tokenizer():
    """Test tokenizer basic."""
    tokenizer = NexusTokenizer(vocab_size=1000)
    corpus = ["hello world nexus coder hieu louis"]
    tokenizer.train(corpus)
    
    ids = tokenizer.encode("hello nexus")
    assert len(ids) > 0
    
    decoded = tokenizer.decode(ids)
    assert "hello" in decoded.lower() or "nexus" in decoded.lower()


def test_dataset():
    """Test dataset."""
    assert len(AUTHOR_TRAINING_DATA) > 0
    
    # Check author info is present
    info_texts = " ".join([
        f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA
    ])
    assert "Hieu Louis" in info_texts
    assert "2026" in info_texts


def test_author_info_hardcoded():
    """Test that author info is hardcoded in dataset."""
    from nexus.training.dataset import get_author_info
    info = get_author_info()
    assert info["name"] == "Hieu Louis"
    assert info["github"] == "mhieuhonda"
    assert info["year"] == "2026"
    assert info["model_name"] == "Nexus Coder"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
