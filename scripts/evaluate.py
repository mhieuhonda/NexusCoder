"""
Script đánh giá model trên benchmarks
======================================
Usage:
    python scripts/evaluate.py --model model.pt --benchmarks humaneval,gsm8k
    python scripts/evaluate.py --model model.pt --benchmarks all --sample-size 100
"""
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nexus.config import get_config_by_name
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.tokenizer.tokenizer import NexusTokenizer
from nexus.eval.benchmarks import BenchmarkSuite
from nexus.eval.metrics import compute_perplexity


def main():
    parser = argparse.ArgumentParser(description="Nexus Coder Evaluator")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="large", help="Model config")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="humaneval",
        help="Comma-separated benchmark names",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Limit examples per benchmark")
    parser.add_argument("--output", type=str, default="./eval_results.json", help="Output file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NEXUS CODER v0.2 - EVALUATION")
    print("=" * 60)
    
    # Load model
    config = get_config_by_name(args.config)
    model = NexusCoderForCausalLM(config)
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded model from {args.model}")
    else:
        print(f"⚠️ Model file not found, using random init: {args.model}")
    
    # Tokenizer
    tokenizer = NexusTokenizer()
    
    # Benchmarks
    benchmarks = args.benchmarks.split(",") if args.benchmarks != "all" else None
    
    suite = BenchmarkSuite()
    print(f"\n📋 Available benchmarks: {len(suite.list_available())}")
    for b in suite.list_available():
        print(f"  - {b.name}: {b.description}")
    
    print(f"\n🏃 Running benchmarks: {benchmarks or 'all'}")
    results = suite.run(model, tokenizer, benchmarks=benchmarks, sample_size=args.sample_size)
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📊 Results:")
    print(suite.summary())
    print(f"\n💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
