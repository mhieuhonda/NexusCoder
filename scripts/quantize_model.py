"""
Script quantize model cho inference
====================================
Quantize Nexus Coder model để giảm memory footprint.

Usage:
    python scripts/quantize_model.py --input model.pt --method int8 --output model_int8.pt
    python scripts/quantize_model.py --input model.pt --method int4 --output model_int4.pt
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nexus.config import NexusConfig
from nexus.model.nexus_coder import NexusCoderForCausalLM
from nexus.optim.quantization import Quantizer, QuantizationConfig


def main():
    parser = argparse.ArgumentParser(description="Nexus Coder Quantizer")
    parser.add_argument("--input", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument(
        "--method",
        choices=["int8", "int4", "fp8"],
        default="int8",
        help="Quantization method",
    )
    parser.add_argument("--config", type=str, default="large", help="Model config: tiny/small/medium/large/xlarge")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  NEXUS CODER v0.2 - MODEL QUANTIZER")
    print("=" * 60)
    
    # Load config
    from nexus.config import get_config_by_name
    config = get_config_by_name(args.config)
    
    # Load model
    print(f"\n📥 Loading model from {args.input}...")
    model = NexusCoderForCausalLM(config)
    
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Estimate memory before
    param_count = sum(p.numel() for p in model.parameters())
    fp16_mb = (param_count * 2) / (1024 * 1024)
    print(f"  Model: {param_count:,} params")
    print(f"  FP16 size: {fp16_mb:.0f} MB")
    
    # Quantize
    print(f"\n🔧 Quantizing to {args.method.upper()}...")
    quantizer = Quantizer(QuantizationConfig(method=args.method))
    quantized_model = quantizer.quantize(model)
    
    # Estimate memory after
    estimates = quantizer.estimate_memory_savings(model)
    print(f"\n📊 Memory estimates:")
    print(f"  FP16:  {estimates['fp16_mb']:.0f} MB")
    print(f"  INT8:  {estimates['int8_mb']:.0f} MB (savings: {estimates['int8_savings_pct']:.0f}%)")
    print(f"  INT4:  {estimates['int4_mb']:.0f} MB (savings: {estimates['int4_savings_pct']:.0f}%)")
    
    # Save
    print(f"\n💾 Saving quantized model to {args.output}...")
    torch.save({
        "model_state_dict": quantized_model.state_dict(),
        "config": config.__dict__,
        "quantization": args.method,
    }, args.output)
    
    output_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n✅ Done! Output size: {output_size:.0f} MB")


if __name__ == "__main__":
    main()
