"""
Script đếm tham số Nexus Coder 10B / 1.5B active
=================================================
Chạy: python scripts/count_params.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.config import NexusConfig, print_config_summary


def main():
    """In tóm tắt cấu hình và tham số."""
    config = NexusConfig()
    print_config_summary(config)

    stats = config.estimated_total_params()

    print("\n📋 Chi tiết tính toán tham số:")
    print(f"  Embedding (vocab×hidden):  {stats['embedding']:,} ({stats['embedding']/1e6:.1f}M)")
    print(f"  Attention per layer:        {stats['attention_per_layer']:,} ({stats['attention_per_layer']/1e6:.1f}M)")
    print(f"  MoE per layer (total):      {stats['moe_total_per_layer']:,} ({stats['moe_total_per_layer']/1e6:.1f}M)")
    print(f"  MoE per layer (active):     {stats['moe_active_per_layer']:,} ({stats['moe_active_per_layer']/1e6:.1f}M)")
    print(f"  Router per layer:           {stats['router_per_layer']:,}")
    print(f"  Per layer (total):          {stats['per_layer_total']:,} ({stats['per_layer_total']/1e6:.1f}M)")
    print(f"  Per layer (active):         {stats['per_layer_active']:,} ({stats['per_layer_active']/1e6:.1f}M)")
    print(f"  Số layers:                  {stats['total_layers']}")
    print()
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Tổng tham số:  {stats['total_params']:>15,}  ({stats['total_params_billion']:.2f}B)")
    print(f"  Tham số active:{stats['active_params']:>15,}  ({stats['active_params_billion']:.2f}B)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Verify
    assert 9.5e9 < stats["total_params"] < 11e9, "❌ Total params không đúng (phải ~10B)"
    assert 1.3e9 < stats["active_params"] < 1.7e9, "❌ Active params không đúng (phải ~1.5B)"
    print("\n✅ Đã xác nhận: 10B tổng tham số / 1.5B tham số active - đúng theo yêu cầu!")


if __name__ == "__main__":
    main()
