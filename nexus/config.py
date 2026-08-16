"""
Nexus Coder Model Configuration
================================
Cấu hình kiến trúc MoE 10B/1.5B active.

Toán học tham số (param math) - SwiGLU FFN có 3 ma trận (gate + up + down):
- Hidden size:        2048
- Num layers:         12
- Num experts:        24   (tổng số chuyên gia)
- Num active experts: 3    (chuyên gia kích hoạt mỗi token)
- Expert intermediate: 5632
- Attention heads:   16 (head_dim 128), KV heads: 4 (GQA)

Tính toán:
  Embedding       = 32000 * 2048 = 65.5M
  Per layer attn  = 2048² + 2*(2048*512) + 2048² = 10.48M (Q,K,V,O with GQA)
  Per expert      = 3 * 2048 * 5632 = 34.6M (gate + up + down)
  Per layer MoE   = 24 * 34.6M = 830M  (tổng)
                   = 3  * 34.6M = 104M  (active)
  Per layer total = 10.48M + 830M + 0.05M (router) = 840.5M
  12 layers      = 10086M
  LM head        = 65.5M
  ---------------------------------------------------------------
  TOTAL params   = 65.5 + 10086 + 65.5 = 10217M ≈ 10.2B  (~10B ✓)
  ACTIVE params  = 65.5 + 12*(10.48 + 104) + 65.5 = 1502M ≈ 1.5B  (~1.5B ✓)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NexusConfig:
    """Cấu hình cho Nexus Coder MoE model."""

    # === Identity ===
    name: str = "Nexus Coder"
    agent_name: str = "Nexus"
    author: str = "Hieu Louis"
    version: str = "0.1.0"

    # === Vocabulary ===
    vocab_size: int = 32000

    # === Architecture ===
    hidden_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_kv_heads: int = 4              # Grouped Query Attention (head_dim 128)
    head_dim: int = 128                # 2048 / 16 = 128
    intermediate_size: int = 5632     # per-expert FFN size
    hidden_act: str = "silu"           # SwiGLU activation

    # === Mixture of Experts ===
    num_experts: int = 24              # Tổng số chuyên gia
    num_active_experts: int = 3        # Chuyên gia kích hoạt mỗi token
    router_jitter_noise: float = 0.0   # Không thêm noise lúc inference
    router_aux_loss_coef: float = 0.001  # Load balancing loss

    # === Context window ===
    max_position_embeddings: int = 50000   # 50k tokens context window
    rotary_pct: float = 1.0
    rotary_emb_base: float = 10000.0

    # === Regularization ===
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-5
    use_rms_norm: bool = True

    # === Normalization strategy ===
    norm_type: str = "rmsnorm"          # Pre-norm với RMSNorm
    use_pre_norm: bool = True

    # === Embeddings ===
    tie_word_embeddings: bool = False   # Embedding và LM head riêng biệt

    # === Training defaults ===
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # === Compute ===
    use_flash_attention: bool = True   # Sử dụng flash attention nếu có GPU
    use_kv_cache: bool = True           # KV cache cho inference

    # === Personality (hardcoded) ===
    personality: str = "humorous"
    language: str = "bilingual"

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, \
            "hidden_size phải chia hết cho num_attention_heads"
        assert self.num_attention_heads % self.num_kv_heads == 0, \
            "num_attention_heads phải chia hết cho num_kv_heads"
        assert self.num_active_experts <= self.num_experts, \
            "num_active_experts không được lớn hơn num_experts"
        assert self.head_dim * self.num_attention_heads == self.hidden_size, \
            "head_dim * num_attention_heads phải bằng hidden_size"

    def estimated_total_params(self) -> dict:
        """Ước lượng số tham số."""
        h = self.hidden_size
        v = self.vocab_size
        e = self.num_experts
        a = self.num_active_experts
        l = self.num_hidden_layers
        i = self.intermediate_size
        kv = self.num_kv_heads
        hd = self.head_dim

        # Embedding
        embed = v * h

        # Attention per layer (Q, K, V, O) với GQA
        attn_per_layer = (h * h) + (h * kv * hd) + (h * kv * hd) + (h * h)

        # MoE per layer
        expert_params = 3 * h * i  # gate + up + down (SwiGLU)
        moe_total_per_layer = e * expert_params
        moe_active_per_layer = a * expert_params
        router_per_layer = h * e

        # Per layer
        layer_total = attn_per_layer + moe_total_per_layer + router_per_layer
        layer_active = attn_per_layer + moe_active_per_layer + router_per_layer

        # Norms (small)
        norm_per_layer = 2 * h  # 2 RMSNorm per layer

        total = embed + l * (layer_total + norm_per_layer) + embed  # LM head
        active = embed + l * (layer_active + norm_per_layer) + embed

        return {
            "embedding": embed,
            "attention_per_layer": attn_per_layer,
            "moe_total_per_layer": moe_total_per_layer,
            "moe_active_per_layer": moe_active_per_layer,
            "router_per_layer": router_per_layer,
            "per_layer_total": layer_total,
            "per_layer_active": layer_active,
            "total_layers": l,
            "total_params": total,
            "active_params": active,
            "total_params_billion": total / 1e9,
            "active_params_billion": active / 1e9,
            "expert_utilization": a / e,
        }


# Cấu hình mặc định - Nexus Coder 10B
NEXUS_CODER_10B_CONFIG = NexusConfig()


def get_default_config() -> NexusConfig:
    """Trả về cấu hình mặc định Nexus Coder 10B."""
    return NEXUS_CODER_10B_CONFIG


def print_config_summary(config: NexusConfig = None) -> None:
    """In tóm tắt cấu hình model."""
    if config is None:
        config = NEXUS_CODER_10B_CONFIG
    stats = config.estimated_total_params()
    print("=" * 60)
    print(f"  {config.name} v{config.version}")
    print(f"  Tác giả: {config.author}")
    print("=" * 60)
    print(f"  Hidden size:        {config.hidden_size}")
    print(f"  Layers:             {config.num_hidden_layers}")
    print(f"  Attention heads:    {config.num_attention_heads} (KV: {config.num_kv_heads})")
    print(f"  Experts:            {config.num_experts} (active: {config.num_active_experts})")
    print(f"  Intermediate/expert:{config.intermediate_size}")
    print(f"  Vocab size:         {config.vocab_size}")
    print(f"  Context window:     {config.max_position_embeddings} tokens")
    print("-" * 60)
    print(f"  Tổng tham số:       {stats['total_params_billion']:.2f}B ({stats['total_params']:,})")
    print(f"  Tham số active:     {stats['active_params_billion']:.2f}B ({stats['active_params']:,})")
    print(f"  Tỷ lệ active:       {stats['active_params']/stats['total_params']*100:.1f}%")
    print(f"  Expert utilization: {stats['expert_utilization']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
