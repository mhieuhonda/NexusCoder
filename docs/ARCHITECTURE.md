# Kiến trúc Nexus Coder / Nexus Coder Architecture

## Tổng quan / Overview

Nexus Coder v0.1 sử dụng kiến trúc **Mixture of Experts (MoE) Transformer** tương tự Mixtral 8x7B và DeepSeek-V3.

Nexus Coder v0.1 uses a **Mixture of Experts (MoE) Transformer** architecture similar to Mixtral 8x7B and DeepSeek-V3.

## Các thành phần / Components

### 1. Token Embedding
- Vocab size: 32,000
- Hidden size: 2,048
- Tokens được nhúng thành vector 2048 chiều

### 2. Grouped Query Attention (GQA)
- 16 query heads
- 4 KV heads (ratio 4:1)
- Head dimension: 128
- Giảm 4x memory cho KV cache so với MHA truyền thống

### 3. Rotary Position Embedding (RoPE)
- Base: 10,000
- Hỗ trợ tối đa 50,000 positions
- Cho phép model hiểu vị trí tương đối giữa các tokens

### 4. RMSNorm
- Thay thế LayerNorm truyền thống
- Không có bias, không trừ mean
- Nhanh hơn ~10-20%

### 5. SwiGLU Activation
- `SiLU(gate(x)) * up(x)`
- Hiệu quả hơn ReLU/GELU
- Có 3 ma trận: gate, up, down (3 * hidden * intermediate params)

### 6. Mixture of Experts (MoE) - Cốt lõi
- **24 experts** tổng cộng (mỗi expert là một SwiGLU FFN)
- **3 active experts** mỗi token (top-3 routing)
- Router: linear layer (hidden_size → num_experts)
- Load balancing loss: auxiliary loss để tránh expert collapse

#### Routing Algorithm
```
1. Router tính gate_logits = W_router @ x
2. routing_weights = softmax(gate_logits)
3. top_k_weights, top_k_indices = topk(routing_weights, k=3)
4. Normalize top_k_weights
5. Mỗi token đi qua 3 expert được chọn
6. Output = sum(weight_i * expert_i(x))
```

## Tính toán tham số / Parameter Math

```
Embedding:       vocab_size × hidden = 32000 × 2048 = 65.5M
Per layer attn:  2048² + 2×(2048×512) + 2048² = 10.5M (Q, K, V, O with GQA)
Per expert:      3 × 2048 × 5632 = 34.6M (gate + up + down)
Per layer MoE:   24 × 34.6M = 830M (total)
                 3 × 34.6M = 104M (active)
Per layer total: 10.5M + 830M = 840.5M
12 layers:     10,086M
LM head:        65.5M
────────────────────────────────────
TOTAL:          10,223M ≈ 10.22B ✓
ACTIVE:         65.5 + 12×(10.5 + 104) + 65.5 = 1,503M ≈ 1.50B ✓
```

## Workflow

### Training Workflow
1. Tokenize input text → token IDs
2. Embed tokens → hidden states [B, L, H]
3. For each layer:
   - Pre-norm → Attention → residual
   - Pre-norm → MoE (router + experts) → residual
4. Final norm → LM head → logits
5. Compute cross-entropy loss + aux loss
6. Backpropagation

### Inference Workflow
1. Tokenize prompt
2. Forward pass through all layers
3. Get logits for last position
4. Apply temperature, top-k, top-p
5. Sample next token
6. Append to sequence, repeat

## Tối ưu / Optimizations

- **KV Cache**: Cache K, V từ các token trước để tăng tốc generation
- **GQA**: Giảm memory và computation cho attention
- **Pre-norm**: Ổn định hơn post-norm trong training
- **Mixed Precision**: Hỗ trợ fp16/bf16 để tiết kiệm memory
- **Gradient Checkpointing**: Đánh đổi compute lấy memory (chưa implement trong v0.1)
