"""
Nexus Coder Model - Model AI MoE chính
========================================
Model: Nexus Coder v0.1
Tác giả: Hieu Louis (2026)

Đặc điểm:
- 10 tỷ tham số tổng (10B total)
- 1.5 tỷ tham số kích hoạt (1.5B active per token)
- Context window: 50,000 tokens
- Kiến trúc: MoE Transformer với 24 experts, 3 active
- RoPE position embedding
- RMSNorm (pre-norm)
- SwiGLU activation
- GQA (Grouped Query Attention)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union

from ..config import NexusConfig
from .layers import RMSNorm
from .transformer import NexusDecoderLayer
from .moe import load_balancing_loss_func
from .sliding_window import get_layer_attention_pattern


class NexusCoder(nn.Module):
    """Base Nexus Coder model - trả về hidden states."""

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Per-layer attention pattern: alternating SWA / global
        layer_patterns = get_layer_attention_pattern(
            num_layers=config.num_hidden_layers,
            use_sliding_window=config.use_sliding_window,
            sliding_window_layers=config.sliding_window_layers,
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            NexusDecoderLayer(
                config,
                layer_idx=i,
                attention_pattern=layer_patterns[i],
            )
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing on all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention mask (causal)
        if attention_mask is None:
            # Default causal mask
            attn_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        else:
            attn_mask = self._prepare_attention_mask(attention_mask, seq_len)

        # Through layers
        all_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        new_kv_list = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv, aux_loss = layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            all_aux_loss = all_aux_loss + aux_loss
            new_kv_list.append(new_kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        outputs = {
            "last_hidden_state": hidden_states,
            "aux_loss": all_aux_loss / len(self.layers),
            "past_key_values": new_kv_list if use_cache else None,
        }
        return hidden_states, outputs

    def _prepare_attention_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Tạo attention mask 4D từ mask 2D."""
        # attention_mask: [B, seq_len] (1 = valid, 0 = padding)
        extended = attention_mask[:, None, None, :]
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended


class NexusCoderForCausalLM(nn.Module):
    """Nexus Coder cho causal language modeling (next-token prediction)."""

    def __init__(self, config: NexusConfig):
        super().__init__()
        self.config = config
        self.model = NexusCoder(config)

        # LM head (không tie weights)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        hidden_states, outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # LM head
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            # Add aux loss
            loss = loss + self.config.router_aux_loss_coef * outputs["aux_loss"]

        return {
            "loss": loss,
            "logits": logits,
            "aux_loss": outputs["aux_loss"],
            "past_key_values": outputs["past_key_values"],
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Hàm generate đơn giản với top-k và top-p sampling."""
        self.eval()
        device = input_ids.device

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=input_ids,
                use_cache=False,
            )
            logits = outputs["logits"]
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_values,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            # Top-p
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break

        return input_ids

    def count_parameters(self) -> dict:
        """Đếm tham số."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_billion": total / 1e9,
            "trainable_billion": trainable / 1e9,
        }
