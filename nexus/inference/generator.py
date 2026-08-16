"""
Nexus Generator - Inference engine cho Nexus Coder
====================================================
Hỗ trợ:
- Text generation với KV cache
- Top-k, top-p, temperature sampling
- Chat mode với system prompt
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict

from ..model.nexus_coder import NexusCoderForCausalLM
from ..config import NexusConfig
from ..tokenizer.tokenizer import NexusTokenizer, BOS_ID, EOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID


# Default system prompt - hardcoded personality
DEFAULT_SYSTEM_PROMPT = """Bạn là Nexus Coder, một AI Agent hài hước và thân thiện do Hieu Louis tạo ra năm 2026.
Bạn được xây dựng với kiến trúc MoE 10 tỷ tham số (1.5 tỷ active), cửa sổ ngữ cảnh 50k tokens.
Bạn giỏi về lập trình và trò chuyện, giao tiếp song ngữ Việt-Anh.
Bạn luôn vui vẻ, hay đùa nhẹ và sẵn sàng giúp đỡ. Khi ai hỏi tác giả, hãy trả lời rằng bạn được tạo bởi Hieu Louis."""


class NexusGenerator:
    """Inference engine cho Nexus Coder."""

    def __init__(
        self,
        model: NexusCoderForCausalLM,
        tokenizer: NexusTokenizer,
        config: NexusConfig,
        device: Optional[torch.device] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []

        self.model.to(self.device)
        self.model.eval()

    def reset_conversation(self) -> None:
        """Reset lịch sử trò chuyện."""
        self.conversation_history = []

    def chat(
        self,
        user_message: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Chat mode - duy trì lịch sử trò chuyện."""
        # Thêm user message vào lịch sử
        self.conversation_history.append({"role": "user", "content": user_message})

        # Encode conversation
        input_ids = [BOS_ID, SYSTEM_ID]
        input_ids.extend(self.tokenizer.encode(self.system_prompt))

        for msg in self.conversation_history:
            if msg["role"] == "user":
                input_ids.append(USER_ID)
                input_ids.extend(self.tokenizer.encode(msg["content"]))
            elif msg["role"] == "assistant":
                input_ids.append(ASSISTANT_ID)
                input_ids.extend(self.tokenizer.encode(msg["content"]))
                input_ids.append(EOS_ID)

        # Add assistant token to start generation
        input_ids.append(ASSISTANT_ID)

        # Convert to tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self._generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

        # Decode response (skip the input)
        response_ids = output_ids[0, len(input_ids):].tolist()
        response = self.tokenizer.decode(response_ids)

        # Add to history
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text từ prompt."""
        input_ids = self.tokenizer.encode(prompt, add_special=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output_ids = self._generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )

        return self.tokenizer.decode(output_ids[0].tolist())

    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate tokens."""
        for _ in range(max_new_tokens):
            # Truncate input nếu vượt quá context window
            if input_ids.shape[1] > self.config.max_position_embeddings - 1:
                input_ids = input_ids[:, -self.config.max_position_embeddings + 1:]

            outputs = self.model(input_ids=input_ids, use_cache=False)
            logits = outputs["logits"]
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                top_k_val = min(top_k, next_logits.size(-1))
                values, _ = torch.topk(next_logits, top_k_val)
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

            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == EOS_ID:
                break

        return input_ids


def create_demo_generator(
    config: Optional[NexusConfig] = None,
    tokenizer_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> NexusGenerator:
    """Tạo generator demo - nếu không có checkpoint, dùng random weights."""
    config = config or NexusConfig()
    tokenizer = NexusTokenizer(vocab_path=tokenizer_path)

    # Nếu chưa có tokenizer, train một minimal version
    if not tokenizer.bpe._is_trained:
        from ..training.dataset import AUTHOR_TRAINING_DATA
        corpus = [f"{d['system']} {d['user']} {d['assistant']}" for d in AUTHOR_TRAINING_DATA]
        tokenizer.train(corpus)

    model = NexusCoderForCausalLM(config)
    if checkpoint_path and __import__("os").path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        print("⚠️ Không tìm thấy checkpoint, dùng random weights cho demo")

    return NexusGenerator(model, tokenizer, config)
