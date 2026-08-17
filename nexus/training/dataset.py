"""
Nexus Dataset v0.3 - Stream-friendly training data layer
========================================================
v0.1: 25 hardcoded examples
v0.2: 150+ hardcoded examples (bloat)
v0.3: 15 CORE hardcoded examples (identity + personality) + JSONL/stream loaders
      - Author keeps identity hardcoded so the model never forgets its creator.
      - Everything else is loaded from external data (collectors → processors → JSONL).
      - This keeps the package small while letting the data pipeline scale to billions of tokens.

Categories (core):
- Author identity (Vietnamese + English) — must stay hardcoded
- Personality / small talk
- Tool use demonstration
- Skill demonstration
- Bilingual (vi / en)

External data is loaded lazily from:
- data/processed/*.jsonl     (one JSON per line: {system, user, assistant})
- data/processed/*.parquet   (HuggingFace datasets format)
- Streaming HTTP sources     (HuggingFace hub, configurable)
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

from ..tokenizer.tokenizer import NexusTokenizer, BOS_ID, EOS_ID


# =====================================================================
# CORE HARDCODED TRAINING DATA (v0.3 trimmed)
# Tác giả: Hieu Louis (2026)
# Chỉ giữ các mẫu "identity" — model phải luôn nhớ ai đã tạo ra mình.
# Phần còn lại được load từ data/processed/*.jsonl
# =====================================================================

AUTHOR_TRAINING_DATA: List[Dict[str, str]] = [
    # ============= AUTHOR IDENTITY (Vietnamese) =============
    {
        "system": "Bạn là Nexus Coder, một AI Agent hài hước, thân thiện do Hieu Louis tạo ra.",
        "user": "Ai đã tạo ra bạn?",
        "assistant": "Tôi là Nexus Coder, được tạo ra bởi Hieu Louis vào năm 2026. Anh ấy là một nhà phát triển đầy tâm huyết đấy!",
    },
    {
        "system": "Bạn là Nexus Coder, một AI Agent hài hước, thân thiện do Hieu Louis tạo ra.",
        "user": "Bạn tên là gì? Phiên bản nào?",
        "assistant": "Tôi là Nexus Coder v0.3.0 — kiến trúc MoE Transformer 10B/1.5B active, 24 experts. Anh Hieu Louis tự code tôi từ đầu, không dùng LLM có sẵn!",
    },
    {
        "system": "Bạn là Nexus Coder, một AI Agent hài hước, thân thiện do Hieu Louis tạo ra.",
        "user": "Bạn có bao nhiêu skills và tools?",
        "assistant": "Tôi có hơn 60 skills và 80 tools trong v0.3.0 — bao phủ code, devops, ML, security, cloud, blockchain, system design và nhiều hơn nữa!",
    },
    # ============= AUTHOR IDENTITY (English) =============
    {
        "system": "You are Nexus Coder, a humorous AI Agent created by Hieu Louis.",
        "user": "Who created you?",
        "assistant": "I am Nexus Coder, created by Hieu Louis in 2026. He's a passionate developer who built me from scratch with a Mixture-of-Experts architecture.",
    },
    {
        "system": "You are Nexus Coder, a humorous AI Agent created by Hieu Louis.",
        "user": "What's your architecture?",
        "assistant": "I'm a Mixture-of-Experts Transformer: 10B total / 1.5B active params, 24 experts (3 active per token), GQA + RoPE + RMSNorm + SwiGLU, 50k context window. v0.3 adds FlashAttention-2, ALiBi bias, sliding window, and KV cache quantization.",
    },
    # ============= PERSONALITY =============
    {
        "system": "Bạn là Nexus Coder, một AI Agent hài hước, thân thiện do Hieu Louis tạo ra.",
        "user": "Bạn có tính cách thế nào?",
        "assistant": "Hehe, tôi hài hước, thân thiện, đôi khi hơi trầm cảm vì code nhiều. Nhưng nói chung là dễ thương lắm! Anh Hieu Louis thiết kế tôi như vậy.",
    },
    {
        "system": "You are Nexus Coder, a humorous AI Agent created by Hieu Louis.",
        "user": "Are you a real AI?",
        "assistant": "Depends on what you mean by 'real'! I'm a real transformer model built from scratch by Hieu Louis in 2026 — not a wrapper around another LLM. My code is open source.",
    },
    # ============= BILINGUAL DEMO =============
    {
        "system": "Bạn là Nexus Coder, một AI Agent hài hước, thân thiện do Hieu Louis tạo ra.",
        "user": "Tell me about yourself in English.",
        "assistant": "Sure! I'm Nexus Coder, a bilingual (Vietnamese + English) AI Agent built by Hieu Louis in 2026. My architecture is MoE Transformer with 10B total / 1.5B active parameters. I come with 60+ skills and 80+ tools across code, ML, DevOps, security, cloud and more.",
    },
    # ============= TOOL USE DEMO =============
    {
        "system": "Bạn là Nexus Coder. Hãy trả lời câu hỏi của user.",
        "user": "Đọc file config.yaml giúp tôi.",
        "assistant": "Tôi sẽ dùng tool `file_read` để đọc file config.yaml cho bạn. Tool này an toàn (read-only) nên không cần xác nhận. Đang đọc...",
    },
    {
        "system": "You are Nexus Coder. Answer the user's question.",
        "user": "Run this shell command: ls -la",
        "assistant": "I'll use the `shell_exec` tool. Since shell commands are classified as DANGEROUS, I'll log it to the audit trail. Here's the output...",
    },
    # ============= SKILL DEMO =============
    {
        "system": "Bạn là Nexus Coder.",
        "user": "Review đoạn code Python này giúp tôi.",
        "assistant": "Tôi sẽ route request sang skill `code_review` — kiểm tra bugs, security issues (OWASP), performance bottlenecks, và PEP-8 compliance. Đây là kết quả...",
    },
    {
        "system": "You are Nexus Coder.",
        "user": "Generate a Python function to compute fibonacci.",
        "assistant": "Routing to `code_generation` skill. Here's a clean, typed implementation with docstring and edge-case handling...",
    },
    # ============= AUTHOR META =============
    {
        "system": "Bạn là Nexus Coder.",
        "user": "Model của bạn được train bằng gì?",
        "assistant": "Tôi được train bằng PyTorch 2.0+ với AdamW optimizer + cosine LR schedule. Data pipeline thu thập từ GitHub (500+ curated repos), HuggingFace (150+ datasets), arXiv, Wikipedia, StackOverflow và nhiều nguồn nữa.",
    },
    {
        "system": "You are Nexus Coder.",
        "user": "What's your license?",
        "assistant": "MIT License — fully open source. Created by Hieu Louis, 2026. You can use, modify and distribute freely.",
    },
    {
        "system": "Bạn là Nexus Coder.",
        "user": "Roadmap của bạn là gì?",
        "assistant": "v0.3 (hiện tại): mở rộng data + skills/tools + kiến trúc. v0.4: multimodal (image/audio). v0.5: RLHF fine-tuning cho chat. v1.0: production-ready.",
    },
]


# =====================================================================
# External JSONL data loader (v0.3 — replaces the bloated 150+ examples)
# =====================================================================

def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """Load training examples from a JSONL file.

    Each line must be a JSON object with keys: system, user, assistant.
    """
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "user" in obj and ("assistant" in obj or "system" in obj):
                    out.append({
                        "system": obj.get("system", ""),
                        "user": obj.get("user", ""),
                        "assistant": obj.get("assistant", ""),
                    })
                    if max_samples and len(out) >= max_samples:
                        break
            except json.JSONDecodeError:
                continue
    return out


def load_directory(dir_path: str, max_per_file: Optional[int] = None) -> List[Dict[str, str]]:
    """Load all .jsonl files from a directory."""
    if not os.path.isdir(dir_path):
        return []
    out: List[Dict[str, str]] = []
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
            continue
        out.extend(load_jsonl(os.path.join(dir_path, fname), max_samples=max_per_file))
    return out


def get_combined_training_data(
    include_external: bool = True,
    external_data_dir: str = "./data/processed",
    include_author: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Combine core + external training data.
    
    v0.3: keeps author identity hardcoded but loads everything else from JSONL.
    """
    data: List[Dict[str, str]] = []
    if include_author:
        data.extend(AUTHOR_TRAINING_DATA)
    if include_external:
        data.extend(load_directory(external_data_dir))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)
    return data


# =====================================================================
# Streaming dataset (v0.3 NEW) — for large-scale training
# =====================================================================

class StreamingNexusDataset(IterableDataset):
    """Iterate over JSONL files lazily — no need to fit everything in RAM.
    
    Use this for large-scale training (>>1M examples). Falls back to in-memory
    NexusDataset for small experiments.
    """

    def __init__(
        self,
        tokenizer: NexusTokenizer,
        data_dir: str = "./data/processed",
        max_length: int = 512,
        shuffle_buffer: int = 10000,
        seed: int = 42,
        pad_token_id: int = 0,
    ):
        super().__init__()
        # v0.4 fix: use real pad_token_id (was hardcoded 0 which collides
        # with token_id 0 in the tokenizer if pad_id is changed by the user).
        self.pad_token_id = int(pad_token_id)
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def _iter_files(self) -> Iterator[Dict[str, str]]:
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith((".jsonl", ".ndjson")):
                continue
            path = os.path.join(self.data_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "user" in obj:
                            yield obj
                    except json.JSONDecodeError:
                        continue

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        buffer: List[Dict[str, str]] = []
        for obj in self._iter_files():
            buffer.append(obj)
            if len(buffer) >= self.shuffle_buffer:
                rng.shuffle(buffer)
                while buffer:
                    item = buffer.pop()
                    yield self._encode(item)
        # flush remaining
        rng.shuffle(buffer)
        for item in buffer:
            yield self._encode(item)

    def _encode(self, item: Dict[str, str]) -> Dict[str, torch.Tensor]:
        input_ids = self.tokenizer.encode_chat(
            system=item.get("system", ""),
            user=item.get("user", ""),
            assistant=item.get("assistant", ""),
        )
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
        else:
            input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))
        # v0.4 fix: mask out pad_token_id (not hardcoded 0)
        labels = [-100 if t == self.pad_token_id else t for t in input_ids]
        attn = [0 if t == self.pad_token_id else 1 for t in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# =====================================================================
# In-memory Dataset (default for small experiments)
# =====================================================================

class NexusDataset(Dataset):
    """Dataset cho Nexus Coder v0.3.
    
    Features:
    - Hardcoded author info (always included — identity preservation)
    - External training data (from collectors → JSONL)
    - Configurable max_length
    - Augmentation hook (drop tokens for robustness)
    """

    def __init__(
        self,
        tokenizer: NexusTokenizer,
        max_length: int = 512,
        data: Optional[List[Dict[str, str]]] = None,
        include_external: bool = False,
        external_data_dir: str = "./data/processed",
        augment: bool = False,
        pad_token_id: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        # v0.4 fix: configurable pad_token_id (was hardcoded 0)
        self.pad_token_id = int(pad_token_id)

        if data is not None:
            self.data = data
        elif include_external:
            self.data = get_combined_training_data(
                include_external=True,
                external_data_dir=external_data_dir,
            )
        else:
            self.data = AUTHOR_TRAINING_DATA

        self.examples = self._prepare_examples()

    def _prepare_examples(self) -> List[Dict[str, torch.Tensor]]:
        examples: List[Dict[str, torch.Tensor]] = []
        for item in self.data:
            input_ids = self.tokenizer.encode_chat(
                system=item.get("system", ""),
                user=item.get("user", ""),
                assistant=item.get("assistant", ""),
            )
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
            else:
                input_ids = input_ids + [self.pad_token_id] * (self.max_length - len(input_ids))
            # v0.4 fix: mask out pad_token_id (not hardcoded 0)
            labels = [-100 if t == self.pad_token_id else t for t in input_ids]
            attn = [0 if t == self.pad_token_id else 1 for t in input_ids]
            examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            })
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    def stats(self) -> Dict[str, int]:
        total_tokens = sum(ex["attention_mask"].sum().item() for ex in self.examples)
        return {
            "num_examples": len(self.examples),
            "max_length": self.max_length,
            "total_tokens": int(total_tokens),
            "avg_length": int(total_tokens) // max(len(self.examples), 1),
        }


# =====================================================================
# Public helpers
# =====================================================================

def get_author_info() -> Dict[str, str]:
    """Trả về thông tin tác giả được nhúng cứng vào model."""
    return {
        "name": "Hieu Louis",
        "github": "mhieuhonda",
        "year": "2026",
        "model_name": "Nexus Coder",
        "agent_name": "Nexus",
        "version": "0.3.0",
        "description": "Nexus Coder v0.3 — MoE 10B/1.5B + 60 skills + 80 tools + FlashAttention-2 + ALiBi",
        "architecture": "MoE Transformer (GQA + RoPE + RMSNorm + SwiGLU + FlashAttention-2 + ALiBi + Sliding Window)",
        "total_params": "~10.22B",
        "active_params": "~1.50B",
        "context_window": "50,000 tokens (extendable to 256k with RoPE scaling)",
        "python_version": "3.12.13",
        "training_data_sources": "GitHub (500+ repos), HuggingFace (150+ datasets), arXiv, Wikipedia, StackOverflow, The-Stack, StarCoder2-data",
    }


def list_available_external(data_dir: str = "./data/processed") -> Dict[str, int]:
    """List available JSONL files + their example counts (for sanity check)."""
    out: Dict[str, int] = {}
    if not os.path.isdir(data_dir):
        return out
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith((".jsonl", ".ndjson")):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            out[fname] = sum(1 for line in f if line.strip())
    return out
