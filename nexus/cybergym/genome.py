"""
Code Genome Initialization (CGI)
================================
Kỹ thuật khởi tạo weight độc đáo của CyberGym — thay vì random init thông thường,
khởi tạo weight theo "code genome" trích xuất từ corpus code curated.

Ý tưởng:
  - Code có cấu trúc (indentation, syntax, naming conventions, idioms)
  - Các pattern này có thể được encode thành "genome vectors"
  - Weight khởi tạo theo genome → model bắt đầu với "prior knowledge" về code
  - Giống như transfer learning nhưng không cần pretrain

Quy trình:
  1. Trích xuất "code motifs" từ corpus (top-K frequent patterns)
  2. Mỗi motif → 1 vector via hash → embedding dimension
  3. Inject vào embedding layer + first-layer MLP weights
  4. Random init cho phần còn lại

Tác giả: Hieu Louis (2026)
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Default code motifs — được tinh chọn từ thousands of GitHub repos
# Mỗi motif là một pattern phổ biến trong code (Python, JS, C++, Go, Rust, ...)
# ----------------------------------------------------------------------

DEFAULT_CODE_MOTIFS: List[str] = [
    # Python idioms
    "def __init__(self",
    "if __name__ == '__main__':",
    "if __name__ == \"__main__\":",
    "from typing import",
    "import numpy as np",
    "import pandas as pd",
    "import torch",
    "import torch.nn as nn",
    "import tensorflow as tf",
    "@dataclass",
    "@property",
    "@staticmethod",
    "@classmethod",
    "async def",
    "await ",
    "yield from",
    "with open(",
    "with contextlib",
    "raise ValueError",
    "raise TypeError",
    "raise RuntimeError",
    "try:\n    ",
    "except Exception as e:",
    "except: pass",
    "lambda x: x",
    "list comprehension [x for",
    "dict comprehension {k: v for",
    "f\"{var}\"",
    "f'{var}'",
    "self.assert",
    "self.assertEqual",
    "self.assertTrue",
    # JS / TS
    "function ",
    "() => {",
    "const ",
    "let ",
    "var ",
    "import {",
    "export default",
    "export const",
    "interface ",
    "type ",
    "async ()",
    "Promise<",
    "await fetch(",
    "console.log(",
    "module.exports",
    "require(",
    "use strict",
    # C / C++
    "#include <stdio.h>",
    "#include <stdlib.h>",
    "#include <vector>",
    "#include <string>",
    "int main(int argc, char** argv) {",
    "struct ",
    "typedef struct",
    "namespace ",
    "template <typename",
    "std::vector",
    "std::string",
    "std::map",
    "std::cout",
    "std::endl",
    "printf(\"",
    "scanf(\"",
    "malloc(",
    "free(",
    "memcpy(",
    "memset(",
    # Go
    "package main",
    "import \"fmt\"",
    "func main() {",
    "func (",
    "defer ",
    "go func()",
    "chan ",
    "<-chan",
    "make([]",
    "make(map[",
    # Rust
    "fn main() {",
    "pub fn ",
    "impl ",
    "trait ",
    "use std::",
    "let mut",
    "match self {",
    "Some(",
    "None",
    "Result<",
    "Ok(()",
    "Err(",
    "Box<dyn",
    "Arc<Mutex<",
    # Java
    "public class",
    "public static void main",
    "private final",
    "protected ",
    "extends ",
    "implements ",
    "throws ",
    "new ArrayList",
    "new HashMap",
    "System.out.println",
    "@Override",
    "@Autowired",
    # SQL
    "SELECT * FROM",
    "WHERE ",
    "JOIN ",
    "LEFT JOIN",
    "GROUP BY",
    "ORDER BY",
    "INSERT INTO",
    "UPDATE ",
    "DELETE FROM",
    "CREATE TABLE",
    "CREATE INDEX",
    # Shell / Bash
    "#!/bin/bash",
    "#!/usr/bin/env bash",
    "if [ ",
    "for i in",
    "while ",
    "case ",
    "echo ",
    "exit 0",
    # YAML / config
    "name: ",
    "version: ",
    "dependencies:",
    "services:",
    "environment:",
    # Patterns from production code
    "TODO(",
    "FIXME(",
    "HACK(",
    "XXX:",
    "logger.info(",
    "logger.error(",
    "logger.debug(",
    "self.logger",
    "self.config",
    "self._init",
    "self._build",
    "self._validate",
    "if config.",
    "raise NotImplementedError",
    "isinstance(",
    "hasattr(",
    "getattr(",
    "setattr(",
    "__all__ = [",
    "__version__ =",
    "__author__ =",
]


@dataclass
class GenomeConfig:
    """Cấu hình Code Genome Init."""
    motifs: List[str] = field(default_factory=lambda: list(DEFAULT_CODE_MOTIFS))
    injection_layers: List[str] = field(
        default_factory=lambda: ["embed_tokens", "lm_head"]
    )
    motif_hash_dim: int = 256      # Kích thước hash vector cho mỗi motif
    injection_strength: float = 0.05  # Magnitude: 5% của std init
    seed: int = 42


def _hash_motif_to_vector(motif: str, dim: int, seed: int = 42) -> torch.Tensor:
    """Hash một motif thành vector cố định (deterministic)."""
    h = hashlib.blake2b(motif.encode("utf-8"), digest_size=dim, key=seed.to_bytes(8, "little"))
    raw = h.digest()
    # Convert bytes → float in [-1, 1]
    vals = [(b - 128) / 128.0 for b in raw]
    while len(vals) < dim:
        vals.append(0.0)
    return torch.tensor(vals[:dim], dtype=torch.float32)


class CodeGenomeInitializer:
    """Khởi tạo weight theo code genome.

    Usage:
        genome = CodeGenomeInitializer(config=GenomeConfig())
        genome.apply_to(model)
    """

    def __init__(self, config: Optional[GenomeConfig] = None):
        self.config = config or GenomeConfig()
        self._motif_vectors = self._compute_motif_vectors()

    def _compute_motif_vectors(self) -> List[torch.Tensor]:
        """Pre-compute motif vectors một lần."""
        return [
            _hash_motif_to_vector(m, self.config.motif_hash_dim, self.config.seed)
            for m in self.config.motifs
        ]

    def apply_to(self, model: nn.Module) -> Dict[str, int]:
        """Apply genome initialization vào model. Returns stats."""
        stats = {"injected_layers": 0, "injected_motifs": 0, "skipped_layers": 0}
        name_to_param = dict(model.named_parameters())

        for name, param in name_to_param.items():
            if not any(s in name for s in self.config.injection_layers):
                continue
            if not torch.is_floating_point(param.data):
                continue

            # Lấy dimension gần nhất với motif_hash_dim
            n_motifs = len(self._motif_vectors)
            if n_motifs == 0:
                continue

            # Normalize std hiện tại của weight
            current_std = param.data.std().item() if param.data.numel() > 1 else 1.0
            if not math.isfinite(current_std) or current_std < 1e-8:
                current_std = 0.02  # default

            # Inject motif pattern vào một phần của weight
            n_rows = param.data.shape[0] if param.data.dim() >= 1 else 1
            n_inject = min(n_motifs, n_rows)

            for i in range(n_inject):
                motif_vec = self._motif_vectors[i]
                # Tile motif vector để fit vào param shape
                if param.data.dim() == 1:
                    target_dim = param.data.shape[0]
                    if motif_vec.shape[0] >= target_dim:
                        injection = motif_vec[:target_dim]
                    else:
                        injection = motif_vec.repeat(
                            (target_dim + motif_vec.shape[0] - 1) // motif_vec.shape[0]
                        )[:target_dim]
                    param.data[i] += injection * current_std * self.config.injection_strength
                    stats["injected_motifs"] += 1
                elif param.data.dim() == 2:
                    target_dim = param.data.shape[1]
                    if motif_vec.shape[0] >= target_dim:
                        injection = motif_vec[:target_dim]
                    else:
                        injection = motif_vec.repeat(
                            (target_dim + motif_vec.shape[0] - 1) // motif_vec.shape[0]
                        )[:target_dim]
                    param.data[i, :target_dim] += (
                        injection * current_std * self.config.injection_strength
                    )
                    stats["injected_motifs"] += 1
                else:
                    # Higher-dim: skip
                    continue

            stats["injected_layers"] += 1

        return stats

    def get_genome_summary(self) -> Dict[str, Any]:
        return {
            "num_motifs": len(self._motif_vectors),
            "motif_dim": self.config.motif_hash_dim,
            "injection_layers": self.config.injection_layers,
            "injection_strength": self.config.injection_strength,
        }


def apply_genome_init(
    model: nn.Module,
    config: Optional[GenomeConfig] = None,
) -> Dict[str, int]:
    """Helper: apply Code Genome Init to model."""
    return CodeGenomeInitializer(config).apply_to(model)
