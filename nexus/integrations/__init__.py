"""
Nexus Coder Integrations v0.3
=============================
Adapters / ported utilities from open-source ML frameworks.

These adapters are inspired by (and copy best practices from) the following
open-source projects. All credit for the original algorithms goes to their
respective authors. The code here is rewritten to integrate cleanly into
the Nexus Coder architecture; it is NOT a vendored copy.

Attribution:
  - litgpt       (Lightning AI, Apache 2.0)         — RoPE scaling, FusedLinear
  - LlamaFactory (hiyouga, Apache 2.0)              — dataset format converters
  - axolotl      (axolotl-ai-cloud, Apache 2.0)     — training config schema
  - OpenHands    (OpenHands, MIT)                    — agent loop patterns
  - omp-gym      (dylantirandaz, MIT)                — OpenMP benchmark hooks

Each adapter module exposes a small public API. They are OPTIONAL — Nexus Coder
does not require these frameworks to be installed.
"""
from __future__ import annotations

__all__ = ["litgpt", "llamafactory", "axolotl", "openhands", "omp_gym"]
