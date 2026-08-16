"""ML Inference Skill - Sinh config / code cho inference serving.

Hỗ trợ vLLM, TGI, SGLang, ONNX Runtime, TensorRT, Triton Inference Server.
Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillContext, SkillCategory, SkillPriority, SkillResult


class MLInferenceSkill(Skill):
    """Sinh inference server config và serving code cho LLM / classical models."""

    category = SkillCategory.ML
    priority = SkillPriority.HIGH
    keywords: List[str] = [
        "inference", "vllm", "sglang", "tgi", "onnx", "tensorrt",
        "triton", "deploy", "serve", "serving", "quantize",
        "int8", "int4", "awq", "gptq", "batch inference", "endpoint",
    ]
    examples = [
        "Deploy a Llama-3 70B with vLLM",
        "Convert model to ONNX for serving",
        "Setup TensorRT-LLM inference server",
    ]

    @property
    def name(self) -> str:
        return "ml_inference"

    @property
    def description(self) -> str:
        return (
            "Sinh inference server config: vLLM (OpenAI-compatible API), "
            "TGI, SGLang, ONNX Runtime, TensorRT, Triton Inference Server. "
            "Hỗ trợ quantization (AWQ/GPTQ/int8) và continuous batching."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.2
        if context and context.metadata.get("framework"):
            score += 0.2
        return min(1.0, score)

    def _pick_framework(self, prompt: str) -> str:
        p = prompt.lower()
        if "vllm" in p:
            return "vllm"
        if "sglang" in p:
            return "sglang"
        if "tgi" in p:
            return "tgi"
        if "tensorrt" in p or "trt" in p:
            return "tensorrt"
        if "triton" in p:
            return "triton"
        if "onnx" in p:
            return "onnx"
        return "vllm"  # default cho LLM serving

    def execute(self, context: SkillContext) -> SkillResult:
        framework = self._pick_framework(context.prompt)
        config = self._build_config(framework, context)

        return SkillResult(
            success=True,
            output=f"[MLInference/{framework}] Serving config ready.",
            artifacts=[config],
            metadata={
                "skill": self.name,
                "framework": framework,
                "hardware": context.metadata.get("hardware", "A100"),
                "features": {
                    "continuous_batching": framework in ("vllm", "sglang", "tgi"),
                    "paged_attention": framework in ("vllm", "sglang"),
                    "tensor_parallel": framework in ("vllm", "tgi", "sglang"),
                    "speculative_decoding": framework in ("vllm", "sglang"),
                },
                "benchmarks": {
                    "throughput_target": ">= 2000 tok/s on A100",
                    "latency_p99_target": "< 500ms TTFT",
                    "concurrency": ">= 256",
                },
            },
            suggestions=[
                "Use bf16 by default; switch to fp16 only if hardware lacks bf16",
                "Enable PagedAttention for long-context workloads",
                "Pre-warm the KV cache before opening to traffic",
                "Expose /metrics for Prometheus and load-test with vLLM benchmark",
                "Pin a CUDA base image with matching driver version",
            ],
        )

    def _build_config(self, framework: str, context: SkillContext) -> Dict[str, str]:
        if framework == "vllm":
            return {"path": "serve_vllm.py", "content": _VLLM_SERVE}
        if framework == "sglang":
            return {"path": "serve_sglang.sh", "content": _SGLANG_SERVE}
        if framework == "tgi":
            return {"path": "tgi.docker-compose.yml", "content": _TGI_COMPOSE}
        if framework == "tensorrt":
            return {"path": "tensorrt_build.py", "content": _TENSORRT_BUILD}
        if framework == "triton":
            return {"path": "triton/config.pbtxt", "content": _TRITON_CONFIG}
        if framework == "onnx":
            return {"path": "serve_onnx.py", "content": _ONNX_SERVE}
        return {"path": "serve_vllm.py", "content": _VLLM_SERVE}


_VLLM_SERVE = '''"""vLLM serving — OpenAI-compatible endpoint with continuous batching."""
import os
from vllm import LLM, SamplingParams

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTIL = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))

llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=TENSOR_PARALLEL,
    gpu_memory_utilization=GPU_MEMORY_UTIL,
    max_model_len=MAX_MODEL_LEN,
    dtype="bfloat16",
    enforce_eager=False,
    quantization=os.getenv("QUANTIZATION"),  # awq | gptq | None
)

# Run server:
#   python -m vllm.entrypoints.openai.api_server \\
#     --model meta-llama/Meta-Llama-3-8B-Instruct \\
#     --tensor-parallel-size 1 --dtype bfloat16 \\
#     --max-model-len 8192 --gpu-memory-utilization 0.9
# Test:
#   curl http://localhost:8000/v1/chat/completions \\
#     -H "Content-Type: application/json" \\
#     -d '{"model":"meta-llama/Meta-Llama-3-8B-Instruct","messages":[{"role":"user","content":"hi"}]}'
'''

_SGLANG_SERVE = '''#!/usr/bin/env bash
# SGLang serving — RadixAttention for prefix caching
set -euo pipefail
MODEL_ID=${MODEL_ID:-"meta-llama/Meta-Llama-3-8B-Instruct"}
TP=${TENSOR_PARALLEL_SIZE:-1}
PORT=${PORT:-8000}

python -m sglang.launch_server \\
  --model-path "$MODEL_ID" \\
  --tp "$TP" \\
  --port "$PORT" \\
  --mem-fraction-static 0.9 \\
  --context-length 8192 \\
  --enable-radix-cache
'''

_TGI_COMPOSE = '''# Text Generation Inference (TGI) via Docker Compose
services:
  tgi:
    image: ghcr.io/huggingface/text-generation-inference:2.3
    ports: ["8080:80"]
    environment:
      MODEL_ID: meta-llama/Meta-Llama-3-8B-Instruct
      NUM_SHARD: "1"
      DTYPE: bfloat16
      MAX_BATCH_PREFILL_TOKENS: "4096"
      MAX_INPUT_LENGTH: "4096"
      MAX_TOTAL_TOKENS: "8192"
      HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN}
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: --model-id $MODEL_ID
'''

_TENSORRT_BUILD = '''"""Build a TensorRT-LLM engine from a HF checkpoint."""
import os
from tensorrt_llm import LLM, BuildConfig

MODEL_DIR = os.getenv("MODEL_DIR", "llama-3-8b-hf")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./trt_engines/llama-3-8b")

build_config = BuildConfig(
    max_input_len=4096,
    max_output_len=2048,
    max_batch_size=256,
    max_num_tokens=8192,
    builder_opt=4,
    plugin_config=None,
)

llm = LLM(
    model=MODEL_DIR,
    build_config=build_config,
    tokenizer=MODEL_DIR,
    dtype="bfloat16",
)
llm.save(OUTPUT_DIR)
# Serve via Triton with the TensorRT-LLM backend.
'''

_TRITON_CONFIG = '''# Triton Inference Server — model config (python backend)
name: "llama_python"
backend: "python"
max_batch_size: 32
input [
  { name: "prompt",   data_type: TYPE_STRING, dims: [ 1 ] },
  { name: "max_tokens", data_type: TYPE_INT32, dims: [ 1 ], optional: true }
]
output [
  { name: "text",     data_type: TYPE_STRING, dims: [ -1 ] }
]
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100000
}
instance_group [
  { kind: KIND_GPU, count: 1, gpus: [ 0 ] }
]
'''

_ONNX_SERVE = '''"""ONNX Runtime inference session with CPU/GPU EP."""
import onnxruntime as ort
import numpy as np

SESSION_OPTIONS = ort.SessionOptions()
SESSION_OPTIONS.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

PROVIDERS = [
    ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}),
    "CPUExecutionProvider",
]

def load_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, sess_options=SESSION_OPTIONS, providers=PROVIDERS)

def predict(session: ort.InferenceSession, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return session.run(None, inputs)
'''
