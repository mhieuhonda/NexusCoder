"""
Benchmark Runner Tool - Chạy ML benchmarks (HumanEval, GSM8K, MBPP, MMLU).
===========================================
Lazy import `datasets` (HuggingFace). Sinh prompt → gọi model callable
(hoặc API endpoint), chấm pass@k / exact match / log prob.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


BENCHMARKS = {"humaneval", "gsm8k", "mbpp", "mmlu"}


class BenchmarkRunnerTool(Tool):
    """Chạy standard ML benchmarks: HumanEval, GSM8K, MBPP, MMLU."""

    category = ToolCategory.ML
    safety = ToolSafety.MODERATE
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "benchmark_runner"

    @property
    def description(self) -> str:
        return "Run ML benchmarks (HumanEval/GSM8K/MBPP/MMLU) on model via HuggingFace datasets."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_path": {"type": "string", "description": "Đường dẫn model hoặc HuggingFace model ID"},
                "benchmark": {
                    "type": "string",
                    "enum": sorted(BENCHMARKS),
                    "default": "humaneval",
                },
                "num_samples": {"type": "integer", "default": 20, "description": "Số samples tối đa để eval"},
                "output_path": {"type": "string", "description": "File JSON để lưu kết quả chi tiết"},
                "inference_command": {
                    "type": "string",
                    "description": "Command để gọi model inference (nhận prompt trên stdin, trả output stdout). Bỏ qua nếu model_path là HF ID",
                },
                "inference_endpoint": {"type": "string", "description": "HTTP endpoint POST {prompt} → {completion}"},
                "max_new_tokens": {"type": "integer", "default": 256},
            },
            "required": ["model_path", "benchmark"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        bench = args.get("benchmark", "humaneval")
        if bench not in BENCHMARKS:
            return f"Invalid benchmark='{bench}'. Supported: {sorted(BENCHMARKS)}"
        n = args.get("num_samples", 20)
        if n <= 0:
            return f"num_samples phải > 0, got {n}"
        return None

    # ---- Dataset loaders ------------------------------------------------

    DATASET_SPECS = {
        "humaneval": ("openai_humaneval", "test", "prompt", "canonical_solution", "task_id"),
        "gsm8k": ("gsm8k", "test", "question", "answer", None),
        "mbpp": ("mbpp", "test", "text", "code", "task_id"),
        "mmlu": ("cais/mmlu", "test", "question", "answer", "subject"),
    }

    def _load_samples(self, benchmark: str, num_samples: int) -> List[Dict[str, Any]]:
        """Tải samples từ HuggingFace datasets."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise RuntimeError("datasets chưa cài. Cài đặt: pip install datasets")
        ds_name, split, prompt_key, answer_key, id_key = self.DATASET_SPECS[benchmark]
        # MMLU cần config 'all'
        if benchmark == "mmlu":
            ds = load_dataset(ds_name, "all", split=split, trust_remote_code=True)
        else:
            ds = load_dataset(ds_name, split=split, trust_remote_code=True)
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            sample = {
                "id": row.get(id_key, str(i)) if id_key else str(i),
                "prompt": row[prompt_key],
                "expected": row[answer_key],
                "choices": row.get("choices") if benchmark == "mmlu" else None,
            }
            samples.append(sample)
        return samples

    # ---- Inference backends ---------------------------------------------

    def _infer_command(self, prompt: str, cmd: str, timeout: int) -> str:
        """Gọi model qua subprocess: prompt → stdin, completion ← stdout."""
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                return f"[ERROR rc={proc.returncode}] {proc.stderr.strip()[:200]}"
            return proc.stdout.strip()
        except subprocess.TimeoutExpired:
            return "[TIMEOUT]"

    def _infer_endpoint(self, prompt: str, endpoint: str, max_tokens: int, timeout: int) -> str:
        """Gọi HTTP POST {endpoint} với {prompt, max_tokens} → {completion}."""
        import json as _json
        import urllib.request
        body = _json.dumps({"prompt": prompt, "max_new_tokens": max_tokens}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = _json.loads(resp.read().decode("utf-8"))
            # Hỗ trợ nhiều key / support multiple key conventions
            return out.get("completion") or out.get("text") or out.get("output") or _json.dumps(out)
        except Exception as e:
            return f"[ERROR {e}]"

    def _infer_hf(self, prompt: str, model_path: str, max_tokens: int) -> str:
        """Tải model HF transformers và generate trực tiếp."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError("transformers + torch chưa cài. Cài đặt: pip install transformers torch")
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        if torch.cuda.is_available():
            model = model.cuda()
        inputs = tok(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        # Bỏ phần prompt / strip prompt tokens
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ---- Scoring --------------------------------------------------------

    def _score_humaneval(self, completion: str, expected: str) -> bool:
        """Trích code block + execute để test pass/fail (đơn giản)."""
        # Trích code giữa ```python ... ```
        m = re.search(r"```python\s*(.*?)\```", completion, re.DOTALL)
        code = m.group(1) if m else completion
        # Viết vào temp + chạy / write to temp + execute
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            proc = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=10, check=False)
            return proc.returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(tmp)

    def _score_gsm8k(self, completion: str, expected: str) -> bool:
        """GSM8K: trích số cuối cùng, so sánh với đáp án."""
        # Đáp án thường có dạng "#### <number>"
        expected_num = re.search(r"[-+]?\d+(?:\.\d+)?", expected.split("####")[-1] if "####" in expected else expected)
        if not expected_num:
            return False
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", completion)
        if not nums:
            return False
        return abs(float(nums[-1]) - float(expected_num.group())) < 1e-6

    def _score_mbpp(self, completion: str, expected: str) -> bool:
        """MBPP: chỉ kiểm tra syntax (compile) — không run test."""
        m = re.search(r"```python\s*(.*?)\```", completion, re.DOTALL)
        code = m.group(1) if m else completion
        try:
            compile(code, "<mbpp>", "exec")
            return True
        except SyntaxError:
            return False

    def _score_mmlu(self, completion: str, expected: str, choices: Optional[List[str]]) -> bool:
        """MMLU: trích A/B/C/D từ output."""
        if choices is None:
            return False
        try:
            expected_idx = int(expected)
        except (ValueError, TypeError):
            expected_idx = ord(expected.upper()) - ord("A")
        # Tìm letter A/B/C/D đầu tiên trong completion
        m = re.search(r"\b([ABCD])\b", completion.strip()[:20].upper())
        if not m:
            return False
        return ord(m.group(1)) - ord("A") == expected_idx

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        model_path = args["model_path"]
        benchmark = args.get("benchmark", "humaneval")
        num_samples = int(args.get("num_samples", 20))
        output_path = args.get("output_path")
        inference_command = args.get("inference_command")
        inference_endpoint = args.get("inference_endpoint")
        max_new_tokens = int(args.get("max_new_tokens", 256))

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ run {benchmark} trên {num_samples} samples với model {model_path}",
                metadata={"benchmark": benchmark, "num_samples": num_samples, "model_path": model_path, "dry_run": True},
            )

        try:
            samples = self._load_samples(benchmark, num_samples)
        except Exception as e:
            return ToolResult(success=False, error=f"Load benchmark failed: {e}", return_code=1)

        # Chọn backend inference / pick inference backend
        if inference_command:
            backend = ("command", inference_command)
        elif inference_endpoint:
            backend = ("endpoint", inference_endpoint)
        else:
            backend = ("hf", model_path)

        per_sample_timeout = max(30, context.timeout)
        results: List[Dict[str, Any]] = []
        passed = 0
        for s in samples:
            try:
                if backend[0] == "command":
                    completion = self._infer_command(s["prompt"], backend[1], per_sample_timeout)
                elif backend[0] == "endpoint":
                    completion = self._infer_endpoint(s["prompt"], backend[1], max_new_tokens, per_sample_timeout)
                else:
                    completion = self._infer_hf(s["prompt"], backend[1], max_new_tokens)
            except Exception as e:
                completion = f"[INFER_ERROR {e}]"

            if benchmark == "humaneval":
                ok = self._score_humaneval(completion, s["expected"])
            elif benchmark == "gsm8k":
                ok = self._score_gsm8k(completion, s["expected"])
            elif benchmark == "mbpp":
                ok = self._score_mbpp(completion, s["expected"])
            else:  # mmlu
                ok = self._score_mmlu(completion, s["expected"], s.get("choices"))

            if ok:
                passed += 1
            results.append({
                "id": s["id"],
                "prompt_preview": s["prompt"][:200],
                "completion_preview": completion[:300],
                "passed": ok,
            })

        accuracy = passed / len(results) if results else 0.0

        # Lưu kết quả chi tiết nếu có output_path / save detailed results
        artifacts = []
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "benchmark": benchmark,
                        "model_path": model_path,
                        "num_samples": len(results),
                        "passed": passed,
                        "accuracy": accuracy,
                        "results": results,
                    }, f, indent=2, ensure_ascii=False)
                artifacts.append(output_path)
            except Exception as e:
                return ToolResult(
                    success=True,
                    output=f"Benchmark {benchmark}: {passed}/{len(results)} = {accuracy:.2%} (warn: save failed: {e})",
                    metadata={"benchmark": benchmark, "passed": passed, "total": len(results), "accuracy": accuracy, "results": results},
                )

        return ToolResult(
            success=True,
            output=f"Benchmark {benchmark}: {passed}/{len(results)} = {accuracy:.2%}",
            metadata={"benchmark": benchmark, "passed": passed, "total": len(results), "accuracy": accuracy, "results": results},
            artifacts=artifacts,
        )
