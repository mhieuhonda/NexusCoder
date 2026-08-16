"""Quantization - INT8/INT4/FP8 quantization cho model."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Config cho quantization."""
    method: str = "int8"  # "int8", "int4", "fp8"
    granularity: str = "per_channel"  # "per_tensor", "per_channel"
    calibration_samples: int = 128
    calibration_batches: int = 4
    skip_layers: list = None  # Layers to skip quantization
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = ["lm_head", "embed_tokens"]


class Quantizer:
    """Quantize model weights để giảm memory footprint.
    
    Supported methods:
    - INT8: 4x memory reduction, minimal quality loss
    - INT4: 8x memory reduction, slight quality loss
    - FP8: 2x memory reduction, almost no quality loss (H100 only)
    
    Usage:
        quantizer = Quantizer(config=QuantizationConfig(method="int8"))
        quantized_model = quantizer.quantize(model, calibration_data)
    """
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
    
    def quantize(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Quantize model in-place.
        
        Args:
            model: Model to quantize
            calibration_data: Sample inputs for activation calibration
        
        Returns:
            Quantized model (same object, modified in-place)
        """
        method = self.config.method
        
        if method == "int8":
            return self._quantize_int8(model, calibration_data)
        elif method == "int4":
            return self._quantize_int4(model, calibration_data)
        elif method == "fp8":
            return self._quantize_fp8(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
    
    def _quantize_int8(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor],
    ) -> nn.Module:
        """Quantize to INT8 using PyTorch dynamic quantization."""
        # Use PyTorch built-in dynamic quantization
        # Works on Linear layers
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info(f"INT8 quantization done. Memory reduced ~2x.")
        return quantized
    
    def _quantize_int4(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor],
    ) -> nn.Module:
        """Quantize to INT4 (requires bitsandbytes library)."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            logger.warning(
                "bitsandbytes not installed. Install with: pip install bitsandbytes. "
                "Falling back to INT8."
            )
            return self._quantize_int8(model, calibration_data)
        
        # Replace Linear layers with INT4 versions
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and name not in self.config.skip_layers:
                new_module = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16,
                )
                setattr(model, name, new_module)
            elif hasattr(module, "children"):
                self._quantize_int4(module, calibration_data)
        
        logger.info("INT4 quantization done. Memory reduced ~4x.")
        return model
    
    def _quantize_fp8(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.Tensor],
    ) -> nn.Module:
        """Quantize to FP8 (requires H100 GPU or newer)."""
        if not torch.cuda.is_available():
            logger.warning("FP8 requires CUDA. Falling back to INT8.")
            return self._quantize_int8(model, calibration_data)
        
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            logger.warning(f"FP8 requires H100 (compute capability 9.0+). Got {capability}. Falling back to INT8.")
            return self._quantize_int8(model, calibration_data)
        
        # FP8 conversion (when torch supports it natively)
        try:
            # Convert model to float8_e4m3fn
            for name, param in model.named_parameters():
                if name not in [f"{s}." for s in self.config.skip_layers]:
                    param.data = param.data.to(torch.float8_e4m3fn)
            logger.info("FP8 quantization done. Memory reduced ~2x.")
        except Exception as e:
            logger.warning(f"FP8 conversion failed: {e}. Falling back to INT8.")
            return self._quantize_int8(model, calibration_data)
        
        return model
    
    def estimate_memory_savings(self, model: nn.Module) -> Dict[str, float]:
        """Estimate memory savings."""
        total_params = sum(p.numel() for p in model.parameters())
        fp16_mb = (total_params * 2) / (1024 * 1024)
        int8_mb = (total_params * 1) / (1024 * 1024)
        int4_mb = (total_params * 0.5) / (1024 * 1024)
        fp8_mb = (total_params * 1) / (1024 * 1024)
        
        return {
            "fp16_mb": fp16_mb,
            "int8_mb": int8_mb,
            "int4_mb": int4_mb,
            "fp8_mb": fp8_mb,
            "int8_savings_pct": (1 - int8_mb / fp16_mb) * 100,
            "int4_savings_pct": (1 - int4_mb / fp16_mb) * 100,
        }
