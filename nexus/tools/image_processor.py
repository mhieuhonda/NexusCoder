"""
Image Processor Tool - Resize/crop/rotate/convert/watermark/info ảnh.
===========================================
Dùng Pillow (PIL) lazy import. Hỗ trợ JPEG/PNG/GIF/WEBP/BMP/TIFF.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


OPERATIONS = {"resize", "crop", "rotate", "convert", "watermark", "info", "thumbnail"}
SUPPORTED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF"}


class ImageProcessorTool(Tool):
    """Xử lý ảnh: resize, crop, rotate, convert, watermark, info."""

    category = ToolCategory.MEDIA
    safety = ToolSafety.MODERATE
    requires_confirmation = True

    @property
    def name(self) -> str:
        return "image_processor"

    @property
    def description(self) -> str:
        return "Image ops (resize/crop/rotate/convert/watermark/info) bằng Pillow."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_path": {"type": "string"},
                "output_path": {"type": "string", "description": "Output file (bỏ qua cho 'info')"},
                "operation": {
                    "type": "string",
                    "enum": sorted(OPERATIONS),
                    "default": "info",
                },
                "params": {
                    "type": "object",
                    "description": "Operation-specific params (width/height/crop_box/angle/format/...)",
                },
            },
            "required": ["input_path", "operation"],
        }

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        if not args.get("input_path"):
            return "Missing required arg: input_path"
        op = args.get("operation", "info")
        if op not in OPERATIONS:
            return f"Invalid operation='{op}'. Supported: {sorted(OPERATIONS)}"
        if op != "info" and not args.get("output_path"):
            return f"Missing required arg: output_path (cho operation='{op}')"
        return None

    # ---- Helpers --------------------------------------------------------

    def _watermark(self, img: Any, text: str) -> Any:
        """Vẽ watermark text lên ảnh bằng ImageDraw (stdlib của PIL)."""
        from PIL import ImageDraw, ImageFont  # type: ignore
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(12, img.width // 30))
        except Exception:
            font = ImageFont.load_default()
        # Tính bounding box text / compute text bbox
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font.getsize(text)
        x = img.width - tw - 10
        y = img.height - th - 10
        # Shadow + text / shadow + main text
        draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 128), font=font)
        draw.text((x, y), text, fill=(255, 255, 255, 200), font=font)
        return img

    # ---- Execute --------------------------------------------------------

    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        input_path = args["input_path"]
        output_path = args.get("output_path")
        op = args.get("operation", "info")
        params: Dict[str, Any] = args.get("params", {}) or {}

        if not os.path.exists(input_path):
            return ToolResult(success=False, error=f"Input không tồn tại: {input_path}", return_code=1)

        if context.dry_run:
            return ToolResult(
                success=True,
                output=f"[dry-run] Sẽ thực hiện '{op}' trên {input_path}",
                metadata={"operation": op, "input_path": input_path, "dry_run": True},
            )

        try:
            from PIL import Image  # type: ignore
        except ImportError:
            return ToolResult(
                success=False,
                error="Pillow chưa cài. Cài đặt: pip install Pillow",
                return_code=127,
            )

        try:
            img = Image.open(input_path)
            # Force load để detect lỗi sớm / force load to catch errors early
            img.load()
            fmt = img.format or os.path.splitext(input_path)[1][1:].upper()

            if op == "info":
                meta: Dict[str, Any] = {
                    "path": input_path,
                    "format": fmt,
                    "size": list(img.size),
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "is_animated": getattr(img, "is_animated", False),
                    "n_frames": getattr(img, "n_frames", 1),
                    "file_size_bytes": os.path.getsize(input_path),
                }
                # EXIF / read EXIF if available
                try:
                    exif = img._getexif()  # type: ignore[attr-defined]
                    if exif:
                        meta["exif"] = {k: str(v) for k, v in exif.items()}
                except Exception:
                    pass
                return ToolResult(
                    success=True,
                    output=f"{fmt} {img.width}x{img.height} mode={img.mode} ({os.path.getsize(input_path)} bytes)",
                    metadata=meta,
                )

            if op == "resize":
                w = int(params.get("width", img.width // 2))
                h = int(params.get("height", img.height // 2))
                resample_name = params.get("resample", "LANCZOS")
                resample = getattr(Image, resample_name, Image.LANCZOS)
                result = img.resize((w, h), resample)
            elif op == "thumbnail":
                w = int(params.get("width", 256))
                h = int(params.get("height", 256))
                result = img.copy()
                result.thumbnail((w, h), Image.LANCZOS)
            elif op == "crop":
                box = params.get("crop_box") or (
                    int(params.get("left", 0)),
                    int(params.get("top", 0)),
                    int(params.get("right", img.width)),
                    int(params.get("bottom", img.height)),
                )
                result = img.crop(box)
            elif op == "rotate":
                angle = float(params.get("angle", 90))
                expand = bool(params.get("expand", True))
                result = img.rotate(angle, expand=expand)
            elif op == "convert":
                target_fmt = str(params.get("format", "PNG")).upper()
                if target_fmt not in SUPPORTED_FORMATS:
                    return ToolResult(success=False, error=f"Unsupported format: {target_fmt}", return_code=1)
                # Convert mode nếu cần / convert mode for format
                if target_fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
                    result = img.convert("RGB")
                else:
                    result = img.copy()
                fmt = target_fmt
            elif op == "watermark":
                text = str(params.get("text", "© Nexus Coder"))
                result = img.copy()
                if result.mode != "RGBA":
                    result = result.convert("RGBA")
                result = self._watermark(result, text)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {op}", return_code=1)

            # Tự suy ra format từ extension / infer format from output extension
            save_fmt = fmt
            ext = os.path.splitext(output_path)[1][1:].upper() if output_path else None
            if ext and ext in SUPPORTED_FORMATS:
                save_fmt = ext
            # Đảm bảo thư mục cha / ensure parent dir
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            save_kwargs: Dict[str, Any] = {}
            if save_fmt == "JPEG":
                save_kwargs["quality"] = int(params.get("quality", 85))
            result.save(output_path, format=save_fmt, **save_kwargs)
            return ToolResult(
                success=True,
                output=f"{op} → {output_path} ({result.width}x{result.height}, {save_fmt})",
                artifacts=[output_path],
                metadata={
                    "operation": op,
                    "input_path": input_path,
                    "output_path": output_path,
                    "input_size": list(img.size),
                    "output_size": list(result.size),
                    "format": save_fmt,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Image op failed: {e}", return_code=1)
