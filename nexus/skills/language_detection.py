"""Language Detection Skill - langdetect + fasttext + ISO 639-1 mapping.

Sinh code phát hiện ngôn ngữ: langdetect (pure Python), fasttext (Facebook,
175 languages, fast & accurate), fallback heuristics, batch processing, và
ISO 639-1 (2-letter) → ISO 639-3 → language name mapping.

Author: Hieu Louis (2026)
"""
from __future__ import annotations

from typing import Dict, List

from .base import Skill, SkillCategory, SkillContext, SkillPriority, SkillResult


LANGDETECT_CODE = '''"""Language detection with langdetect / Phát hiện ngôn ngữ bằng langdetect."""
from __future__ import annotations
from typing import Dict, List, Optional
import re
import pandas as pd

try:
    from langdetect import detect, detect_langs, DetectorFactory
    DetectorFactory.seed = 0   # deterministic / xác định
    HAS_LD = True
except ImportError:
    HAS_LD = False


def detect_single(text: str) -> Optional[Dict[str, object]]:
    """Detect ngôn ngữ của một văn bản."""
    text = _normalize(text)
    if len(text) < 10:
        return None   # too short / quá ngắn
    if not HAS_LD:
        return {"error": "langdetect not installed"}
    try:
        probs = detect_langs(text)
        lang = probs[0].lang
        return {
            "lang_code": lang,
            "lang_name": ISO_639_1.get(lang, "Unknown"),
            "confidence": float(probs[0].prob),
            "all_candidates": [(l.lang, float(l.prob)) for l in probs],
        }
    except Exception as e:
        return {"error": str(e)}


def detect_batch(texts: List[str]) -> pd.DataFrame:
    """Detect cho nhiều văn bản / Phát hiện hàng loạt."""
    rows = []
    for i, t in enumerate(texts):
        r = detect_single(t) or {}
        rows.append({"index": i, "text_preview": t[:60], **r})
    return pd.DataFrame(rows)


def _normalize(text: str) -> str:
    """Loại bỏ URL, emoji, dấu câu dư thừa."""
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


# Heuristic fallback / Dự phòng bằng heuristic
SCRIPT_HINTS = {
    "vi": ["đ", "ă", "â", "ê", "ô", "ơ", "ư"],
    "ja": ["の", "は", "です", "ます"],
    "ko": ["입니다", "하는", "그리고"],
    "zh": ["的", "是", "和", "中国"],
    "ru": ["ый", "ость", "ние", "ться"],
    "ar": ["ال", "ون", "ين"],
    "th": ["คือ", "และ", "ที่"],
}


def heuristic_detect(text: str) -> Optional[str]:
    """Fallback khi không có thư viện / Dự phòng khi thiếu thư viện."""
    text_lower = text.lower()
    for code, hints in SCRIPT_HINTS.items():
        if any(h in text_lower for h in hints):
            return code
    return None


ISO_639_1: Dict[str, str] = {
    "en": "English", "vi": "Vietnamese", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "ar": "Arabic", "hi": "Hindi",
    "th": "Thai", "id": "Indonesian", "tr": "Turkish", "pl": "Polish",
    "sv": "Swedish", "fi": "Finnish", "da": "Danish", "no": "Norwegian",
    "cs": "Czech", "el": "Greek", "he": "Hebrew", "ro": "Romanian",
    "hu": "Hungarian", "uk": "Ukrainian", "ms": "Malay", "fa": "Persian",
}

ISO_639_1_TO_639_3: Dict[str, str] = {
    "en": "eng", "vi": "vie", "fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt": "por", "nl": "nld", "ru": "rus", "ja": "jpn",
    "ko": "kor", "zh-cn": "zho", "zh-tw": "zho", "ar": "ara", "hi": "hin",
    "th": "tha", "id": "ind", "tr": "tur", "pl": "pol", "sv": "swe",
    "fi": "fin", "da": "dan", "no": "nor", "cs": "ces", "el": "ell",
    "he": "heb", "ro": "ron", "hu": "hun", "uk": "ukr", "ms": "msa", "fa": "fas",
}


if __name__ == "__main__":
    samples = [
        "Xin chào, đây là một câu tiếng Việt.",
        "Hello world, this is an English sentence.",
        "Bonjour, ceci est une phrase française.",
        "こんにちは、これは日本語の文です。",
    ]
    print(detect_batch(samples))
    # Heuristic fallback demo / Demo heuristic
    print("heuristic:", heuristic_detect("私の名前は田中です。の は です"))
'''

FASTTEXT_CODE = '''"""fasttext-based detection (175 languages) / Phát hiện với fasttext."""
from __future__ import annotations
from typing import Dict, List
import pandas as pd

try:
    import fasttext
    # Download pretrained model:
    #   wget https://dl.fbaipublicfiles.com/nllb/lid.176.bin
    MODEL = fasttext.load_model("lid.176.bin")
    HAS_FT = True
except Exception:
    HAS_FT = False


def detect_fasttext(text: str, k: int = 3) -> Dict[str, object]:
    """Phát hiện ngôn ngữ với fasttext (top-k)."""
    if not HAS_FT:
        return {"error": "fasttext model not found — download lid.176.bin"}
    text = text.replace("\\n", " ").strip()
    labels, probs = MODEL.predict(text, k=k)
    return {
        "lang_codes": [lab.replace("__label__", "") for lab in labels],
        "probabilities": [float(p) for p in probs],
        "top_lang": labels[0].replace("__label__", ""),
        "confidence": float(probs[0]),
    }


def detect_batch_fasttext(texts: List[str], k: int = 1) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(texts):
        r = detect_fasttext(t, k=k)
        rows.append({"index": i, "text_preview": t[:60], **r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(detect_fasttext("Hello, how are you doing today?"))
'''

STRATEGY = """
Language Detection — Strategy Guide / Hướng dẫn chiến lược
=============================================================
1. libChromium / cld3 (compact language detector v3):
   - Memory-efficient, neural, good for short texts.

2. langdetect (Python port of Google's language-detection):
   - Pure-Python, no extra deps. Non-deterministic by default → set seed.
   - Struggles with very short texts (<20 chars) and code-mixed content.

3. fasttext lid.176 (Facebook):
   - Best accuracy (176 languages), very fast (CPU). Needs model file (~125MB).
   - Recommended for production multilingual systems.

4. Heuristic fallback:
   - Use Unicode script ranges + common stopwords for unknown-language backup.

Tips:
  - Strip URLs, mentions, emojis before detection.
  - Require minimum 10-20 characters for reliable detection.
  - For code-mixed text (Hinglish, Spanglish), combine with script-based heuristics.
  - Cache results — language doesn't change for the same source.

ISO codes:
  - ISO 639-1 (2-letter, e.g. en/vi/fr) — most common, used by langdetect.
  - ISO 639-3 (3-letter, e.g. eng/vie/fra) — used by fasttext & NLLB.
"""


class LanguageDetectionSkill(Skill):
    """Sinh code phát hiện ngôn ngữ (langdetect + fasttext + heuristic)."""

    category = SkillCategory.LANGUAGE
    priority = SkillPriority.LOW
    keywords: List[str] = [
        "language detection", "langdetect", "detect language",
        "what language", "which language", "language identification",
        "fasttext lid", "lingua", "cld3", "iso 639",
    ]
    examples = [
        "Detect ngôn ngữ của văn bản",
        "Batch language detection cho dataset",
        "Map language code sang ISO 639-3",
    ]

    @property
    def name(self) -> str:
        return "language_detection"

    @property
    def description(self) -> str:
        return (
            "Sinh pipeline phát hiện ngôn ngữ: langdetect (Python), fasttext lid.176 "
            "(176 langs, production-grade), heuristic fallback, ISO 639-1 ↔ 639-3 mapping."
        )

    def can_handle(self, prompt: str, context: SkillContext = None) -> float:
        prompt_lower = prompt.lower()
        score = 0.0
        for kw in self.keywords:
            if kw in prompt_lower:
                score += 0.18
        return min(1.0, score)

    def execute(self, context: SkillContext) -> SkillResult:
        prompt_lower = (context.prompt or "").lower()
        if "fasttext" in prompt_lower:
            recommended = "fasttext"
        elif "heuristic" in prompt_lower:
            recommended = "heuristic"
        else:
            recommended = "langdetect"

        artifacts: List[Dict[str, str]] = [
            {"name": "langdetect_pipeline.py", "language": "python", "content": LANGDETECT_CODE},
            {"name": "fasttext_pipeline.py", "language": "python", "content": FASTTEXT_CODE},
            {"name": "STRATEGY.md", "language": "markdown", "content": STRATEGY},
        ]

        return SkillResult(
            success=True,
            output=(
                f"[language_detection] recommended={recommended}\n"
                f"Generated langdetect + fasttext (176 langs) + heuristic fallback "
                f"pipelines with ISO 639-1 ↔ 639-3 mapping."
            ),
            artifacts=artifacts,
            suggestions=[
                "Set DetectorFactory.seed = 0 for reproducible langdetect results",
                "Use fasttext lid.176.bin for production (best accuracy)",
                "Strip URLs, mentions, emojis before detection",
                "Require minimum 10-20 characters for reliable detection",
                "Cache results — language doesn't change for the same source",
            ],
            metadata={
                "skill": self.name,
                "recommended": recommended,
                "libraries": ["langdetect", "fasttext", "heuristic"],
                "languages_supported": 176,
                "iso_codes": ["639-1", "639-3"],
                "version": self.version,
                "author": self.author,
            },
        )
