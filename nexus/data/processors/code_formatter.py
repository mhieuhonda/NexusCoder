"""Code Formatter - Format code samples cho training."""
from __future__ import annotations

import re
from typing import Dict, Any, List, Optional


class CodeFormatter:
    """Format code samples cho training.
    
    Features:
    - Strip excessive blank lines
    - Normalize indentation
    - Add language tags to code blocks
    - Wrap code in markdown fences if needed
    - Detect language automatically
    """
    
    LANG_BY_EXT = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".rb": "ruby", ".php": "php",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".sql": "sql", ".sh": "bash", ".bash": "bash",
        ".html": "html", ".css": "css", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
        ".xml": "xml", ".md": "markdown",
    }
    
    # Language detection patterns
    LANG_PATTERNS = {
        "python": [r"^\s*def\s+\w+", r"^\s*class\s+\w+", r"^\s*import\s+\w+", r"^\s*from\s+\w+\s+import"],
        "javascript": [r"^\s*function\s+\w+", r"^\s*const\s+\w+\s*=", r"^\s*let\s+\w+\s*=", r"=>\s*\{?"],
        "typescript": [r":\s*(string|number|boolean|void|any)\b", r"interface\s+\w+", r"type\s+\w+\s*="],
        "go": [r"^\s*func\s+\w+", r"^\s*package\s+\w+", r"^\s*import\s+\("],
        "rust": [r"^\s*fn\s+\w+", r"^\s*impl\s+\w+", r"^\s*use\s+\w+", r"^\s*let\s+mut\s+"],
        "java": [r"^\s*public\s+class\s+\w+", r"^\s*private\s+\w+\s+\w+", r"^\s*import\s+java\."],
        "c": [r"^\s*#include\s*<", r"^\s*int\s+main\s*\("],
        "cpp": [r"^\s*#include\s*<", r"^\s*std::", r"^\s*template\s*<"],
    }
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> Optional[str]:
        """Detect programming language of code."""
        if filename:
            import os
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.LANG_BY_EXT:
                return self.LANG_BY_EXT[ext]
        
        # Pattern matching
        for lang, patterns in self.LANG_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return lang
        
        return None
    
    def format(self, code: str, language: Optional[str] = None) -> str:
        """Format code sample."""
        # Detect language if not provided
        if not language:
            language = self.detect_language(code) or ""
        
        # Strip trailing whitespace on each line
        lines = [line.rstrip() for line in code.splitlines()]
        
        # Remove excessive blank lines (max 2 consecutive)
        formatted_lines = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    formatted_lines.append("")
            else:
                blank_count = 0
                formatted_lines.append(line)
        
        # Remove leading/trailing blank lines
        while formatted_lines and formatted_lines[0] == "":
            formatted_lines.pop(0)
        while formatted_lines and formatted_lines[-1] == "":
            formatted_lines.pop()
        
        code_clean = "\n".join(formatted_lines)
        
        return code_clean
    
    def wrap_in_markdown(self, code: str, language: Optional[str] = None) -> str:
        """Wrap code in markdown fence."""
        if not language:
            language = self.detect_language(code) or ""
        return f"```{language}\n{code}\n```"
    
    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a code sample."""
        sample = dict(sample)
        text = sample.get("text", "")
        language = sample.get("language") or sample.get("metadata", {}).get("language")
        
        # Check if it's code
        is_code = (
            sample.get("language") or
            sample.get("metadata", {}).get("language") or
            self.detect_language(text) is not None
        )
        
        if is_code:
            formatted = self.format(text, language)
            sample["text"] = formatted
            sample["metadata"] = sample.get("metadata", {})
            sample["metadata"]["formatted"] = True
            if not language:
                language = self.detect_language(text)
            sample["metadata"]["detected_language"] = language
        
        return sample
