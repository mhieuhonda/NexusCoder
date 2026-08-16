"""Crypto Tools - hashing, encryption."""
from __future__ import annotations

import hashlib
import hmac
import base64
from typing import Dict, Any

from .base import Tool, ToolResult, ToolContext, ToolCategory, ToolSafety


class HashTool(Tool):
    """Compute hash của data/file."""
    category = ToolCategory.CRYPTO
    safety = ToolSafety.SAFE
    
    @property
    def name(self) -> str:
        return "hash"
    
    @property
    def description(self) -> str:
        return "Compute hash (md5, sha1, sha256, sha512, blake2) của string hoặc file."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "String or file path"},
                "algorithm": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512", "blake2b", "blake2s"], "default": "sha256"},
                "is_file": {"type": "boolean", "default": False},
            },
            "required": ["input"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        import os
        
        inp = args["input"]
        algorithm = args.get("algorithm", "sha256")
        is_file = args.get("is_file", False)
        
        try:
            h = hashlib.new(algorithm)
            
            if is_file or (os.path.exists(inp) and len(inp) < 1024):
                with open(inp, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                source = f"file:{inp}"
            else:
                h.update(inp.encode("utf-8"))
                source = "string"
            
            return ToolResult(
                success=True,
                output=f"{algorithm}({source}) = {h.hexdigest()}",
                metadata={
                    "algorithm": algorithm,
                    "hash": h.hexdigest(),
                    "source": source,
                },
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)


class EncryptTool(Tool):
    """Encrypt/decrypt data với AES (requires cryptography lib)."""
    category = ToolCategory.CRYPTO
    safety = ToolSafety.DANGEROUS
    
    @property
    def name(self) -> str:
        return "encrypt"
    
    @property
    def description(self) -> str:
        return "Encrypt/decrypt data với AES-256-GCM. Requires 'cryptography' lib."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["encrypt", "decrypt"]},
                "data": {"type": "string"},
                "password": {"type": "string"},
            },
            "required": ["action", "data", "password"],
        }
    
    def execute(self, args: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
            from cryptography.hazmat.primitives import hashes
            import os as _os
        except ImportError:
            return ToolResult(
                success=False,
                error="cryptography not installed. Run: pip install cryptography",
                return_code=1,
            )
        
        action = args["action"]
        data = args["data"]
        password = args["password"]
        
        try:
            if action == "encrypt":
                salt = _os.urandom(16)
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = kdf.derive(password.encode())
                aesgcm = AESGCM(key)
                nonce = _os.urandom(12)
                ciphertext = aesgcm.encrypt(nonce, data.encode(), None)
                encrypted = base64.b64encode(salt + nonce + ciphertext).decode()
                return ToolResult(
                    success=True,
                    output=encrypted,
                    metadata={"action": "encrypt", "algorithm": "AES-256-GCM"},
                )
            else:  # decrypt
                raw = base64.b64decode(data)
                salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = kdf.derive(password.encode())
                aesgcm = AESGCM(key)
                plaintext = aesgcm.decrypt(nonce, ciphertext, None).decode()
                return ToolResult(
                    success=True,
                    output=plaintext,
                    metadata={"action": "decrypt"},
                )
        except Exception as e:
            return ToolResult(success=False, error=str(e), return_code=1)
