"""Deduplicator - Loại bỏ duplicate samples bằng MinHash."""
from __future__ import annotations

import re
import hashlib
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple, Iterator
from dataclasses import dataclass, field


@dataclass
class DeduplicationConfig:
    """Config cho Deduplicator."""
    ngram_size: int = 5  # Word n-grams
    num_perm: int = 128  # Number of permutations (MinHash)
    similarity_threshold: float = 0.8  # Jaccard threshold
    hash_size: int = 2**21  # Hash space size
    exact_match_first: bool = True  # Quick exact hash dedup first


class MinHash:
    """Simple MinHash implementation."""
    
    def __init__(self, num_perm: int = 128, seed: int = 42):
        import random
        self.num_perm = num_perm
        rng = random.Random(seed)
        # Generate random hash functions: h(x) = (a*x + b) mod p
        self.p = (1 << 61) - 1  # Mersenne prime
        self.a = [rng.randint(1, self.p - 1) for _ in range(num_perm)]
        self.b = [rng.randint(0, self.p - 1) for _ in range(num_perm)]
        self._min_hashes = [self.p] * num_perm
    
    def update(self, token: str):
        """Update with a token."""
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(self.num_perm):
            val = (self.a[i] * h + self.b[i]) % self.p
            if val < self._min_hashes[i]:
                self._min_hashes[i] = val
    
    def update_batch(self, tokens: List[str]):
        for t in tokens:
            self.update(t)
    
    def signature(self) -> List[int]:
        return list(self._min_hashes)
    
    def jaccard(self, other: "MinHash") -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("Different num_perm")
        if not self._min_hashes or not other._min_hashes:
            return 0.0
        matches = sum(1 for a, b in zip(self._min_hashes, other._min_hashes) if a == b)
        return matches / self.num_perm


class Deduplicator:
    """Loại bỏ duplicate samples.
    
    Uses:
    1. Exact hash dedup (fast, MD5 of full text)
    2. MinHash LSH (fuzzy, near-duplicate detection)
    
    Usage:
        dedup = Deduplicator()
        unique_samples = list(dedup.process(samples_iter))
    """
    
    def __init__(self, config: DeduplicationConfig = None):
        self.config = config or DeduplicationConfig()
        self._seen_hashes: Set[str] = set()
        self._buckets: Dict[int, List[Tuple[MinHash, int]]] = defaultdict(list)
        self._samples: List[Dict[str, Any]] = []
    
    def _get_ngrams(self, text: str, n: int = 5) -> List[str]:
        """Get word n-grams."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < n:
            return [" ".join(words)]
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def _exact_hash(self, text: str) -> str:
        """Quick exact hash."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()
    
    def _minhash(self, text: str) -> MinHash:
        """Compute MinHash of text."""
        mh = MinHash(num_perm=self.config.num_perm)
        mh.update_batch(self._get_ngrams(text, self.config.ngram_size))
        return mh
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is duplicate of seen samples."""
        # Quick exact check first
        if self.config.exact_match_first:
            h = self._exact_hash(text)
            if h in self._seen_hashes:
                return True
        
        # MinHash check
        mh = self._minhash(text)
        sig = mh.signature()
        
        # Check LSH buckets
        for band_start in range(0, self.config.num_perm, 16):
            band = tuple(sig[band_start:band_start+16])
            band_hash = hash(band) % 1000
            
            if band_hash in self._buckets:
                for existing_mh, _ in self._buckets[band_hash]:
                    if mh.jaccard(existing_mh) >= self.config.similarity_threshold:
                        return True
        
        return False
    
    def add(self, text: str, sample: Dict[str, Any] = None):
        """Add a text/sample to the deduplicator."""
        if self.config.exact_match_first:
            h = self._exact_hash(text)
            self._seen_hashes.add(h)
        
        mh = self._minhash(text)
        idx = len(self._samples)
        self._samples.append(sample or {"text": text})
        
        # Add to LSH buckets
        sig = mh.signature()
        for band_start in range(0, self.config.num_perm, 16):
            band = tuple(sig[band_start:band_start+16])
            band_hash = hash(band) % 1000
            self._buckets[band_hash].append((mh, idx))
    
    def process(self, samples: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Filter an iterator of samples, yielding only unique ones."""
        seen = 0
        deduped = 0
        
        for sample in samples:
            seen += 1
            text = sample.get("text", "")
            
            if self.is_duplicate(text):
                deduped += 1
                continue
            
            self.add(text, sample)
            yield sample
        
        if seen > 0:
            from ...utils.logging import get_logger
            logger = get_logger()
            logger.info(f"Dedup: {seen} → {seen - deduped} (removed {deduped})")
    
    def stats(self) -> Dict[str, int]:
        """Get deduplication stats."""
        return {
            "total_added": len(self._samples),
            "exact_hashes": len(self._seen_hashes),
            "buckets": len(self._buckets),
        }
