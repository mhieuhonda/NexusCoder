"""Evaluation Metrics - Perplexity, BLEU, ROUGE, F1."""
from __future__ import annotations

import math
from typing import List, Dict, Any, Optional
from collections import Counter


def compute_perplexity(
    model,
    input_ids,
    labels=None,
) -> float:
    """Compute perplexity trên input.
    
    Args:
        model: NexusCoderForCausalLM
        input_ids: [B, T] token ids
        labels: Optional labels (defaults to input_ids)
    
    Returns:
        Perplexity (lower is better)
    """
    import torch
    
    if labels is None:
        labels = input_ids.clone()
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
    
    return math.exp(loss.item())


def compute_bleu(
    references: List[str],
    hypothesis: str,
    max_n: int = 4,
) -> Dict[str, float]:
    """Compute BLEU score (simplified).
    
    Args:
        references: List of reference translations
        hypothesis: Generated translation
        max_n: Maximum n-gram (BLEU-4 default)
    
    Returns:
        Dict with 'bleu', 'brevity_penalty', and per-ngram precision
    """
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    hyp_tokens = hypothesis.lower().split()
    
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        if not hyp_ngrams:
            precisions.append(0)
            continue
        
        # Count matches against any reference
        matches = 0
        total = sum(hyp_ngrams.values())
        
        for ref in references:
            ref_tokens = ref.lower().split()
            ref_ngrams = get_ngrams(ref_tokens, n)
            for ngram, count in hyp_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
        
        precisions.append(matches / total if total > 0 else 0)
    
    # Brevity penalty
    ref_lens = [len(r.split()) for r in references]
    # v0.4 fix: guard against empty references list
    if not ref_lens:
        result = {"bleu": 0.0, "brevity_penalty": 0.0}
        for i in range(1, max_n + 1):
            result[f"precision_{i}"] = 0.0
        return result
    closest_ref_len = min(ref_lens, key=lambda l: abs(l - len(hyp_tokens)))
    bp = 1.0 if len(hyp_tokens) > closest_ref_len else math.exp(1 - closest_ref_len / max(len(hyp_tokens), 1))
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geo_mean = 0.0
    
    bleu = bp * geo_mean
    
    result = {"bleu": bleu, "brevity_penalty": bp}
    for i, p in enumerate(precisions, 1):
        result[f"precision_{i}"] = p
    return result


def compute_rouge(
    reference: str,
    hypothesis: str,
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores (simplified)."""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    # ROUGE-1 (unigram) — v0.4 fix: recall (÷ ref length), not precision (÷ hyp)
    ref_1 = get_ngrams(ref_tokens, 1)
    hyp_1 = get_ngrams(hyp_tokens, 1)
    overlap_1 = sum((ref_1 & hyp_1).values())
    rouge_1_recall = overlap_1 / max(len(ref_tokens), 1)
    rouge_1_precision = overlap_1 / max(len(hyp_tokens), 1)
    rouge_1 = (
        2 * rouge_1_recall * rouge_1_precision / max(rouge_1_recall + rouge_1_precision, 1e-9)
        if (rouge_1_recall + rouge_1_precision) > 0
        else 0.0
    )

    # ROUGE-2 (bigram)
    ref_2 = get_ngrams(ref_tokens, 2)
    hyp_2 = get_ngrams(hyp_tokens, 2)
    overlap_2 = sum((ref_2 & hyp_2).values())
    rouge_2_recall = overlap_2 / max(sum(ref_2.values()), 1)
    rouge_2_precision = overlap_2 / max(sum(hyp_2.values()), 1)
    rouge_2 = (
        2 * rouge_2_recall * rouge_2_precision / max(rouge_2_recall + rouge_2_precision, 1e-9)
        if (rouge_2_recall + rouge_2_precision) > 0
        else 0.0
    )
    
    # ROUGE-L (LCS)
    def lcs_length(a: List, b: List) -> int:
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs = lcs_length(ref_tokens, hyp_tokens)
    rouge_l = lcs / max(len(ref_tokens), 1)
    
    return {
        "rouge_1": rouge_1,
        "rouge_2": rouge_2,
        "rouge_l": rouge_l,
    }


def compute_f1(
    predicted: List[str],
    gold: List[str],
) -> Dict[str, float]:
    """Compute F1, precision, recall (token-level)."""
    pred_set = set(predicted)
    gold_set = set(gold)
    
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {"precision": precision, "recall": recall, "f1": f1}
