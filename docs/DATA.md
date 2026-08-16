# Data Pipeline Documentation

Nexus Coder v0.2 có pipeline thu thập và xử lý training data hoàn chỉnh.

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  COLLECT    │ ──> │   PROCESS    │ ──> │   TRAIN     │ ──> │    EVALUATE  │
│ (5 sources) │     │ (4 stages)   │     │ (curriculum)│     │  (8 benches) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

## Sources (Collectors)

### 1. GitHub
- **60+ curated repos** (Python, JS, TS, Go, Rust, C, C++)
- Categories: Python core, Data science, ML/DL, Web, CLI, Async, Database, Tools
- Quality filter: size, content, auto-generated detection
- File extensions: .py, .js, .ts, .go, .rs, .java, .c, .cpp, .sql, .sh, .md

### 2. HuggingFace
- **20+ curated datasets**:
  - Code: codeparrot, the-stack, CodeAlpaca
  - Text: Wikipedia (vi, en), C4, OSCAR
  - Chat: UltraChat, OpenOrca, OpenHermes, Dolly
  - Math: MetaMathQA, GSM8K, MATH
  - Vietnamese: news_corpus, PhoATC

### 3. arXiv
- 20 curated queries (transformer, MoE, LLM, code generation, etc.)
- Categories: cs.CL, cs.LG, cs.AI, cs.SE, cs.PL, cs.CV, stat.ML
- Rate limit: 1 request per 3 seconds

### 4. Wikipedia
- Vietnamese + English
- 20 curated topics per language
- Random article collection supported

### 5. StackOverflow
- 30 curated tags (python, javascript, java, etc.)
- Filter by minimum score (default: 5)
- Includes accepted answers
- Rate limit: 30 req/s

## Processing Pipeline

### Stage 1: Clean (TextCleaner)
- HTML tag removal
- Unicode normalization (NFC)
- Control character removal
- HTML entity decoding
- Whitespace normalization
- Encoding fix

### Stage 2: Format (CodeFormatter)
- Language detection (by extension + patterns)
- Trailing whitespace removal
- Excessive blank line removal (max 2 consecutive)
- Leading/trailing blank line removal
- Markdown fence wrapping

### Stage 3: Quality Filter (QualityFilter)
- Length check (50-100,000 chars)
- Word count (min 10)
- Unique word ratio (min 0.3)
- Repetition score (max 0.5)
- Spam pattern detection
- Code presence bonus

### Stage 4: Deduplicate (Deduplicator)
- Exact hash dedup (MD5)
- MinHash LSH for near-duplicates
- 128 permutations, 5-gram
- Jaccard threshold: 0.8

## Curriculum Learning

4-stage curriculum:

| Stage | Difficulty | Length | Quality | Description |
|-------|-----------|--------|---------|-------------|
| 1 | EASY | 50-500 | ≥0.7 | Short basic text - vocabulary |
| 2 | MEDIUM | 500-5000 | ≥0.6 | Standard length - grammar |
| 3 | HARD | 5000-30000 | ≥0.7 | Long technical - deep understanding |
| 4 | EXPERT | 30000-100000 | ≥0.8 | Multi-step reasoning |

## Usage

### Collect raw data

```bash
# Collect from all sources
python scripts/collect_data.py --source all --output ./data/raw

# Or specific source
python scripts/collect_data.py --source github --max-repos 10
python scripts/collect_data.py --source huggingface --max-datasets 5
```

### Process raw data

```bash
python scripts/prepare_dataset.py --input ./data/raw --output ./data/processed
```

### Train with external data

```bash
python scripts/train.py --config large --include-external --steps 5000
```

## Output Format

Processed data saved as JSONL files by difficulty:

```
data/processed/
├── train_easy.jsonl      # Stage 1 samples
├── train_medium.jsonl    # Stage 2 samples
├── train_hard.jsonl      # Stage 3 samples
├── train_expert.jsonl    # Stage 4 samples
└── processing_stats.json # Statistics
```

Each JSONL line:
```json
{
  "text": "...",
  "source": "github:python/cpython",
  "language": "python",
  "metadata": {
    "file_path": "Lib/os.py",
    "size": 45678,
    "quality_score": 0.85,
    "quality": {"score": 0.85, "length": 45678, "word_count": 1200, "has_code": true},
    "cleaned": true,
    "cleaned_length": 45678,
    "formatted": true,
    "detected_language": "python"
  }
}
```

## Environment Variables

```bash
# GitHub API (for search)
export GITHUB_TOKEN=ghp_xxx

# HuggingFace Hub (for gated datasets)
export HF_TOKEN=hf_xxx

# Web search API (optional)
export SEARCH_API_KEY=xxx
export BRAVE_SEARCH_API_KEY=xxx
```

## Estimate Data Volume

| Source | Estimated samples | Estimated size |
|--------|------------------|----------------|
| GitHub (60 repos) | ~50,000 files | ~500 MB |
| HuggingFace (20 datasets) | ~200,000 samples | ~2 GB (streamed) |
| arXiv (20 queries) | ~400 papers | ~50 MB |
| Wikipedia (vi+en) | ~40 articles | ~5 MB |
| StackOverflow (30 tags) | ~1,500 Q&A | ~10 MB |
| **Total** | **~250,000 samples** | **~2.5 GB** |

After deduplication and quality filter: ~150,000 high-quality samples.

## Custom Sources

Add your own collector:

```python
from nexus.data.collectors.base import Collector

class MyCollector(Collector):
    def collect(self):
        # Yield samples as dicts
        yield {
            "text": "...",
            "source": "my_source",
            "language": "en",
            "metadata": {...},
        }
```
