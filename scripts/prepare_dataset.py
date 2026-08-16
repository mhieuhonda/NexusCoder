"""
Script chuẩn bị dataset cho training
====================================
Process raw collected data → cleaned, deduplicated, formatted training data.

Steps:
1. Load raw data from ./data/raw/
2. Clean text (TextCleaner)
3. Format code samples (CodeFormatter)
4. Filter by quality (QualityFilter)
5. Deduplicate (Deduplicator)
6. Save processed data to ./data/processed/

Usage:
    python scripts/prepare_dataset.py --input ./data/raw --output ./data/processed
"""
import sys
import os
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(input_dir: str):
    """Load all JSONL files from input directory."""
    files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".jsonl")
    ]
    
    total = 0
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    yield item
                    count += 1
                except json.JSONDecodeError:
                    continue
        logger.info(f"  Loaded {count} from {fname}")
        total += count
    
    logger.info(f"Total raw samples: {total}")


def process_data(input_dir: str, output_dir: str, max_samples: int = None):
    """Process raw data through cleaning, dedup, quality filter."""
    from nexus.data.processors.cleaner import TextCleaner
    from nexus.data.processors.quality_filter import QualityFilter
    from nexus.data.processors.code_formatter import CodeFormatter
    from nexus.data.processors.deduplicator import Deduplicator
    from nexus.data.curriculum import CurriculumLearning
    
    cleaner = TextCleaner()
    quality_filter = QualityFilter()
    code_formatter = CodeFormatter()
    deduplicator = Deduplicator()
    curriculum = CurriculumLearning()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files by difficulty
    output_files = {
        "easy": open(os.path.join(output_dir, "train_easy.jsonl"), "w", encoding="utf-8"),
        "medium": open(os.path.join(output_dir, "train_medium.jsonl"), "w", encoding="utf-8"),
        "hard": open(os.path.join(output_dir, "train_hard.jsonl"), "w", encoding="utf-8"),
        "expert": open(os.path.join(output_dir, "train_expert.jsonl"), "w", encoding="utf-8"),
    }
    
    stats = {
        "total_input": 0,
        "cleaned": 0,
        "quality_passed": 0,
        "deduplicated": 0,
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0, "expert": 0},
    }
    
    logger.info("Processing samples...")
    
    for sample in load_raw_data(input_dir):
        if max_samples and stats["total_input"] >= max_samples:
            break
        
        stats["total_input"] += 1
        
        # Step 1: Clean
        sample = cleaner.process(sample)
        if sample is None:
            continue
        stats["cleaned"] += 1
        
        # Step 2: Format code
        sample = code_formatter.process(sample)
        
        # Step 3: Quality filter
        if not quality_filter.filter(sample):
            continue
        sample = next(quality_filter.process([sample]), None)
        if sample is None:
            continue
        stats["quality_passed"] += 1
        
        # Step 4: Dedup
        if deduplicator.is_duplicate(sample.get("text", "")):
            continue
        deduplicator.add(sample["text"], sample)
        stats["deduplicated"] += 1
        
        # Step 5: Classify by difficulty
        difficulty = curriculum.classify_sample(sample).value
        output_files[difficulty].write(json.dumps(sample, ensure_ascii=False) + "\n")
        stats["by_difficulty"][difficulty] += 1
        
        if stats["deduplicated"] % 1000 == 0:
            logger.info(f"  Processed {stats['deduplicated']} unique samples...")
    
    # Close files
    for f in output_files.values():
        f.close()
    
    # Print stats
    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Input samples:    {stats['total_input']:,}")
    print(f"  After cleaning:   {stats['cleaned']:,}")
    print(f"  Quality passed:   {stats['quality_passed']:,}")
    print(f"  After dedup:      {stats['deduplicated']:,}")
    print("-" * 60)
    print("  By difficulty:")
    for level, count in stats["by_difficulty"].items():
        print(f"    {level:8s}: {count:,}")
    print("-" * 60)
    print(f"  Output dir: {output_dir}")
    print("=" * 60)
    
    # Save stats
    stats_path = os.path.join(output_dir, "processing_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Nexus Coder Dataset Processor")
    parser.add_argument("--input", type=str, default="./data/raw")
    parser.add_argument("--output", type=str, default="./data/processed")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NEXUS CODER v0.2 - DATASET PROCESSOR")
    print("  Tác giả: Hieu Louis")
    print("=" * 70)
    
    if not os.path.exists(args.input):
        print(f"\n❌ Input dir not found: {args.input}")
        print("Run scripts/collect_data.py first to collect raw data.")
        return 1
    
    process_data(args.input, args.output, args.max_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
