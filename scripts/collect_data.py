"""
Script thu thập training data từ GitHub + HuggingFace
=====================================================
Chạy script này để collect training data cho Nexus Coder v0.2.

Sources:
- GitHub repos (curated list trong nexus.data.collectors.github_collector.CURATED_REPOS)
- HuggingFace datasets (curated list trong nexus.data.collectors.huggingface_collector.CURATED_DATASETS)
- arXiv papers (curated queries)
- Wikipedia (Vietnamese + English)
- StackOverflow Q&A

Usage:
    python scripts/collect_data.py --source github --max-repos 10
    python scripts/collect_data.py --source huggingface --max-datasets 5
    python scripts/collect_data.py --source all --output ./data/raw
"""
import sys
import os
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def collect_github(output_dir: str, max_repos: int = 10, token: str = None):
    """Collect code từ GitHub repos."""
    from nexus.data.collectors.github_collector import GitHubCollector, CURATED_REPOS
    
    collector = GitHubCollector(token=token, cache_dir=os.path.join(output_dir, "github_cache"))
    repos = CURATED_REPOS[:max_repos]
    
    logger.info(f"Collecting from {len(repos)} GitHub repos...")
    
    output_file = os.path.join(output_dir, "github_code.jsonl")
    count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collector.collect(repos):
            entry = {
                "text": sample.content,
                "source": f"github:{sample.repo}",
                "language": sample.language,
                "metadata": {
                    "file_path": sample.file_path,
                    "size": sample.size,
                    "quality_score": sample.quality_score,
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 100 == 0:
                logger.info(f"  Collected {count} samples...")
    
    logger.info(f"✓ GitHub: {count} samples → {output_file}")
    return count


def collect_huggingface(output_dir: str, max_datasets: int = 5, token: str = None):
    """Collect từ HuggingFace datasets."""
    from nexus.data.collectors.huggingface_collector import HuggingFaceCollector, CURATED_DATASETS
    
    collector = HuggingFaceCollector(cache_dir=os.path.join(output_dir, "hf_cache"), token=token)
    datasets = CURATED_DATASETS[:max_datasets]
    
    logger.info(f"Collecting from {len(datasets)} HuggingFace datasets...")
    
    output_file = os.path.join(output_dir, "hf_data.jsonl")
    count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collector.collect(datasets):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 1000 == 0:
                logger.info(f"  Collected {count} samples...")
    
    logger.info(f"✓ HuggingFace: {count} samples → {output_file}")
    return count


def collect_arxiv(output_dir: str, max_queries: int = 5):
    """Collect papers từ arXiv."""
    from nexus.data.collectors.arxiv_collector import ArxivCollector, CURATED_QUERIES
    
    collector = ArxivCollector()
    queries = CURATED_QUERIES[:max_queries]
    
    logger.info(f"Collecting arXiv papers ({len(queries)} queries)...")
    
    output_file = os.path.join(output_dir, "arxiv_papers.jsonl")
    count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collector.collect(queries, max_per_query=20):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    
    logger.info(f"✓ arXiv: {count} samples → {output_file}")
    return count


def collect_wikipedia(output_dir: str, language: str = "vi"):
    """Collect articles từ Wikipedia."""
    from nexus.data.collectors.wikipedia_collector import WikipediaCollector
    
    collector = WikipediaCollector(language=language)
    
    logger.info(f"Collecting Wikipedia ({language}) articles...")
    
    output_file = os.path.join(output_dir, f"wikipedia_{language}.jsonl")
    count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collector.collect():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    
    logger.info(f"✓ Wikipedia ({language}): {count} samples → {output_file}")
    return count


def collect_stackoverflow(output_dir: str, max_tags: int = 5, token: str = None):
    """Collect Q&A từ StackOverflow."""
    from nexus.data.collectors.stackoverflow_collector import StackOverflowCollector, CURATED_TAGS
    
    collector = StackOverflowCollector(key=token)
    tags = CURATED_TAGS[:max_tags]
    
    logger.info(f"Collecting StackOverflow Q&A ({len(tags)} tags)...")
    
    output_file = os.path.join(output_dir, "stackoverflow.jsonl")
    count = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in collector.collect(tags, max_per_tag=50):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
    
    logger.info(f"✓ StackOverflow: {count} samples → {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Nexus Coder Data Collector")
    parser.add_argument(
        "--source",
        choices=["github", "huggingface", "arxiv", "wikipedia", "stackoverflow", "all"],
        default="all",
        help="Data source to collect from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw",
        help="Output directory",
    )
    parser.add_argument("--max-repos", type=int, default=10, help="Max GitHub repos")
    parser.add_argument("--max-datasets", type=int, default=5, help="Max HF datasets")
    parser.add_argument("--max-queries", type=int, default=5, help="Max arXiv queries")
    parser.add_argument("--max-tags", type=int, default=5, help="Max SO tags")
    parser.add_argument("--language", type=str, default="vi", help="Wikipedia language")
    parser.add_argument("--github-token", type=str, default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  NEXUS CODER v0.2 - DATA COLLECTOR")
    print("  Tác giả: Hieu Louis")
    print("=" * 70)
    
    os.makedirs(args.output, exist_ok=True)
    
    total = 0
    
    if args.source in ("github", "all"):
        total += collect_github(args.output, args.max_repos, args.github_token)
    
    if args.source in ("huggingface", "all"):
        total += collect_huggingface(args.output, args.max_datasets, args.hf_token)
    
    if args.source in ("arxiv", "all"):
        total += collect_arxiv(args.output, args.max_queries)
    
    if args.source in ("wikipedia", "all"):
        total += collect_wikipedia(args.output, args.language)
    
    if args.source in ("stackoverflow", "all"):
        total += collect_stackoverflow(args.output, args.max_tags)
    
    print(f"\n{'=' * 70}")
    print(f"  ✅ Total collected: {total} samples")
    print(f"  📁 Output: {args.output}")
    print(f"{'=' * 70}")
    print(f"\nNext step: Run scripts/prepare_dataset.py to process the raw data.")


if __name__ == "__main__":
    main()
