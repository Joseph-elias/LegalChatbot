from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_search import get_embedding_cache_path, load_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute and persist SBERT corpus embeddings.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration even if cache exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = get_embedding_cache_path()
    ids, _, emb = load_embeddings(rebuild=args.rebuild)
    print(f"Embedding cache ready: {cache_path}")
    print(f"Rows: {len(ids)} | Shape: {tuple(emb.shape)}")


if __name__ == "__main__":
    main()
