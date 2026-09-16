"""Retrieval Index and Query module for Hiver Customer Support Agent.

Extracts historical resolved customer support threads, indexes initial customer queries
in a FAISS inner-product (cosine similarity) vector index, and retrieves top-k
grounded brand resolutions with similarity threshold filtering and full traceability.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
import yaml
from sklearn.preprocessing import normalize

from src.intents import embed_messages, get_embedding_model

logger = logging.getLogger(__name__)


def build_resolution_corpus(
    threads: List[Dict[str, Any]],
    classifier_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    excluded_tweet_ids: Optional[Set[str]] = None,
    max_items: int = 10000,
) -> List[Dict[str, Any]]:
    """Construct paired customer queries and brand resolutions from resolved threads.

    Per thread:
    - customer_message: First customer turn text.
    - brand_resolution: All brand turns concatenated together.
    - intent: Intent classification if classifier_fn provided, else 'general_inquiry'.
    Drops threads where brand_resolution is shorter than 20 characters (pure redirects
    with nothing useful to ground a reply on).
    Excludes golden evaluation tweet IDs to eliminate data leakage.

    Args:
        threads: List of thread dictionaries.
        classifier_fn: Optional intent classification function.
        excluded_tweet_ids: Optional set of tweet IDs to strictly exclude (leak prevention).
        max_items: Maximum number of corpus records to index.

    Returns:
        List of corpus item dictionaries.
    """
    logger.info("Building resolution corpus from %d threads...", len(threads))
    corpus: List[Dict[str, Any]] = []
    dropped_short_count = 0
    excluded = set(excluded_tweet_ids or [])

    for thread in threads:
        turns = thread.get("turns", [])
        cust_turns = [t for t in turns if t["role"] == "customer" and t.get("text")]
        brand_turns = [t["text"] for t in turns if t["role"] == "brand" and t.get("text")]

        if not cust_turns or not brand_turns:
            continue

        first_turn = cust_turns[0]
        tid = str(first_turn.get("tweet_id", ""))
        if tid in excluded:
            continue

        customer_message = first_turn["text"]
        brand_resolution = " ".join(brand_turns).strip()

        # Quality threshold: Drop resolutions under 20 characters
        if len(brand_resolution) < 20:
            dropped_short_count += 1
            continue

        corpus.append({
            "thread_id": thread["thread_id"],
            "tweet_id": tid,
            "customer_message": customer_message,
            "brand_resolution": brand_resolution,
            "intent": "general_inquiry",
            "created_at": thread.get("created_at", ""),
        })

        if len(corpus) >= max_items:
            break

    logger.info(
        "Built resolution corpus with %d threads (dropped %d short/uninformative resolutions)",
        len(corpus),
        dropped_short_count,
    )
    return corpus


def build_index(
    corpus: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[faiss.IndexFlatIP, Path, Path]:
    """Embed corpus customer queries and build a FAISS inner-product (cosine) index.

    Saves the FAISS index and metadata JSON to disk.

    Args:
        corpus: Resolution corpus items.
        cfg: Configuration dictionary.

    Returns:
        Tuple of (index, index_path, meta_path).
    """
    index_base = Path(cfg["data"]["faiss_index"])
    cache_dir = Path(cfg["data"].get("cache_dir", ".cache"))
    embed_model_name = cfg["model"].get("embedding_model", "all-MiniLM-L6-v2")

    index_path = Path(f"{index_base}.index")
    meta_path = Path(f"{index_base}_meta.json")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    queries = [item["customer_message"] for item in corpus]
    logger.info("Computing embeddings for %d corpus queries...", len(queries))
    embeddings = embed_messages(queries, model_name=embed_model_name, cache_dir=cache_dir)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP (cosine via normalized vectors) with dim=%d...", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(index_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)

    logger.info("Saved FAISS index to %s and metadata to %s", index_path, meta_path)
    return index, index_path, meta_path


def load_index(cfg: Dict[str, Any]) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
    """Load FAISS index and metadata JSON from disk.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Tuple of (FAISS index, corpus_metadata list).
    """
    index_base = Path(cfg["data"]["faiss_index"])
    index_path = Path(f"{index_base}.index")
    meta_path = Path(f"{index_base}_meta.json")

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"FAISS index files not found at {index_path} / {meta_path}")

    logger.info("Loading FAISS index from %s", index_path)
    index = faiss.read_index(str(index_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        corpus_meta = json.load(f)

    logger.info("Loaded index with %d vectors and %d metadata records", index.ntotal, len(corpus_meta))
    return index, corpus_meta


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    corpus_meta: List[Dict[str, Any]],
    top_k: int = 3,
    min_similarity: float = 0.40,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """Retrieve top-k historically resolved threads similar to query.

    L2-normalizes the query embedding, computes inner products (cosine similarity),
    and filters results by min_similarity. Returns empty list if no matches meet
    the threshold (triggering NO_GROUNDING in routing).

    Logs retrieved thread_ids at INFO level for complete traceability.

    Args:
        query: Customer message query string.
        index: Loaded FAISS inner product index.
        corpus_meta: Parallel metadata records.
        top_k: Maximum number of neighbors to return.
        min_similarity: Minimum cosine similarity threshold.
        model_name: SentenceTransformers model name.

    Returns:
        List of retrieved match dictionaries with similarity_score.
    """
    model = get_embedding_model(model_name)
    raw_vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    norm_vec = normalize(raw_vec, norm="l2").astype(np.float32)

    k_to_search = min(top_k, index.ntotal)
    if k_to_search <= 0:
        logger.warning("Empty index; retrieval returned 0 results.")
        return []

    distances, indices = index.search(norm_vec, k_to_search)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(corpus_meta):
            continue
        sim = float(score)
        if sim >= min_similarity:
            meta = corpus_meta[idx]
            results.append({
                "thread_id": meta["thread_id"],
                "customer_message": meta["customer_message"],
                "brand_resolution": meta["brand_resolution"],
                "intent": meta.get("intent", "unknown"),
                "similarity_score": round(sim, 4),
            })

    retrieved_ids = [r["thread_id"] for r in results]
    logger.info(
        "Retrieved %d threads (IDs: %s) for query: '%s' (threshold=%.2f)",
        len(results),
        retrieved_ids,
        query[:50],
        min_similarity,
    )
    return results


def main() -> None:
    """CLI entrypoint for building FAISS index."""
    parser = argparse.ArgumentParser(description="Build FAISS resolution index.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    threads_path = Path(cfg["data"]["threads_jsonl"])
    golden_path = Path(cfg["data"]["golden_jsonl"])
    excluded_ids: Set[str] = set()
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if "tweet_id" in rec:
                    excluded_ids.add(str(rec["tweet_id"]))
        logger.info("Excluding %d golden eval tweet IDs from retrieval corpus", len(excluded_ids))

    threads = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            threads.append(json.loads(line))

    corpus = build_resolution_corpus(threads, excluded_tweet_ids=excluded_ids)
    build_index(corpus, cfg)


if __name__ == "__main__":
    main()
