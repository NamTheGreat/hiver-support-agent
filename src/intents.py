"""Intent Taxonomy and Classification module for Hiver Customer Support Agent.

Empirically discovers customer support intent taxonomy from tweet embeddings
using HDBSCAN with silhouette-optimized KMeans fallback, extracts central exemplars,
synthesizes cluster definitions via LLM, and provides calibrated intent classification.
"""

import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hdbscan
import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from src.llm import generate

logger = logging.getLogger(__name__)

# Global model cache to avoid re-loading sentence transformer weights
_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load and cache SentenceTransformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model: %s", model_name)
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


def embed_messages(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    cache_dir: Path = Path(".cache"),
) -> np.ndarray:
    """Compute and cache message embeddings using sentence-transformers.

    Caches embeddings to disk indexed by the SHA-256 hash of the concatenated texts
    and model name. Skips re-embedding on cache hit.

    Args:
        texts: List of message strings to embed.
        model_name: SentenceTransformers model identifier.
        cache_dir: Directory path for disk caching.

    Returns:
        L2-normalized 2D numpy array of shape (N, D).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute content hash for the text batch
    hasher = hashlib.sha256()
    hasher.update(model_name.encode("utf-8"))
    for t in texts:
        hasher.update(t.encode("utf-8"))
    cache_key = hasher.hexdigest()
    cache_file = cache_dir / f"embeddings_{cache_key[:16]}_{len(texts)}.npy"

    if cache_file.exists():
        logger.info("Loading %d cached embeddings from %s", len(texts), cache_file)
        try:
            embeddings = np.load(cache_file)
            if len(embeddings) == len(texts):
                return embeddings
        except Exception as exc:
            logger.warning("Cache read failed: %s; recomputing.", exc)

    logger.info("Embedding %d messages with %s...", len(texts), model_name)
    model = get_embedding_model(model_name)
    raw_embeddings = model.encode(texts, show_progress_bar=False, batch_size=64, convert_to_numpy=True)
    embeddings = normalize(raw_embeddings, norm="l2")

    np.save(cache_file, embeddings)
    logger.info("Cached %d embeddings to %s", len(embeddings), cache_file)
    return embeddings


def cluster_messages(
    embeddings: np.ndarray,
    n_clusters_range: Tuple[int, int] = (5, 8),
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    """Cluster embeddings using HDBSCAN first, with KMeans silhouette fallback.

    If HDBSCAN designates > 30% of data points as noise (-1), falls back to KMeans,
    searching n_clusters_range and picking the k that maximizes the silhouette score.

    Args:
        embeddings: Normalized 2D embedding array.
        n_clusters_range: Inclusive [min_k, max_k] search range for KMeans fallback.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (labels array, clustering_method_used string).
    """
    logger.info("Attempting primary clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=10, metric="euclidean")
    hdb_labels = clusterer.fit_predict(embeddings)

    noise_ratio = float((hdb_labels == -1).mean())
    unique_clusters = set(hdb_labels) - {-1}
    logger.info("HDBSCAN produced %d clusters with %.1f%% noise points", len(unique_clusters), noise_ratio * 100)

    if noise_ratio <= 0.30 and len(unique_clusters) >= n_clusters_range[0]:
        logger.info("HDBSCAN clustering accepted (noise <= 30%%).")
        return hdb_labels, "HDBSCAN"

    logger.warning(
        "HDBSCAN noise ratio (%.1f%%) exceeds 30%% or insufficient clusters; falling back to KMeans",
        noise_ratio * 100,
    )

    best_k = n_clusters_range[0]
    best_score = -1.0
    best_labels = None

    min_k, max_k = n_clusters_range
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        logger.info("KMeans k=%d -> silhouette score: %.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    logger.info("KMeans chosen: k=%d with silhouette score %.4f", best_k, best_score)
    return best_labels, f"KMeans(k={best_k}, silhouette={best_score:.4f})"


def propose_taxonomy(
    texts: List[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    llm_generate: Callable[[str], str] = generate,
) -> Dict[str, Dict[str, Any]]:
    """Derive semantic intent taxonomy from clusters and exemplars.

    For each cluster, extracts 10 texts closest to the centroid, prompts LLM
    for a snake_case name and one-line definition, merges clusters with
    centroid cosine similarity > 0.85, and appends an 'other' intent.

    Args:
        texts: Raw message strings.
        labels: Cluster label assignments.
        embeddings: Normalized embeddings corresponding to texts.
        llm_generate: LLM generation callable.

    Returns:
        Taxonomy dictionary: {intent_name: {definition, example_tweets: [3 tweets]}}.
    """
    logger.info("Proposing intent taxonomy from clusters...")
    unique_labels = sorted([lbl for lbl in set(labels) if lbl != -1])

    cluster_centroids: Dict[int, np.ndarray] = {}
    cluster_exemplars: Dict[int, List[str]] = {}

    for lbl in unique_labels:
        mask = labels == lbl
        c_embeds = embeddings[mask]
        c_texts = [t for t, m in zip(texts, mask) if m]

        # Centroid is mean normalized embedding
        centroid = normalize(c_embeds.mean(axis=0, keepdims=True), norm="l2")[0]
        cluster_centroids[lbl] = centroid

        # Find 10 texts closest to centroid via cosine similarity (dot product)
        sims = np.dot(c_embeds, centroid)
        top10_idx = np.argsort(-sims)[:10]
        cluster_exemplars[lbl] = [c_texts[i] for i in top10_idx]

    # Check for merging clusters with centroid cosine similarity > 0.85
    merged_labels: Dict[int, int] = {lbl: lbl for lbl in unique_labels}
    for i, lbl1 in enumerate(unique_labels):
        for lbl2 in unique_labels[i + 1:]:
            sim = float(np.dot(cluster_centroids[lbl1], cluster_centroids[lbl2]))
            if sim > 0.85:
                logger.info("Merging cluster %d into %d (centroid cosine similarity %.3f > 0.85)", lbl2, lbl1, sim)
                merged_labels[lbl2] = merged_labels[lbl1]

    # Group exemplars for merged clusters
    final_clusters: Dict[int, List[str]] = {}
    for lbl, parent in merged_labels.items():
        final_clusters.setdefault(parent, []).extend(cluster_exemplars[lbl])

    taxonomy: Dict[str, Dict[str, Any]] = {}

    for cluster_id, exemplars in final_clusters.items():
        sample_prompt = "\n".join(f"- {t}" for t in exemplars[:8])
        prompt = (
            "Analyze the following customer support tweets from Apple customers:\n"
            f"{sample_prompt}\n\n"
            "Provide a short snake_case intent name (e.g. battery_charging, software_update, connectivity_issue) "
            "and a clear one-line definition summarizing what the customer is experiencing.\n"
            "Output strict JSON with keys 'intent_name' and 'definition'. Do not include markdown formatting or backticks."
        )

        resp = llm_generate(prompt)
        # Parse JSON
        try:
            # Strip backticks if present
            clean_resp = re.sub(r"```json|```", "", resp).strip()
            parsed = json.loads(clean_resp)
            intent_name = parsed["intent_name"].strip().lower()
            definition = parsed["definition"].strip()
        except Exception as exc:
            logger.warning("Failed to parse LLM taxonomy response: %s; using fallback naming.", exc)
            intent_name = f"intent_cluster_{cluster_id}"
            definition = f"Issues related to customer inquiry cluster {cluster_id}."

        # Ensure snake_case
        intent_name = re.sub(r"[^a-z0-9_]", "_", intent_name.replace("-", "_").replace(" ", "_"))

        # Deduplicate intent name if already present
        base_name = intent_name
        counter = 1
        while intent_name in taxonomy:
            intent_name = f"{base_name}_{counter}"
            counter += 1

        # Select 3 distinct representative tweets
        example_tweets = exemplars[:3]
        taxonomy[intent_name] = {
            "definition": definition,
            "example_tweets": example_tweets,
        }
        logger.info("Added taxonomy intent: '%s' -> '%s'", intent_name, definition)

    # Always append 'other' intent
    taxonomy["other"] = {
        "definition": "Inquiries that are ambiguous, multi-intent, out-of-scope, or do not fit defined categories.",
        "example_tweets": [
            "What time does the store open?",
            "Can you tell Tim Cook I love the keynote?",
            "Hey",
        ],
    }
    logger.info("Taxonomy finalized with %d intents including 'other'", len(taxonomy))
    return taxonomy


def save_taxonomy(taxonomy: Dict[str, Dict[str, Any]], path: Path) -> None:
    """Save intent taxonomy JSON to disk.

    Args:
        taxonomy: Taxonomy dictionary.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    logger.info("Saved intent taxonomy to %s", path)


def load_taxonomy(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load intent taxonomy JSON from disk.

    Args:
        path: Path to taxonomy JSON file.

    Returns:
        Taxonomy dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_intent(
    message: str,
    taxonomy: Dict[str, Dict[str, Any]],
    llm_generate: Callable[[str], str] = generate,
) -> Dict[str, Any]:
    """Classify a customer message into an intent from the taxonomy with calibrated confidence.

    Confidence calibration:
    - 0.90+ : unambiguous match
    - 0.60 - 0.89 : plausible match
    - < 0.60 : ambiguous / multi-intent -> falls back to 'other' if needed

    Args:
        message: Cleaned customer message.
        taxonomy: Intent taxonomy dictionary.
        llm_generate: LLM generation callable.

    Returns:
        Dict with keys 'intent', 'confidence', 'reasoning'.
    """
    tax_prompt_lines = []
    for name, data in taxonomy.items():
        exs = "; ".join(f'"{ex}"' for ex in data.get("example_tweets", [])[:2])
        tax_prompt_lines.append(f"- {name}: {data['definition']} (Examples: {exs})")
    tax_prompt = "\n".join(tax_prompt_lines)

    system_prompt = (
        "You are an expert customer support intent classifier for Apple Support on Twitter.\n"
        "Given a customer message, classify it into exactly ONE of the available taxonomy intents below.\n"
        "Return strict JSON with keys 'intent', 'confidence' (float between 0.0 and 1.0), and 'reasoning'.\n"
        "Calibration guidelines:\n"
        "- 0.90+ for clear, unambiguous intent matching definition.\n"
        "- 0.60-0.89 for plausible but contestable interpretations.\n"
        "- <0.60 for ambiguous or multi-intent messages.\n"
        "- If nothing fits, classify as 'other'.\n"
        "Output ONLY raw JSON. No markdown backticks, no preamble."
    )

    user_prompt = (
        f"Available Intents:\n{tax_prompt}\n\n"
        f"Customer Message: \"{message}\"\n\n"
        "JSON output:"
    )

    resp = llm_generate(user_prompt, system_prompt=system_prompt)

    try:
        clean_resp = re.sub(r"```json|```", "", resp).strip()
        parsed = json.loads(clean_resp)
        intent = parsed.get("intent", "other").strip().lower()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning = str(parsed.get("reasoning", ""))

        # Guardrails: Ensure intent is in taxonomy
        if intent not in taxonomy:
            # Fuzzy match or fallback to other
            matching = [k for k in taxonomy if k in intent]
            intent = matching[0] if matching else "other"
            confidence = min(confidence, 0.60)

        # Cap confidence in [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    except Exception as exc:
        logger.warning("Intent classification parse failed for '%s': %s", message[:40], exc)
        return {
            "intent": "other",
            "confidence": 0.40,
            "reasoning": "Failed to parse classification output; defaulted to 'other'.",
        }


def batch_classify(
    messages: List[str],
    taxonomy: Dict[str, Dict[str, Any]],
    llm_generate: Callable[[str], str] = generate,
    delay: float = 0.02,
) -> List[Dict[str, Any]]:
    """Classify a batch of messages with a slight throttle to respect rate limits.

    Args:
        messages: List of customer message strings.
        taxonomy: Intent taxonomy dictionary.
        llm_generate: LLM generation callable.
        delay: Seconds to sleep between calls.

    Returns:
        List of classification dictionaries.
    """
    results = []
    for msg in messages:
        res = classify_intent(msg, taxonomy, llm_generate=llm_generate)
        results.append(res)
        if delay > 0:
            time.sleep(delay)
    return results


def build_taxonomy(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build intent taxonomy from threads dataset and persist to JSON.

    Args:
        cfg: Configuration dictionary.

    Returns:
        Constructed intent taxonomy.
    """
    threads_path = Path(cfg["data"]["threads_jsonl"])
    tax_path = Path(cfg["data"]["taxonomy_json"])
    cache_dir = Path(cfg["data"].get("cache_dir", ".cache"))
    n_clusters_target = tuple(cfg["intents"].get("n_clusters_target", [5, 8]))
    embed_model_name = cfg["model"].get("embedding_model", "all-MiniLM-L6-v2")
    embed_n = cfg["sampling"].get("taxonomy_embed_n", 2000)
    seed = cfg["sampling"].get("random_seed", 42)

    logger.info("Reading customer messages from %s for taxonomy discovery...", threads_path)
    pairs: List[Tuple[str, str]] = []
    with open(threads_path, "r", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            cust_turns = [turn for turn in t["turns"] if turn["role"] == "customer" and turn["text"]]
            if cust_turns:
                pairs.append((str(cust_turns[0]["tweet_id"]), cust_turns[0]["text"]))

    logger.info("Found %d initial customer inquiries across threads", len(pairs))

    np.random.seed(seed)
    if len(pairs) > embed_n:
        indices = np.random.choice(len(pairs), size=embed_n, replace=False)
        pairs = [pairs[i] for i in indices]
        logger.info("Subsampled to %d messages for clustering", len(pairs))

    taxonomy_tids = [p[0] for p in pairs]
    messages = [p[1] for p in pairs]

    # Persist taxonomy tweet IDs to ensure golden evaluation set is strictly disjoint
    tax_ids_file = cache_dir / "taxonomy_tweet_ids.json"
    with open(tax_ids_file, "w", encoding="utf-8") as f:
        json.dump(taxonomy_tids, f)
    logger.info("Saved %d taxonomy tweet IDs to %s", len(taxonomy_tids), tax_ids_file)

    embeddings = embed_messages(messages, model_name=embed_model_name, cache_dir=cache_dir)
    labels, method = cluster_messages(embeddings, n_clusters_range=n_clusters_target, random_state=seed)
    logger.info("Clustering completed via %s", method)

    taxonomy = propose_taxonomy(messages, labels, embeddings, llm_generate=generate)
    save_taxonomy(taxonomy, tax_path)
    return taxonomy


def main() -> None:
    """CLI entrypoint for building intent taxonomy."""
    parser = argparse.ArgumentParser(description="Build empirical intent taxonomy.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    build_taxonomy(cfg)


if __name__ == "__main__":
    main()
