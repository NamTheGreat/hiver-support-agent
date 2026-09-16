"""Data Preparation module for Hiver Customer Support Agent.

Filters raw Twitter customer support dataset for AppleSupport interactions,
reconstructs multi-turn conversational threads, cleans customer and brand
messages, applies quality filters, and extracts subsamples for reproducible evaluation.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yaml
from langdetect import DetectorFactory, detect_langs

# Deterministic language detection
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw Twitter Customer Support CSV dataset.

    Args:
        path: Path to CSV file (raw or subsampled).

    Returns:
        Loaded DataFrame with parsed timestamps and boolean inbound flags.
    """
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(
        path,
        dtype={
            "tweet_id": "str",
            "author_id": "str",
            "inbound": "str",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "str",
            "text": "str",
            "created_at": "str",
        },
        keep_default_na=False,
        low_memory=False,
    )
    df = df.fillna("")

    # Coerce inbound to boolean
    df["inbound"] = df["inbound"].astype(str).str.strip().str.lower().isin(["true", "1", "t"])

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def filter_brand(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """Filter dataset to include all tweets belonging to conversations involving brand_handle.

    Collects tweet_ids authored by the brand, walks in_response_to_tweet_id chains
    upward to include full thread contexts.

    Args:
        df: Raw tweets DataFrame.
        brand_handle: Twitter handle of brand (e.g. 'AppleSupport').

    Returns:
        DataFrame filtered to relevant threads containing brand turns.
    """
    logger.info("Filtering for brand %s", brand_handle)
    brand_lower = brand_handle.lower()

    # Identify brand tweets
    brand_mask = df["author_id"].astype(str).str.lower() == brand_lower
    brand_tweets = df[brand_mask]
    logger.info("Found %d tweets authored by %s", len(brand_tweets), brand_handle)

    if brand_tweets.empty:
        logger.warning("No tweets found for brand %s", brand_handle)
        return pd.DataFrame(columns=df.columns)

    # Fast indexing via dictionaries
    tweet_id_to_parent: Dict[str, str] = {}
    for tid, pid in zip(df["tweet_id"].astype(str), df["in_response_to_tweet_id"].astype(str)):
        if pid and pid.lower() not in ["nan", "none", ""]:
            tweet_id_to_parent[tid] = pid

    relevant_tweet_ids: Set[str] = set(brand_tweets["tweet_id"].astype(str))

    # Walk upward to find root and all parent tweets in thread
    for b_tid in brand_tweets["tweet_id"].astype(str):
        curr = b_tid
        visited = set()
        while curr in tweet_id_to_parent and curr not in visited:
            visited.add(curr)
            parent = tweet_id_to_parent[curr]
            relevant_tweet_ids.add(parent)
            curr = parent

    # Also collect direct brand responses listed in response_tweet_id
    for resp_str in brand_tweets["response_tweet_id"].dropna().astype(str):
        for r_id in resp_str.split(","):
            r_id = r_id.strip()
            if r_id and r_id.lower() not in ["nan", "none"]:
                relevant_tweet_ids.add(r_id)

    filtered_df = df[df["tweet_id"].astype(str).isin(relevant_tweet_ids)].copy()
    filtered_df["created_at_dt"] = pd.to_datetime(filtered_df["created_at"], format="%a %b %d %H:%M:%S +0000 %Y", errors="coerce", utc=True)
    logger.info("Retained %d tweets belonging to %s conversation threads", len(filtered_df), brand_handle)
    return filtered_df


def clean_text(text: str) -> str:
    """Clean tweet text while preserving natural casing for LLMs.

    Strips leading @mentions, removes URLs, and collapses extra whitespace.

    Args:
        text: Raw tweet text.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    # Strip leading @mentions (e.g. "@AppleSupport @user Hello" -> "Hello")
    cleaned = re.sub(r"^(@[A-Za-z0-9_]+\s*)+", "", text)
    # Remove remaining inline @mentions if bare
    cleaned = re.sub(r"@[A-Za-z0-9_]+", "", cleaned)
    # Remove URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Collapse consecutive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def reconstruct_threads(df: pd.DataFrame, brand_handle: str = "AppleSupport") -> List[Dict[str, Any]]:
    """Reconstruct multi-turn threads from tweet rows.

    Each thread = {thread_id, created_at, turns: [{role, text, tweet_id}, ...]}.
    Caps each thread at 10 turns. Discards threads with zero brand turns.

    Args:
        df: Filtered DataFrame containing brand and customer tweets.
        brand_handle: Brand handle to distinguish customer vs brand turns.

    Returns:
        List of thread dictionaries.
    """
    logger.info("Reconstructing conversation threads...")
    brand_lower = brand_handle.lower()

    # Build parent pointers and map tweet info
    tweet_dict: Dict[str, Dict[str, Any]] = {}
    parent_map: Dict[str, str] = {}

    for _, row in df.iterrows():
        tid = str(row["tweet_id"])
        pid = str(row.get("in_response_to_tweet_id", ""))
        author = str(row.get("author_id", ""))
        inbound = bool(row.get("inbound", False))
        created_at_dt = row.get("created_at_dt")
        created_at_str = str(row.get("created_at", ""))
        raw_text = str(row.get("text", ""))

        tweet_dict[tid] = {
            "tweet_id": tid,
            "parent_id": pid if pid and pid.lower() not in ["nan", "none", ""] else None,
            "author": author,
            "inbound": inbound,
            "created_at_dt": created_at_dt,
            "created_at_str": created_at_str,
            "text": raw_text,
        }
        if pid and pid.lower() not in ["nan", "none", ""]:
            parent_map[tid] = pid

    # Trace each tweet up to its thread root
    root_cache: Dict[str, str] = {}
    for tid in tweet_dict:
        curr = tid
        chain = []
        while curr in parent_map and parent_map[curr] in tweet_dict and curr not in chain:
            chain.append(curr)
            curr = parent_map[curr]
        root_cache[tid] = curr

    # Group tweets by thread root
    threads_by_root: Dict[str, List[Dict[str, Any]]] = {}
    for tid, info in tweet_dict.items():
        root = root_cache.get(tid, tid)
        threads_by_root.setdefault(root, []).append(info)

    reconstructed: List[Dict[str, Any]] = []
    for root_id, turns in threads_by_root.items():
        # Sort turns chronologically
        turns_sorted = sorted(turns, key=lambda x: (x["created_at_dt"] is pd.NaT, x["created_at_dt"]))

        # Cap at 10 turns
        turns_capped = turns_sorted[:10]

        # Check for at least one brand turn
        has_brand = any(
            (not t["inbound"]) or (t["author"].lower() == brand_lower)
            for t in turns_capped
        )
        if not has_brand:
            continue

        turn_objects = []
        for t in turns_capped:
            is_brand = (not t["inbound"]) or (t["author"].lower() == brand_lower)
            cleaned = clean_text(t["text"])
            turn_objects.append({
                "role": "brand" if is_brand else "customer",
                "text": cleaned,
                "raw_text": t["text"],
                "tweet_id": t["tweet_id"],
                "created_at": t["created_at_str"],
            })

        # Earliest created_at
        root_created_at = turns_capped[0]["created_at_str"]

        reconstructed.append({
            "thread_id": root_id,
            "created_at": root_created_at,
            "turns": turn_objects,
        })

    logger.info("Reconstructed %d threads with valid brand participation", len(reconstructed))
    return reconstructed


def is_bare_dm_redirect(brand_text: str) -> bool:
    """Check whether a brand response is solely a bare 'please DM us' redirect.

    Args:
        brand_text: Cleaned brand reply text.

    Returns:
        True if the reply contains no technical assistance and only requests a DM.
    """
    text = brand_text.lower().strip()
    bare_patterns = [
        r"^(please\s+)?(dm|send\s+a\s+dm|direct\s+message)\s+(us|me)?[\s\.\!]*$",
        r"^(please\s+)?reach\s+out\s+in\s+(dm|direct\s+message)[\s\.\!]*$",
        r"^(please\s+)?send\s+us\s+a\s+dm\s+with\s+more\s+details[\s\.\!]*$",
        r"^send\s+us\s+a\s+dm[\s\.\!]*$",
    ]
    for pat in bare_patterns:
        if re.match(pat, text):
            return True
    return False


def filter_threads(threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter threads by quality, language, and informative content.

    Drops threads where:
    - Only brand reply is a bare 'please DM us' redirect.
    - Non-English language detected (confidence > 0.9).
    - Every customer turn is empty after cleaning.

    Args:
        threads: Raw reconstructed threads list.

    Returns:
        Filtered list of clean threads.
    """
    logger.info("Filtering reconstructed threads for quality and language...")
    valid_threads: List[Dict[str, Any]] = []

    bare_dm_count = 0
    non_english_count = 0
    empty_customer_count = 0

    for thread in threads:
        turns = thread["turns"]

        # 1. Customer turns must not be completely empty
        customer_texts = [t["text"] for t in turns if t["role"] == "customer" and t["text"]]
        if not customer_texts:
            empty_customer_count += 1
            continue

        # 2. Check bare DM redirect
        brand_texts = [t["text"] for t in turns if t["role"] == "brand" and t["text"]]
        if brand_texts and all(is_bare_dm_redirect(bt) for bt in brand_texts):
            bare_dm_count += 1
            continue

        # 3. Language check on primary customer query
        first_query = customer_texts[0]
        if len(first_query) >= 15:
            try:
                langs = detect_langs(first_query)
                top_lang = langs[0]
                if top_lang.lang != "en" and top_lang.prob > 0.90:
                    non_english_count += 1
                    continue
            except Exception:
                pass

        valid_threads.append(thread)

    logger.info(
        "Filtered out: %d bare DM redirects, %d non-English threads, %d empty customer queries",
        bare_dm_count,
        non_english_count,
        empty_customer_count,
    )
    logger.info("Retained %d high-quality threads", len(valid_threads))
    return valid_threads


def prepare_dataset(cfg: Dict[str, Any]) -> None:
    """Orchestrate end-to-end data preparation pipeline.

    Loads raw CSV (or committed sample subsample), filters for brand,
    reconstructs threads, cleans, filters, generates subsample CSV if needed,
    and writes apple_threads.jsonl.

    Args:
        cfg: Configuration dictionary from config.yaml.
    """
    raw_path = Path(cfg["data"]["raw_csv"])
    subsample_path = Path(cfg["data"]["subsample_csv"])
    threads_path = Path(cfg["data"]["threads_jsonl"])
    brand_handle = cfg.get("brand", "AppleSupport")
    subsample_n = cfg["sampling"].get("data_subsample_n", 5000)
    seed = cfg["sampling"].get("random_seed", 42)

    # Determine which input CSV to load
    if raw_path.exists() and raw_path.stat().st_size > 1000:
        logger.info("Found raw CSV at %s; processing full dataset", raw_path)
        input_csv = raw_path
    elif subsample_path.exists() and subsample_path.stat().st_size > 1000:
        logger.info("Raw CSV absent; falling back to committed subsample at %s", subsample_path)
        input_csv = subsample_path
    else:
        raise FileNotFoundError(
            f"Neither raw CSV ({raw_path}) nor subsample CSV ({subsample_path}) found."
        )

    df_raw = load_raw(input_csv)
    df_brand = filter_brand(df_raw, brand_handle)

    # Reconstruct and filter threads
    threads = reconstruct_threads(df_brand, brand_handle=brand_handle)
    clean_threads = filter_threads(threads)

    # Save threads to JSONL
    threads_path.parent.mkdir(parents=True, exist_ok=True)
    with open(threads_path, "w", encoding="utf-8") as f:
        for t in clean_threads:
            f.write(json.dumps(t) + "\n")
    logger.info("Saved %d clean threads to %s", len(clean_threads), threads_path)

    # If processing raw twcs.csv and subsample_csv doesn't exist or needs refresh:
    if input_csv == raw_path and (not subsample_path.exists() or subsample_path.stat().st_size < 1000):
        logger.info("Extracting %d tweet subsample to %s for grader reproduction", subsample_n, subsample_path)
        # Collect tweet IDs from the clean threads
        clean_tweet_ids = set()
        for t in clean_threads:
            for turn in t["turns"]:
                clean_tweet_ids.add(turn["tweet_id"])

        subsample_tweets_df = df_brand[df_brand["tweet_id"].astype(str).isin(clean_tweet_ids)]

        np.random.seed(seed)
        if len(subsample_tweets_df) > subsample_n:
            # Sample by thread roots to keep full threads intact
            thread_ids = [t["thread_id"] for t in clean_threads]
            np.random.shuffle(thread_ids)
            selected_ids: Set[str] = set()
            count = 0
            for tid in thread_ids:
                matching_t = next(t for t in clean_threads if t["thread_id"] == tid)
                t_ids = {turn["tweet_id"] for turn in matching_t["turns"]}
                selected_ids.update(t_ids)
                count += len(t_ids)
                if count >= subsample_n:
                    break
            subsample_tweets_df = df_brand[df_brand["tweet_id"].astype(str).isin(selected_ids)]

        subsample_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop temporary parsing columns before saving CSV
        cols_to_save = [c for c in subsample_tweets_df.columns if c != "created_at_dt"]
        subsample_tweets_df[cols_to_save].to_csv(subsample_path, index=False)
        logger.info("Successfully exported %d tweets to %s", len(subsample_tweets_df), subsample_path)


def main() -> None:
    """CLI entrypoint for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare AppleSupport Twitter dataset.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
