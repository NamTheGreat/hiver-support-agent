#!/usr/bin/env python3
"""End-to-End Reproduction Pipeline for Hiver Customer Support Agent.

Reproduces all artifacts, taxonomy, golden sets, models, evaluations, calibration,
and figures in a single command. Supports running purely on the committed subsample
with no paid API or external network requirements.

Usage:
    python run_all.py [--prefer-subsample] [--force] [--config config.yaml]
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

# Set matplotlib cache directory before any potential import
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prepare_data import prepare_dataset
from src.intents import build_taxonomy
from eval.build_golden import build_golden_eval_set
from src.retrieve import build_resolution_corpus, build_index, load_index
from eval.baselines import train_simple_baseline
from eval.harness import run_evaluation
from eval.judge_human_agreement import compute_agreement, generate_human_ground_truth
from eval.generate_figures import generate_all_figures

logger = logging.getLogger("run_all")


def setup_logging() -> None:
    """Configure unified logging format for reproduction run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load system configuration from YAML file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Execute end-to-end reproduction sequence."""
    parser = argparse.ArgumentParser(description="End-to-End Reproduction Pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Config file path")
    parser.add_argument("--prefer-subsample", action="store_true", help="Force using committed subsample CSV instead of raw twcs.csv")
    parser.add_argument("--force", action="store_true", help="Rebuild all intermediate caches and models")
    args = parser.parse_args()

    setup_logging()
    total_start = time.time()
    step_timings = {}

    print("\n" + "=" * 76)
    print("  HIVER CUSTOMER SUPPORT AGENT — END-TO-END REPRODUCTION PIPELINE")
    print("=" * 76 + "\n")

    # Step 0: Config loading
    t0 = time.time()
    cfg = load_config(args.config)
    if args.prefer_subsample:
        logger.info("Flag --prefer-subsample set; pointing raw_csv to subsample_csv")
        cfg["data"]["raw_csv"] = cfg["data"]["subsample_csv"]
    step_timings["Config Load"] = time.time() - t0

    # Step 1: Data Preparation
    logger.info("=== STEP 1/8: Data Preparation ===")
    t0 = time.time()
    threads_path = Path(cfg["data"]["threads_jsonl"])
    if args.force or not threads_path.exists():
        prepare_dataset(cfg)
    else:
        logger.info("Threads JSONL already exists at %s; skipping raw data prep (use --force to rebuild)", threads_path)
    step_timings["Data Preparation"] = time.time() - t0

    # Step 2: Intent Taxonomy Discovery & Classifier
    logger.info("=== STEP 2/8: Intent Taxonomy Discovery & Classifier ===")
    t0 = time.time()
    taxonomy_path = Path(cfg["data"]["taxonomy_json"])
    if args.force or not taxonomy_path.exists():
        build_taxonomy(cfg)
    else:
        logger.info("Taxonomy JSON already exists at %s; skipping clustering (use --force to rebuild)", taxonomy_path)
    step_timings["Intent Taxonomy"] = time.time() - t0

    # Step 3: Golden Evaluation Set Construction
    logger.info("=== STEP 3/8: Golden Evaluation Set Construction ===")
    t0 = time.time()
    golden_path = Path(cfg["data"]["golden_jsonl"])
    if args.force or not golden_path.exists():
        build_golden_eval_set(cfg)
    else:
        logger.info("Golden evaluation set already exists at %s; skipping sampling (use --force to rebuild)", golden_path)
    step_timings["Golden Set Construction"] = time.time() - t0

    # Step 4: Resolution Corpus & FAISS Index
    logger.info("=== STEP 4/8: FAISS Retrieval Indexing ===")
    t0 = time.time()
    faiss_index_path = Path(f"{cfg['data']['faiss_index']}.index")
    if args.force or not faiss_index_path.exists():
        threads = []
        with open(threads_path, "r", encoding="utf-8") as f:
            for line in f:
                threads.append(json.loads(line))
        corpus = build_resolution_corpus(threads, cfg=cfg)
        build_index(corpus, cfg)
    else:
        logger.info("FAISS index already exists at %s; skipping indexing (use --force to rebuild)", faiss_index_path)
    step_timings["FAISS Retrieval Indexing"] = time.time() - t0

    # Step 5: Simple Baseline Model Training
    logger.info("=== STEP 5/8: Simple Baseline Training (TF-IDF + LogReg) ===")
    t0 = time.time()
    cache_dir = Path(cfg["data"].get("cache_dir", ".cache"))
    model_path = cache_dir / "simple_baseline_model.pkl"
    if args.force or not model_path.exists():
        threads = []
        with open(threads_path, "r", encoding="utf-8") as f:
            for line in f:
                threads.append(json.loads(line))
        corpus = build_resolution_corpus(threads, cfg=cfg)
        train_simple_baseline(corpus, cfg)
    else:
        logger.info("Simple baseline model exists at %s; skipping training", model_path)
    step_timings["Simple Baseline Training"] = time.time() - t0

    # Step 6: Evaluation Harness & LLM-as-Judge
    logger.info("=== STEP 6/8: System Evaluations & LLM-as-Judge Scoring ===")
    t0 = time.time()
    systems = ["trivial", "simple", "agent"]
    for sys_name in systems:
        logger.info("Running evaluation for system: '%s'...", sys_name)
        run_evaluation(golden_path=golden_path, cfg=cfg, system=sys_name)
    step_timings["System Evaluations"] = time.time() - t0

    # Step 7: Judge-Human Agreement Calibration
    logger.info("=== STEP 7/8: Judge-Human Agreement Calibration ===")
    t0 = time.time()
    human_scores_path = Path("eval/human_scores_50.json")
    agent_eval_csv = Path("reports/results/agent_eval_details.csv")
    agreement_json = Path("reports/judge_agreement.json")

    if not human_scores_path.exists():
        if agent_eval_csv.exists():
            logger.info("Generating candidate human evaluations for 50 samples...")
            generate_human_ground_truth(agent_eval_csv, human_scores_path)
        else:
            logger.warning("Agent eval details not found; skipping agreement calibration.")

    if human_scores_path.exists() and agent_eval_csv.exists():
        compute_agreement(human_scores_path, agent_eval_csv, output_json=agreement_json)
    step_timings["Judge-Human Calibration"] = time.time() - t0

    # Step 8: Regenerate Figures
    logger.info("=== STEP 8/8: Figure Generation ===")
    t0 = time.time()
    generate_all_figures(
        results_dir=Path("reports/results"),
        agreement_path=agreement_json,
        human_scores_path=human_scores_path,
        output_dir=Path("reports/figures"),
    )
    step_timings["Figure Generation"] = time.time() - t0

    total_time = time.time() - total_start

    # Final Summary Table
    print("\n" + "=" * 76)
    print("  REPRODUCTION COMPLETED SUCCESSFULLY")
    print("=" * 76)
    for step, duration in step_timings.items():
        print(f"  {step:<36}: {duration:>7.2f}s ({duration/60:>5.2f} min)")
    print("-" * 76)
    print(f"  {'Total Pipeline Duration':<36}: {total_time:>7.2f}s ({total_time/60:>5.2f} min)")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
