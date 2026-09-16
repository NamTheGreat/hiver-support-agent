"""Evaluation Harness module for Hiver Customer Support Agent.

Runs uniform evaluation across Trivial, Simple, and Agent systems against the
golden evaluation set. Computes intent classification accuracy and macro-F1,
routing precision/recall/F1 and reason breakdown, and dimensional LLM-as-judge scores.
Exports standardized metrics JSON and per-example audit CSV.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from eval.baselines import SimpleBaseline, TrivialBaseline
from eval.judge import judge_reply
from src.agent import run_agent
from src.intents import load_taxonomy
from src.retrieve import load_index, retrieve

logger = logging.getLogger(__name__)


def compute_metrics(
    records: List[Dict[str, Any]],
    system_name: str,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics across classification, routing, and judging.

    Args:
        records: List of per-example evaluation record dictionaries.
        system_name: System identifier string ('trivial', 'simple', 'agent').

    Returns:
        Structured metrics dictionary.
    """
    n_examples = len(records)
    if n_examples == 0:
        return {"system": system_name, "n_examples": 0}

    gold_intents = [r["gold_intent"] for r in records]
    pred_intents = [r["pred_intent"] for r in records]

    # Intent Classification Metrics
    intent_acc = float(accuracy_score(gold_intents, pred_intents))
    intent_macro_f1 = float(f1_score(gold_intents, pred_intents, average="macro", zero_division=0))
    clf_rep = classification_report(gold_intents, pred_intents, output_dict=True, zero_division=0)

    per_class_metrics = {}
    for cls_name, vals in clf_rep.items():
        if isinstance(vals, dict):
            per_class_metrics[cls_name] = {
                "precision": round(vals.get("precision", 0.0), 4),
                "recall": round(vals.get("recall", 0.0), 4),
                "f1": round(vals.get("f1-score", 0.0), 4),
                "support": int(vals.get("support", 0)),
            }

    # Routing Metrics (treating 'escalate' as positive class 1)
    gold_escalate_bin = [1 if r["gold_escalate"] == "yes" else 0 for r in records]
    pred_escalate_bin = [1 if r["pred_action"] == "escalate" else 0 for r in records]

    routing_acc = float(accuracy_score(gold_escalate_bin, pred_escalate_bin))
    routing_p = float(precision_score(gold_escalate_bin, pred_escalate_bin, zero_division=0))
    routing_r = float(recall_score(gold_escalate_bin, pred_escalate_bin, zero_division=0))
    routing_f1 = float(f1_score(gold_escalate_bin, pred_escalate_bin, zero_division=0))

    # Reason breakdown
    reason_counts: Dict[str, int] = {}
    for r in records:
        reason = r.get("pred_reason") or "NONE"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # Judge Metrics
    correctness_scores = [r["judge_correctness"] for r in records if "judge_correctness" in r]
    groundedness_scores = [r["judge_groundedness"] for r in records if "judge_groundedness" in r]
    tone_scores = [r["judge_tone"] for r in records if "judge_tone" in r]
    actionability_scores = [r["judge_actionability"] for r in records if "judge_actionability" in r]
    composite_scores = [r["judge_composite"] for r in records if "judge_composite" in r]

    judge_metrics = {
        "mean_correctness": round(float(np.mean(correctness_scores)), 3) if correctness_scores else 0.0,
        "mean_groundedness": round(float(np.mean(groundedness_scores)), 3) if groundedness_scores else 0.0,
        "mean_tone": round(float(np.mean(tone_scores)), 3) if tone_scores else 0.0,
        "mean_actionability": round(float(np.mean(actionability_scores)), 3) if actionability_scores else 0.0,
        "mean_composite": round(float(np.mean(composite_scores)), 3) if composite_scores else 0.0,
    }

    return {
        "system": system_name,
        "n_examples": n_examples,
        "intent_accuracy": round(intent_acc, 4),
        "intent_macro_f1": round(intent_macro_f1, 4),
        "per_class_metrics": per_class_metrics,
        "routing": {
            "accuracy": round(routing_acc, 4),
            "precision": round(routing_p, 4),
            "recall": round(routing_r, 4),
            "f1": round(routing_f1, 4),
            "escalation_rate": round(float(np.mean(pred_escalate_bin)), 4),
            "reason_breakdown": reason_counts,
        },
        "judge": judge_metrics,
    }


def run_evaluation(
    golden_path: Path,
    cfg: Dict[str, Any],
    system: str = "agent",
) -> Dict[str, Any]:
    """Execute evaluation harness on the golden dataset for a designated system.

    Evaluates:
    - 'trivial': Constant majority class, canned static link, always escalates.
    - 'simple': TF-IDF + LogisticRegression, verbatim top-1 historical reply, rule router.
    - 'agent': Full agent (LLM intent classifier + FAISS + rule router + drafted reply).

    Args:
        golden_path: Path to golden_eval.jsonl.
        cfg: Configuration dictionary.
        system: System identifier ('trivial', 'simple', 'agent').

    Returns:
        Evaluation metrics dictionary.
    """
    golden_path = Path(golden_path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    logger.info("Running evaluation for system '%s' on %s...", system, golden_path)

    # Load golden records
    golden_items: List[Dict[str, Any]] = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            golden_items.append(json.loads(line))

    logger.info("Loaded %d golden evaluation examples", len(golden_items))

    # Initialize components
    index, corpus_meta = load_index(cfg)
    taxonomy = load_taxonomy(Path(cfg["data"]["taxonomy_json"]))

    # System setup
    if system == "trivial":
        model = TrivialBaseline(majority_intent="software_update")
    elif system == "simple":
        cache_dir = Path(cfg["data"].get("cache_dir", ".cache"))
        model_path = cache_dir / "simple_baseline_model.pkl"
        model = SimpleBaseline(model_path=model_path)
    elif system == "agent":
        model = None
    else:
        raise ValueError(f"Unknown system: {system}. Must be 'trivial', 'simple', or 'agent'.")

    top_k = cfg.get("retrieval", {}).get("top_k", 3)
    min_sim = cfg.get("retrieval", {}).get("min_similarity", 0.40)
    embed_model_name = cfg.get("model", {}).get("embedding_model", "all-MiniLM-L6-v2")

    records: List[Dict[str, Any]] = []

    for idx, item in enumerate(golden_items, 1):
        msg = item["customer_message"]
        gold_intent = item["intent"]
        gold_escalate = item["escalate"]

        # Run system pipeline
        if system in ["trivial", "simple"]:
            retrieved = retrieve(
                query=msg,
                index=index,
                corpus_meta=corpus_meta,
                top_k=top_k,
                min_similarity=min_sim,
                model_name=embed_model_name,
            )
            out = model.run(msg, retrieved, cfg)
        else:
            out = run_agent(
                customer_message=msg,
                index=index,
                corpus_meta=corpus_meta,
                taxonomy=taxonomy,
                cfg=cfg,
            )

        pred_intent = out.get("intent", {}).get("intent", "other")
        pred_confidence = out.get("intent", {}).get("confidence", 0.0)
        pred_action = out.get("routing", {}).get("action", "escalate")
        pred_reason = out.get("routing", {}).get("reason")
        draft = out.get("reply", {}).get("draft", "")
        retrieved_items = out.get("retrieved", [])

        # Judge reply
        j_score = judge_reply(
            customer_message=msg,
            draft=draft,
            retrieved=retrieved_items,
            true_intent=gold_intent,
            cfg=cfg,
        )

        record = {
            "id": item["id"],
            "tweet_id": item["tweet_id"],
            "customer_message": msg,
            "gold_intent": gold_intent,
            "pred_intent": pred_intent,
            "intent_match": bool(gold_intent == pred_intent),
            "confidence": pred_confidence,
            "gold_escalate": gold_escalate,
            "pred_action": pred_action,
            "routing_match": bool(gold_escalate == ("yes" if pred_action == "escalate" else "no")),
            "pred_reason": pred_reason,
            "draft_reply": draft,
            "grounding_count": len(retrieved_items),
            "judge_correctness": j_score["correctness"],
            "judge_groundedness": j_score["groundedness"],
            "judge_tone": j_score["tone"],
            "judge_actionability": j_score["actionability"],
            "judge_composite": j_score["composite"],
            "judge_reasoning": j_score["reasoning"],
        }
        records.append(record)

        if idx % 50 == 0 or idx == len(golden_items):
            logger.info("Evaluated %d/%d items for '%s'", idx, len(golden_items), system)

    # Compute metrics
    metrics = compute_metrics(records, system_name=system)

    # Save outputs
    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = results_dir / f"{system}_metrics.json"
    details_file = results_dir / f"{system}_eval_details.csv"

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(records).to_csv(details_file, index=False)
    logger.info("Saved %s metrics to %s and details to %s", system, metrics_file, details_file)

    return metrics


def main() -> None:
    """CLI entrypoint for evaluation harness."""
    parser = argparse.ArgumentParser(description="Evaluate support agent system.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--system", type=str, default="agent", choices=["trivial", "simple", "agent", "all"], help="System to evaluate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    golden_path = Path(cfg["data"]["golden_jsonl"])

    systems = ["trivial", "simple", "agent"] if args.system == "all" else [args.system]
    for sys in systems:
        run_evaluation(golden_path, cfg, system=sys)


if __name__ == "__main__":
    main()
