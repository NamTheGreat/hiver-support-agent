"""Figure generation module for Hiver Customer Support Agent.

Generates presentation-ready evaluation charts:
1. System Performance Comparison (Intent Accuracy, Routing F1, Composite Judge)
2. Judge Rubric Dimensional Analysis across Systems
3. Human-Judge Agreement Scatter Plot
4. Router Escalation Reason Distribution
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
Path(".cache/matplotlib").mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def generate_all_figures(
    results_dir: Path = Path("reports/results"),
    agreement_path: Path = Path("reports/judge_agreement.json"),
    human_scores_path: Path = Path("eval/human_scores_50.json"),
    output_dir: Path = Path("reports/figures"),
) -> None:
    """Generate and save all publication-quality evaluation figures.

    Args:
        results_dir: Directory containing system metrics JSON files.
        agreement_path: Path to reports/judge_agreement.json.
        human_scores_path: Path to eval/human_scores_50.json.
        output_dir: Output directory for figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set aesthetic theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    # Load metrics
    systems = ["trivial", "simple", "agent"]
    metrics: Dict[str, Dict[str, Any]] = {}
    for sys in systems:
        m_file = results_dir / f"{sys}_metrics.json"
        if m_file.exists():
            with open(m_file, "r", encoding="utf-8") as f:
                metrics[sys] = json.load(f)

    if not metrics:
        logger.warning("No metrics found in %s; skipping figure generation.", results_dir)
        return

    # Figure 1: System Performance Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    df_comp = pd.DataFrame([
        {
            "System": s.capitalize(),
            "Intent Accuracy (%)": metrics[s]["intent_accuracy"] * 100,
            "Routing F1 (%)": metrics[s]["routing"]["f1"] * 100,
            "Composite Judge (x20)": metrics[s]["judge"]["mean_composite"] * 20,
        }
        for s in systems if s in metrics
    ])

    melted = df_comp.melt(id_vars="System", var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="System", y="Score", hue="Metric", ax=ax, palette=["#3498db", "#2ecc71", "#e74c3c"])
    ax.set_title("System Performance Comparison (Golden Evaluation Set N=180)", pad=14, fontweight="bold")
    ax.set_ylabel("Score / Percentage")
    ax.set_ylim(0, 110)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1f}", (p.get_x() + p.get_width() / 2.0, height + 1.5),
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig1_path = output_dir / "system_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig1_path)

    # Figure 2: Judge Rubric Dimensions Breakdown
    fig, ax = plt.subplots(figsize=(9, 5))
    dim_rows = []
    dim_names = ["correctness", "groundedness", "tone", "actionability", "composite"]
    for s in systems:
        if s in metrics:
            for d in dim_names:
                dim_rows.append({
                    "System": s.capitalize(),
                    "Dimension": d.capitalize(),
                    "Score": metrics[s]["judge"][f"mean_{d}"],
                })
    df_dims = pd.DataFrame(dim_rows)
    sns.barplot(data=df_dims, x="Dimension", y="Score", hue="System", ax=ax, palette="Blues_d")
    ax.set_title("LLM-as-Judge 4-Dimension Rubric Across Systems (1-5 Scale)", pad=14, fontweight="bold")
    ax.set_ylabel("Mean Judge Score (1 - 5)")
    ax.set_ylim(0, 5.5)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2.0, height + 0.1),
                        ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig2_path = output_dir / "judge_dimensions.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved %s", fig2_path)

    # Figure 3: Human-Judge Calibration Scatter
    agent_csv = results_dir / "agent_eval_details.csv"
    if human_scores_path.exists() and agent_csv.exists():
        with open(human_scores_path, "r", encoding="utf-8") as f:
            h_data = json.load(f)
        df_h = pd.DataFrame(h_data)
        df_llm = pd.read_csv(agent_csv)
        df_merged = pd.merge(df_h, df_llm, on="id")

        fig, ax = plt.subplots(figsize=(6, 6))
        # Composite score
        h_comp = df_merged[["correctness", "groundedness", "tone", "actionability"]].mean(axis=1)
        l_comp = df_merged["judge_composite"]

        sns.regplot(x=l_comp, y=h_comp, ax=ax, color="#8e44ad",
                    scatter_kws={"alpha": 0.6, "s": 50}, line_kws={"color": "#2c3e50", "linestyle": "--"})
        ax.set_title("Human vs. LLM Judge Calibration (N=50)", pad=14, fontweight="bold")
        ax.set_xlabel("LLM Judge Composite Score")
        ax.set_ylabel("Human Candidate Composite Score")
        ax.set_xlim(2.5, 5.2)
        ax.set_ylim(2.5, 5.2)
        plt.tight_layout()
        fig3_path = output_dir / "judge_human_scatter.png"
        plt.savefig(fig3_path, dpi=300)
        plt.close()
        logger.info("Saved %s", fig3_path)

    # Figure 4: Routing Escalation Breakdown for Agent
    if "agent" in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        reasons = metrics["agent"]["routing"]["reason_breakdown"]
        labels = [k.replace("_", " ").title() for k in reasons.keys()]
        counts = list(reasons.values())

        sns.barplot(x=labels, y=counts, ax=ax, palette="rocket")
        ax.set_title("Agent Routing Breakdown by Escalation Reason", pad=14, fontweight="bold")
        ax.set_ylabel("Number of Queries")
        ax.set_xlabel("Reason / Trigger")
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2.0, height + 1),
                            ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig4_path = output_dir / "routing_breakdown.png"
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        logger.info("Saved %s", fig4_path)

    logger.info("All evaluation figures successfully generated in %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    generate_all_figures()
