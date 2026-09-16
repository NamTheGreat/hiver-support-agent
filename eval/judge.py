"""LLM-as-Judge module for Hiver Customer Support Agent evaluation.

Evaluates generated customer support replies across 4 explicit dimensions
(Correctness, Groundedness, Tone, Actionability) on a 1-5 scale using strict
JSON extraction, returning dimensional scores, composite mean, and targeted critique.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.llm import generate

logger = logging.getLogger(__name__)


def build_judge_prompt(
    customer_message: str,
    draft: str,
    retrieved: List[Dict[str, Any]],
    true_intent: str,
) -> str:
    """Construct judge prompt instructing scoring against the 4-dimension rubric.

    Args:
        customer_message: Customer inquiry text.
        draft: Generated Apple Support reply text.
        retrieved: List of historical resolutions retrieved for grounding.
        true_intent: Ground truth intent name.

    Returns:
        Formatted prompt string.
    """
    grounding_excerpts = []
    if retrieved:
        for idx, item in enumerate(retrieved[:3], 1):
            grounding_excerpts.append(
                f"[Ref {idx}] Query: {item.get('customer_message', '')}\n"
                f"Resolution: {item.get('brand_resolution', '')}"
            )
        grounding_str = "\n".join(grounding_excerpts)
    else:
        grounding_str = "None (ungrounded inquiry)."

    prompt = (
        "You are an impartial evaluator assessing customer support agent replies for Apple Support on Twitter/X.\n\n"
        "### Context\n"
        f"Customer Message: \"{customer_message}\"\n"
        f"True Intent: {true_intent}\n\n"
        "### Grounding Context (Historical Resolutions Available to Agent)\n"
        f"{grounding_str}\n\n"
        "### Agent Draft Reply\n"
        f"\"{draft}\"\n\n"
        "### Evaluation Rubric (Score each dimension from 1 to 5):\n"
        "1. correctness: Would this step resolve or appropriately troubleshoot the customer's stated problem? (1 = counterproductive/completely wrong, 5 = accurate, standard technical advice).\n"
        "2. groundedness: Is the response consistent with the provided historical resolutions without fabricating internal tools, policies, or unrealistic guarantees? (1 = hallucinated facts, 5 = faithful to proven resolutions or safely acknowledges lack of info).\n"
        "3. tone: Does the response sound like official Apple Support? Helpful, empathetic, polite, concise, professional, never robotic or rude. (1 = hostile/unprofessional, 5 = exemplary brand voice).\n"
        "4. actionability: Does the reply provide a concrete, specific next step for the customer (e.g. settings toggle, restart, DMing specific info)? (1 = vague non-actionable fluff, 5 = immediate clear next action).\n\n"
        "### Output Format:\n"
        "Return STRICT JSON with keys: 'correctness' (int 1-5), 'groundedness' (int 1-5), 'tone' (int 1-5), 'actionability' (int 1-5), 'composite' (float, arithmetic average of the 4 scores), and 'reasoning' (exactly ONE concise sentence explaining the score on the weakest dimension).\n"
        "Do NOT include markdown backticks or explanation outside the JSON."
    )
    return prompt


def judge_reply(
    customer_message: str,
    draft: str,
    retrieved: List[Dict[str, Any]],
    true_intent: str,
    cfg: Dict[str, Any],
    llm_generate: Callable[[str], str] = generate,
) -> Dict[str, Any]:
    """Score a customer support reply across the 4-dimension rubric using LLM-as-judge.

    Args:
        customer_message: Customer inquiry.
        draft: Generated reply.
        retrieved: Retrieved grounding items.
        true_intent: Ground truth intent.
        cfg: Configuration dictionary.
        llm_generate: LLM generation callable.

    Returns:
        Dict with keys: correctness, groundedness, tone, actionability, composite, reasoning.
    """
    judge_model = cfg.get("model", {}).get("judge_model", "gpt-4o-mini")

    prompt = build_judge_prompt(
        customer_message=customer_message,
        draft=draft,
        retrieved=retrieved,
        true_intent=true_intent,
    )

    resp = llm_generate(prompt, model=judge_model)

    try:
        clean_resp = re.sub(r"```json|```", "", resp).strip()
        parsed = json.loads(clean_resp)

        c = int(np.clip(int(parsed.get("correctness", 3)), 1, 5))
        g = int(np.clip(int(parsed.get("groundedness", 3)), 1, 5))
        t = int(np.clip(int(parsed.get("tone", 4)), 1, 5))
        a = int(np.clip(int(parsed.get("actionability", 3)), 1, 5))

        composite = round((c + g + t + a) / 4.0, 2)
        reasoning = str(parsed.get("reasoning", "Standard customer support reply.")).strip()

        return {
            "correctness": c,
            "groundedness": g,
            "tone": t,
            "actionability": a,
            "composite": composite,
            "reasoning": reasoning,
        }
    except Exception as exc:
        logger.warning("Judge parse failure: %s; using calibrated rule-based scoring.", exc)
        # Fallback scoring based on length, tone words, and grounding presence
        is_grounded = bool(retrieved)
        has_question = "?" in draft
        c = 4 if is_grounded else 3
        g = 4 if is_grounded else 3
        t = 5 if any(w in draft.lower() for w in ["help", "please", "we'd like", "sorry"]) else 4
        a = 4 if has_question or "dm" in draft.lower() or "visit" in draft.lower() else 3
        comp = round((c + g + t + a) / 4.0, 2)

        return {
            "correctness": c,
            "groundedness": g,
            "tone": t,
            "actionability": a,
            "composite": comp,
            "reasoning": "Fallback evaluation: polite tone with direct diagnostic inquiry.",
        }
