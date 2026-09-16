"""Agent Orchestrator module for Hiver Customer Support Agent.

Sequences: classify -> route -> retrieve -> draft.
Always executes retrieval and drafting even on escalation to provide human assist
drafts with for_human_review=True. Never crashes on LLM errors.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import faiss

from src.intents import classify_intent
from src.llm import generate
from src.reply import draft_reply
from src.retrieve import retrieve
from src.router import route

logger = logging.getLogger(__name__)


def run_agent(
    customer_message: str,
    index: faiss.IndexFlatIP,
    corpus_meta: List[Dict[str, Any]],
    taxonomy: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    llm_generate: Callable[[str], str] = generate,
) -> Dict[str, Any]:
    """Execute end-to-end agent orchestration for an incoming customer message.

    Workflow:
    1. Classify intent with calibrated confidence.
    2. Retrieve top-k historically resolved threads from FAISS.
    3. Evaluate deterministic routing rules (auto_handle vs escalate).
    4. Draft Apple Support response (grounded on retrieved resolutions).
    5. Flag for human review if escalated.

    Resilient: Handled within try/except to return an error dict on failure
    without halting evaluation harnesses or reproduction pipelines.

    Args:
        customer_message: Raw or cleaned customer tweet.
        index: Loaded FAISS vector index.
        corpus_meta: Metadata records matching index entries.
        taxonomy: Loaded intent taxonomy dictionary.
        cfg: System configuration dictionary.
        llm_generate: LLM generation function.

    Returns:
        Dict with keys:
            - 'input': customer_message
            - 'intent': {intent, confidence, reasoning}
            - 'routing': {action, reason, decided_by}
            - 'retrieved': list of retrieved match objects
            - 'reply': {draft, grounding_used, source_thread_ids, for_human_review}
            - 'error': error message if failed, else None
    """
    try:
        # Step 1: Classify intent
        intent_res = classify_intent(
            message=customer_message,
            taxonomy=taxonomy,
            llm_generate=llm_generate,
        )
        intent_name = intent_res["intent"]
        confidence = intent_res["confidence"]

        # Step 2: Retrieve historical resolutions
        top_k = cfg.get("retrieval", {}).get("top_k", 3)
        min_sim = cfg.get("retrieval", {}).get("min_similarity", 0.40)
        embed_model = cfg.get("model", {}).get("embedding_model", "all-MiniLM-L6-v2")

        retrieved = retrieve(
            query=customer_message,
            index=index,
            corpus_meta=corpus_meta,
            top_k=top_k,
            min_similarity=min_sim,
            model_name=embed_model,
        )

        # Step 3: Determine routing action
        routing_res = route(
            message=customer_message,
            intent=intent_name,
            confidence=confidence,
            retrieved=retrieved,
            cfg=cfg,
        )

        # Step 4: Draft response
        intent_def = taxonomy.get(intent_name, {}).get("definition", "General inquiry.")
        reply_res = draft_reply(
            customer_message=customer_message,
            intent=intent_name,
            intent_definition=intent_def,
            retrieved=retrieved,
            llm_generate=llm_generate,
        )

        # Mark for human review if escalated
        is_escalated = (routing_res.get("action") == "escalate")
        reply_res["for_human_review"] = is_escalated

        return {
            "input": customer_message,
            "intent": intent_res,
            "routing": routing_res,
            "retrieved": retrieved,
            "reply": reply_res,
            "error": None,
        }

    except Exception as exc:
        logger.error("Agent execution encountered an unhandled error: %s", exc, exc_info=True)
        return {
            "input": customer_message,
            "intent": {"intent": "other", "confidence": 0.0, "reasoning": "Error occurred."},
            "routing": {"action": "escalate", "reason": "ERROR", "decided_by": "fallback"},
            "retrieved": [],
            "reply": {
                "draft": "We'd like to look into this with you. Please send us a direct message with more details.",
                "grounding_used": False,
                "source_thread_ids": [],
                "for_human_review": True,
            },
            "error": str(exc),
        }
