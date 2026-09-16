"""Reply Drafting module for Hiver Customer Support Agent.

Constructs structured, persona-driven prompts grounding on historical resolutions
retrieved via FAISS, and drafts concise, empathetic Apple Support replies capped
at 280 characters with full audit logging of source thread IDs.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from src.llm import generate

logger = logging.getLogger(__name__)


def build_prompt(
    customer_message: str,
    intent: str,
    intent_definition: str,
    retrieved: List[Dict[str, Any]],
) -> str:
    """Build grounding prompt for drafting an Apple Support reply.

    Includes persona instructions, intent context, and retrieved historical resolution
    exemplars. If no retrieval results are provided, instructs the model to generate
    a safe, generic troubleshooting inquiry without hallucinating grounding.

    Args:
        customer_message: Cleaned customer message text.
        intent: Classified intent name.
        intent_definition: Definition corresponding to the intent.
        retrieved: List of retrieved historical resolution dictionaries.

    Returns:
        Formatted prompt string.
    """
    grounding_blocks = []
    if retrieved:
        for idx, item in enumerate(retrieved[:3], 1):
            grounding_blocks.append(
                f"[Example {idx}]\n"
                f"Customer Query: {item.get('customer_message', '')}\n"
                f"Brand Resolution: {item.get('brand_resolution', '')}"
            )
        grounding_section = (
            "Historical Verified Resolutions for Similar Issues:\n"
            + "\n\n".join(grounding_blocks)
            + "\n\nUse the technical steps from the above examples as grounding, but adapt them "
            "specifically to the customer's query. DO NOT copy word-for-word."
        )
    else:
        grounding_section = (
            "Note: No historical resolution matches were found for this inquiry. "
            "Provide a helpful, accurate, general initial troubleshooting step or ask clarifying "
            "questions (e.g. device model, OS version). Do not fabricate specific policies or internal references."
        )

    prompt = (
        "You are an Apple Support specialist on Twitter/X.\n"
        "Brand Voice: Helpful, empathetic, polite, concise, and professional. Never robotic.\n\n"
        f"Customer Inquiry: \"{customer_message}\"\n"
        f"Identified Intent: {intent}\n"
        f"Intent Definition: {intent_definition}\n\n"
        f"{grounding_section}\n\n"
        "Guidelines:\n"
        "- Maximum length: strictly UNDER 280 characters.\n"
        "- Do NOT copy historical examples verbatim.\n"
        "- Do NOT mention internal systems, ticket IDs, or tool names.\n"
        "- Do NOT make guarantees or promises you cannot confirm (e.g. free replacements).\n"
        "- If serial number or private order details are needed, advise sending them via DM rather than in public.\n"
        "- Reply directly to the customer.\n\n"
        "Draft Reply:"
    )
    return prompt


def draft_reply(
    customer_message: str,
    intent: str,
    intent_definition: str,
    retrieved: List[Dict[str, Any]],
    llm_generate: Callable[[str], str] = generate,
) -> Dict[str, Any]:
    """Draft an Apple Support response using retrieval grounding.

    Logs source thread IDs at INFO level for complete traceability.

    Args:
        customer_message: Cleaned customer message.
        intent: Classified intent name.
        intent_definition: Definition of the intent.
        retrieved: List of retrieved resolution dictionaries.
        llm_generate: LLM generation callable.

    Returns:
        Dict with keys:
            - 'draft': Response text capped at 280 characters.
            - 'grounding_used': Boolean indicating if retrieval examples were provided.
            - 'source_thread_ids': List of thread IDs used for grounding.
    """
    source_thread_ids = [str(r.get("thread_id", "")) for r in retrieved if r.get("thread_id")]
    grounding_used = bool(retrieved)

    logger.info("Drafting reply using source thread IDs: %s", source_thread_ids)

    prompt = build_prompt(
        customer_message=customer_message,
        intent=intent,
        intent_definition=intent_definition,
        retrieved=retrieved,
    )

    draft = llm_generate(prompt).strip()

    # Clean any surrounding quotation marks
    if (draft.startswith('"') and draft.endswith('"')) or (draft.startswith("'") and draft.endswith("'")):
        draft = draft[1:-1].strip()

    # Hard cap at 280 characters
    if len(draft) > 280:
        logger.warning("Draft exceeded 280 characters (%d); truncating.", len(draft))
        draft = draft[:277] + "..."

    return {
        "draft": draft,
        "grounding_used": grounding_used,
        "source_thread_ids": source_thread_ids,
    }
