"""Deterministic Rule-Based Router module for Hiver Customer Support Agent.

Auditable, fast, and testable routing without LLM dependencies. Evaluates
calibrated confidence thresholds, safety keywords, sentiment polarity,
and retrieval grounding against explicit escalation criteria.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Single shared VADER sentiment analyzer instance
_SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()

# High-risk safety keywords that demand human agent escalation
HIGH_RISK_KEYWORDS = {
    "hacked",
    "stolen",
    "unauthorized",
    "fraud",
    "legal",
    "sue",
    "lawsuit",
    "lawyer",
    "attorney",
    "court",
    "police",
    "compromised",
    "breached",
}

# Intents that present elevated financial or account security exposure
HIGH_RISK_INTENT_FAMILIES = {
    "account_access",
    "billing_order",
    "billing_charge",
    "payment",
    "security",
    "legal",
    "hardware_repair",
}


class EscalationReason:
    """Typed constants for escalation reasons."""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    HIGH_RISK_INTENT = "HIGH_RISK_INTENT"
    NEGATIVE_SENTIMENT = "NEGATIVE_SENTIMENT"
    OTHER_INTENT = "OTHER_INTENT"
    NO_GROUNDING = "NO_GROUNDING"


def route(
    message: str,
    intent: str,
    confidence: float,
    retrieved: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate routing rules to determine whether to auto_handle or escalate.

    Escalation rules (evaluated in deterministic order):
    1. confidence below threshold -> LOW_CONFIDENCE
    2. intent == "other" -> OTHER_INTENT
    3. intent in high-risk set and message contains risk keywords -> HIGH_RISK_INTENT
    4. VADER sentiment compound below negative threshold -> NEGATIVE_SENTIMENT
    5. empty retrieval results -> NO_GROUNDING
    6. If all checks pass:
       - Auto-handle if intent is in auto_handle_allowlist
       - Otherwise, conservatively escalate with HIGH_RISK_INTENT

    Args:
        message: Cleaned customer message text.
        intent: Classified intent string.
        confidence: Classification confidence float [0.0, 1.0].
        retrieved: List of retrieved historical resolution matches.
        cfg: Configuration dictionary from config.yaml.

    Returns:
        Dict with keys:
            - 'action': 'auto_handle' | 'escalate'
            - 'reason': EscalationReason constant or None if auto_handle
            - 'decided_by': 'rule'
    """
    routing_cfg = cfg.get("routing", {})
    conf_threshold = routing_cfg.get("confidence_threshold", 0.80)
    sentiment_threshold = routing_cfg.get("sentiment_negative_threshold", -0.5)
    allowlist = set(routing_cfg.get("auto_handle_allowlist", ["connectivity_issue", "software_update", "battery_charging"]))

    # Rule 1: Confidence threshold check
    if confidence < conf_threshold:
        logger.info("Escalation triggered: LOW_CONFIDENCE (%.2f < %.2f)", confidence, conf_threshold)
        return {
            "action": "escalate",
            "reason": EscalationReason.LOW_CONFIDENCE,
            "decided_by": "rule",
        }

    # Rule 2: Out of scope / Ambiguous 'other' intent
    if intent.lower() == "other":
        logger.info("Escalation triggered: OTHER_INTENT")
        return {
            "action": "escalate",
            "reason": EscalationReason.OTHER_INTENT,
            "decided_by": "rule",
        }

    # Rule 3: High-risk intent + safety keyword trigger
    msg_words = set(re.findall(r"\b[a-z]+\b", message.lower()))
    has_risk_keyword = bool(msg_words & HIGH_RISK_KEYWORDS)
    if intent.lower() in HIGH_RISK_INTENT_FAMILIES and has_risk_keyword:
        logger.info("Escalation triggered: HIGH_RISK_INTENT (intent '%s' + keywords)", intent)
        return {
            "action": "escalate",
            "reason": EscalationReason.HIGH_RISK_INTENT,
            "decided_by": "rule",
        }

    # Rule 4: Severe negative customer sentiment (frustration / anger)
    sentiment_scores = _SENTIMENT_ANALYZER.polarity_scores(message)
    compound = sentiment_scores.get("compound", 0.0)
    if compound < sentiment_threshold:
        logger.info("Escalation triggered: NEGATIVE_SENTIMENT (compound %.3f < %.3f)", compound, sentiment_threshold)
        return {
            "action": "escalate",
            "reason": EscalationReason.NEGATIVE_SENTIMENT,
            "decided_by": "rule",
        }

    # Rule 5: No historical grounding retrieved
    if not retrieved:
        logger.info("Escalation triggered: NO_GROUNDING (empty retrieval matches)")
        return {
            "action": "escalate",
            "reason": EscalationReason.NO_GROUNDING,
            "decided_by": "rule",
        }

    # Rule 6: Allowlist qualification
    if intent.lower() in allowlist:
        logger.info("Auto-handle approved for intent '%s' (confidence=%.2f, sentiment=%.2f)", intent, confidence, compound)
        return {
            "action": "auto_handle",
            "reason": None,
            "decided_by": "rule",
        }

    # Conservative default: High confidence and grounded, but intent not allowlisted for autonomous replies
    logger.info("Escalation triggered: HIGH_RISK_INTENT (intent '%s' not in auto_handle allowlist)", intent)
    return {
        "action": "escalate",
        "reason": EscalationReason.HIGH_RISK_INTENT,
        "decided_by": "rule",
    }
