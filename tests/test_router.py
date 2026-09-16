"""Unit tests for src/router.py.

Tests each escalation rule independently and verifies the auto-handle happy path,
mocking inputs and using no network or paid APIs.
"""

import pytest
from src.router import EscalationReason, route


@pytest.fixture
def base_cfg():
    """Default configuration dictionary fixture."""
    return {
        "routing": {
            "confidence_threshold": 0.80,
            "auto_handle_allowlist": ["connectivity_issue", "software_update", "battery_charging"],
            "sentiment_negative_threshold": -0.5,
        }
    }


@pytest.fixture
def mock_retrieved():
    """Mock non-empty retrieved grounding list."""
    return [
        {
            "thread_id": "1001",
            "customer_message": "Wi-Fi is disconnecting continuously.",
            "brand_resolution": "Try resetting network settings in Settings > General > Reset.",
            "intent": "connectivity_issue",
            "similarity_score": 0.82,
        }
    ]


def test_auto_handle_happy_path(base_cfg, mock_retrieved):
    """Test auto-handle when all criteria are satisfied for an allowlisted intent."""
    res = route(
        message="My Wi-Fi keeps disconnecting on iOS 17.",
        intent="connectivity_issue",
        confidence=0.95,
        retrieved=mock_retrieved,
        cfg=base_cfg,
    )
    assert res["action"] == "auto_handle"
    assert res["reason"] is None
    assert res["decided_by"] == "rule"


def test_escalation_low_confidence(base_cfg, mock_retrieved):
    """Test escalation when classifier confidence falls below threshold."""
    res = route(
        message="My Wi-Fi keeps disconnecting on iOS 17.",
        intent="connectivity_issue",
        confidence=0.65,  # Below 0.80 threshold
        retrieved=mock_retrieved,
        cfg=base_cfg,
    )
    assert res["action"] == "escalate"
    assert res["reason"] == EscalationReason.LOW_CONFIDENCE
    assert res["decided_by"] == "rule"


def test_escalation_other_intent(base_cfg, mock_retrieved):
    """Test escalation when intent is classified as 'other'."""
    res = route(
        message="What is the meaning of life?",
        intent="other",
        confidence=0.99,
        retrieved=mock_retrieved,
        cfg=base_cfg,
    )
    assert res["action"] == "escalate"
    assert res["reason"] == EscalationReason.OTHER_INTENT


def test_escalation_high_risk_intent_with_keywords(base_cfg, mock_retrieved):
    """Test escalation when high-risk intent contains risk keywords like 'hacked'."""
    res = route(
        message="My Apple ID was hacked and my account is locked!",
        intent="account_access",
        confidence=0.92,
        retrieved=mock_retrieved,
        cfg=base_cfg,
    )
    assert res["action"] == "escalate"
    assert res["reason"] == EscalationReason.HIGH_RISK_INTENT


def test_escalation_negative_sentiment(base_cfg, mock_retrieved):
    """Test escalation when customer message exhibits extreme negative sentiment."""
    res = route(
        message="I hate Apple so much, this terrible horrible service is disgusting and sickening.",
        intent="connectivity_issue",
        confidence=0.95,
        retrieved=mock_retrieved,
        cfg=base_cfg,
    )
    assert res["action"] == "escalate"
    assert res["reason"] == EscalationReason.NEGATIVE_SENTIMENT


def test_escalation_no_grounding(base_cfg):
    """Test escalation when retrieval returns no similar historical threads."""
    res = route(
        message="My Wi-Fi keeps disconnecting on iOS 17.",
        intent="connectivity_issue",
        confidence=0.95,
        retrieved=[],  # Empty retrieval
        cfg=base_cfg,
    )
    assert res["action"] == "escalate"
    assert res["reason"] == EscalationReason.NO_GROUNDING


def test_escalation_non_allowlisted_intent(base_cfg, mock_retrieved):
    """Test conservative escalation when intent is not in the auto-handle allowlist."""
    res = route(
        message="My iPhone screen is cracked and needs glass replacement.",
        intent="hardware_repair",  # Valid intent, but not allowlisted
        confidence=0.95,
        retrieved=mock_retrieved,
        cfg=base_cfg,
    )
    assert res["action"] == "escalate"
    assert res["reason"] == EscalationReason.HIGH_RISK_INTENT
