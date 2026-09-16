"""End-to-end tests for the support agent pipeline.

Evaluates run_agent() across 5 distinct intent scenarios with mocked LLM and
retrieval functions to ensure full offline, network-free execution and proper
response schema validation.
"""

import json
from unittest.mock import patch
import pytest
from src.agent import run_agent


@pytest.fixture
def agent_cfg():
    """Default agent configuration fixture."""
    return {
        "routing": {
            "confidence_threshold": 0.80,
            "auto_handle_allowlist": ["battery_charging", "connectivity_issue", "software_update"],
            "sentiment_negative_threshold": -0.5,
        },
        "retrieval": {
            "top_k": 3,
            "min_similarity": 0.40,
        },
        "model": {
            "embedding_model": "all-MiniLM-L6-v2",
        },
    }


@pytest.fixture
def agent_taxonomy():
    """Mock intent taxonomy covering 5 core intents + other."""
    return {
        "battery_charging": {
            "definition": "Battery drain, percentage indicator, or charging issues.",
            "example_tweets": ["Battery drains in 1 hour."],
        },
        "software_update": {
            "definition": "iOS update installation issues or version questions.",
            "example_tweets": ["Cannot install iOS update."],
        },
        "connectivity_issue": {
            "definition": "Wi-Fi, Bluetooth, or cellular network connection problems.",
            "example_tweets": ["Wi-Fi disconnected."],
        },
        "account_access": {
            "definition": "Apple ID login, iCloud sign-in, or password lockout.",
            "example_tweets": ["Apple ID password reset."],
        },
        "hardware_repair": {
            "definition": "Physical hardware defects, cracked screens, or repair appointments.",
            "example_tweets": ["Cracked iPhone screen."],
        },
        "other": {
            "definition": "Ambiguous, out of scope, or non-technical inquiries.",
            "example_tweets": ["Hello Apple!"],
        },
    }


TEST_SCENARIOS = [
    {
        "message": "My iPhone battery dies within 2 hours after charging to 100%.",
        "mock_intent": "battery_charging",
        "confidence": 0.94,
    },
    {
        "message": "I keep getting an error trying to install the latest iOS update.",
        "mock_intent": "software_update",
        "confidence": 0.91,
    },
    {
        "message": "My Wi-Fi keeps disconnecting and dropping the signal constantly.",
        "mock_intent": "connectivity_issue",
        "confidence": 0.93,
    },
    {
        "message": "My Apple ID is locked and I need to reset my account password.",
        "mock_intent": "account_access",
        "confidence": 0.95,
    },
    {
        "message": "I dropped my phone and the screen glass is completely shattered.",
        "mock_intent": "hardware_repair",
        "confidence": 0.89,
    },
]


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_run_agent_5_scenarios(scenario, agent_cfg, agent_taxonomy):
    """Verify run_agent returns all expected keys for 5 different intents without error."""
    msg = scenario["message"]
    mock_intent = scenario["mock_intent"]
    mock_conf = scenario["confidence"]

    def mock_llm(prompt: str, **kwargs) -> str:
        # If prompt is intent classification:
        if "available intents" in prompt.lower():
            return json.dumps({
                "intent": mock_intent,
                "confidence": mock_conf,
                "reasoning": f"Classified as {mock_intent}",
            })
        # If prompt is reply drafting:
        return "We'd like to help you with this. Could you let us know what iOS version your device is running?"

    mock_retrieved = [
        {
            "thread_id": "thread_123",
            "customer_message": "Similar issue description",
            "brand_resolution": "Try checking Settings > General to troubleshoot.",
            "intent": mock_intent,
            "similarity_score": 0.85,
        }
    ]

    with patch("src.agent.retrieve", return_value=mock_retrieved):
        res = run_agent(
            customer_message=msg,
            index=None,  # Not accessed due to mock
            corpus_meta=[],
            taxonomy=agent_taxonomy,
            cfg=agent_cfg,
            llm_generate=mock_llm,
        )

    assert "input" in res
    assert "intent" in res
    assert "routing" in res
    assert "retrieved" in res
    assert "reply" in res
    assert res.get("error") is None

    # Check intent schema
    assert res["intent"]["intent"] == mock_intent
    assert res["intent"]["confidence"] == mock_conf

    # Check routing schema
    assert res["routing"]["action"] in ["auto_handle", "escalate"]
    assert res["routing"]["decided_by"] == "rule"

    # Check reply schema
    assert "draft" in res["reply"]
    assert "grounding_used" in res["reply"]
    assert "source_thread_ids" in res["reply"]
    assert "for_human_review" in res["reply"]
    assert len(res["reply"]["draft"]) <= 280
