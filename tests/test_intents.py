"""Unit tests for src/intents.py.

Verifies intent classification schema, taxonomy key validation, confidence bounds [0, 1],
and mock generation without network or paid APIs.
"""

import json
import pytest
from src.intents import classify_intent


@pytest.fixture
def mock_taxonomy():
    """Mock taxonomy fixture with standard categories and 'other'."""
    return {
        "battery_charging": {
            "definition": "Issues related to battery life, battery drain, or charging problems.",
            "example_tweets": ["My battery drops from 100% to 20% in an hour."],
        },
        "software_update": {
            "definition": "Issues with updating iOS, installation errors, or post-update bugs.",
            "example_tweets": ["Cannot update to iOS 17."],
        },
        "connectivity_issue": {
            "definition": "Wi-Fi, Bluetooth, or cellular network connection problems.",
            "example_tweets": ["Wi-Fi keeps dropping."],
        },
        "other": {
            "definition": "Ambiguous, multi-intent, or out of scope inquiries.",
            "example_tweets": ["What is Apple's address?"],
        },
    }


def test_classify_intent_valid_mock(mock_taxonomy):
    """Test classify_intent when LLM returns well-formed JSON."""
    def mock_llm(prompt: str, **kwargs) -> str:
        return json.dumps({
            "intent": "battery_charging",
            "confidence": 0.94,
            "reasoning": "Explicit mention of battery drain.",
        })

    res = classify_intent("My phone battery dies instantly.", mock_taxonomy, llm_generate=mock_llm)

    assert "intent" in res
    assert "confidence" in res
    assert "reasoning" in res
    assert res["intent"] in mock_taxonomy
    assert isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["intent"] == "battery_charging"
    assert res["confidence"] == 0.94


def test_classify_intent_out_of_taxonomy_fallback(mock_taxonomy):
    """Test fallback when LLM invents an intent not present in the taxonomy."""
    def mock_llm(prompt: str, **kwargs) -> str:
        return json.dumps({
            "intent": "non_existent_invented_category",
            "confidence": 0.98,
            "reasoning": "Made up category.",
        })

    res = classify_intent("Random query", mock_taxonomy, llm_generate=mock_llm)

    assert res["intent"] in mock_taxonomy
    assert res["intent"] == "other"
    assert 0.0 <= res["confidence"] <= 1.0


def test_classify_intent_invalid_json_fallback(mock_taxonomy):
    """Test graceful handling when LLM returns malformed or non-JSON output."""
    def mock_llm(prompt: str, **kwargs) -> str:
        return "I am an AI and I think this is about battery charging."

    res = classify_intent("My battery is dying fast.", mock_taxonomy, llm_generate=mock_llm)

    assert "intent" in res
    assert "confidence" in res
    assert res["intent"] in mock_taxonomy
    assert 0.0 <= res["confidence"] <= 1.0


def test_classify_intent_confidence_clamping(mock_taxonomy):
    """Test confidence clamping when LLM returns values outside [0.0, 1.0]."""
    def mock_llm_high(prompt: str, **kwargs) -> str:
        return json.dumps({"intent": "software_update", "confidence": 1.5, "reasoning": "Overconfident"})

    def mock_llm_low(prompt: str, **kwargs) -> str:
        return json.dumps({"intent": "software_update", "confidence": -0.5, "reasoning": "Negative"})

    res_high = classify_intent("Update error", mock_taxonomy, llm_generate=mock_llm_high)
    assert res_high["confidence"] <= 1.0

    res_low = classify_intent("Update error", mock_taxonomy, llm_generate=mock_llm_low)
    assert res_low["confidence"] >= 0.0
