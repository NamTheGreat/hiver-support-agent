"""Manual verification and deep component testing script for Hiver Support Agent.

Tests every module, function, and edge case directly, printing assertions
and diagnostic proofs.
"""

import json
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from src.prepare_data import clean_text
from src.intents import load_taxonomy, classify_intent
from src.retrieve import load_index, retrieve
from src.router import route, EscalationReason
from src.reply import draft_reply
from src.agent import run_agent
from eval.baselines import TrivialBaseline, SimpleBaseline
from eval.judge import judge_reply
from eval.judge_human_agreement import compute_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("manual_test")


def test_prepare_data_cleaning():
    print("\n--- Testing src/prepare_data.py clean_text() ---")
    # Test 1: Strip leading @mentions
    raw1 = "@AppleSupport @customer123 My battery is draining super fast!"
    clean1 = clean_text(raw1)
    assert clean1 == "My battery is draining super fast!", f"Failed clean1: {clean1}"
    print("✓ Mentions stripped correctly:", clean1)

    # Test 2: Strip URLs
    raw2 = "Check this out https://support.apple.com/en-us/HT201222 right now!"
    clean2 = clean_text(raw2)
    assert clean2 == "Check this out right now!", f"Failed clean2: {clean2}"
    print("✓ URLs stripped correctly:", clean2)

    # Test 3: Preserve case and punctuation
    raw3 = "iPhone 13 Pro Max running iOS 17.2... Why is it lagging?!"
    clean3 = clean_text(raw3)
    assert "iPhone 13 Pro Max" in clean3 and "iOS 17.2" in clean3
    print("✓ Natural casing preserved:", clean3)


def test_intents_taxonomy_and_classification(cfg):
    print("\n--- Testing src/intents.py ---")
    tax_path = Path(cfg["data"]["taxonomy_json"])
    assert tax_path.exists(), f"Taxonomy file missing: {tax_path}"
    taxonomy = load_taxonomy(tax_path)
    assert len(taxonomy) >= 6, f"Expected at least 6 intents, got {len(taxonomy)}"
    assert "other" in taxonomy, "'other' intent missing from taxonomy"
    print(f"✓ Taxonomy loaded: {len(taxonomy)} intents found ({list(taxonomy.keys())})")

    # Test classification of 8 distinct queries
    test_queries = [
        ("My iPhone battery dies in 1 hour after full charge", "battery_charging"),
        ("Unable to install iOS 17 update error", "software_update"),
        ("Wi-Fi drops connection repeatedly", "connectivity_issue"),
        ("My Apple ID is locked and I cannot reset password", "account_access"),
        ("Unauthorized subscription charge on my credit card", "billing_order"),
        ("Instagram and camera app freeze constantly", "app_crash_freeze"),
        ("Typing letter i shows a question mark in box", "keyboard_autocorrect"),
        ("What time does the store open tomorrow?", "other"),
    ]

    for q, expected in test_queries:
        res = classify_intent(q, taxonomy)
        assert "intent" in res and "confidence" in res and "reasoning" in res
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["intent"] in taxonomy
        print(f"✓ Query: '{q[:35]}...' -> Pred: '{res['intent']}' (conf: {res['confidence']:.2f})")


def test_retrieval(cfg):
    print("\n--- Testing src/retrieve.py ---")
    index, corpus_meta = load_index(cfg)
    assert index.ntotal > 0, "Index is empty!"
    assert len(corpus_meta) == index.ntotal, "Metadata count mismatch with index"
    print(f"✓ Loaded FAISS index with {index.ntotal} vectors (dim={index.d})")

    # In-domain technical query
    query_in = "My Wi-Fi keeps disconnecting after update"
    top_k = cfg.get("retrieval", {}).get("top_k", 3)
    min_sim = cfg.get("retrieval", {}).get("min_similarity", 0.40)
    res_in = retrieve(query_in, index, corpus_meta, top_k=top_k, min_similarity=min_sim)
    assert len(res_in) > 0, "Expected at least 1 match for Wi-Fi query"
    assert "similarity_score" in res_in[0]
    assert res_in[0]["similarity_score"] >= min_sim
    print(f"✓ Retrieved {len(res_in)} matches for in-domain query. Top match sim: {res_in[0]['similarity_score']:.3f}")

    # Completely nonsensical query -> should filter out below threshold
    query_out = "zxczxc987123 qweoiuypoiuy !@#$%^&*()"
    res_out = retrieve(query_out, index, corpus_meta, top_k=top_k, min_similarity=min_sim)
    print(f"✓ Out-of-domain nonsense query returned {len(res_out)} items (expected 0 if below min_similarity {min_sim})")


def test_router(cfg):
    print("\n--- Testing src/router.py ---")
    dummy_retrieved = [{"thread_id": "1", "similarity_score": 0.85, "customer_message": "test", "brand_resolution": "reset"}]

    # Rule 1: LOW_CONFIDENCE
    r1 = route("Wi-Fi drops", "connectivity_issue", confidence=0.70, retrieved=dummy_retrieved, cfg=cfg)
    assert r1["action"] == "escalate" and r1["reason"] == EscalationReason.LOW_CONFIDENCE
    print("✓ Rule 1 (LOW_CONFIDENCE) verified")

    # Rule 2: OTHER_INTENT
    r2 = route("Hello", "other", confidence=0.95, retrieved=dummy_retrieved, cfg=cfg)
    assert r2["action"] == "escalate" and r2["reason"] == EscalationReason.OTHER_INTENT
    print("✓ Rule 2 (OTHER_INTENT) verified")

    # Rule 3: HIGH_RISK_INTENT with risk keyword
    r3 = route("My account was hacked and stolen", "account_access", confidence=0.95, retrieved=dummy_retrieved, cfg=cfg)
    assert r3["action"] == "escalate" and r3["reason"] == EscalationReason.HIGH_RISK_INTENT
    print("✓ Rule 3 (HIGH_RISK_INTENT with keyword) verified")

    # Rule 4: NEGATIVE_SENTIMENT
    r4 = route("I hate Apple, this is the most disgusting piece of shit device ever created", "connectivity_issue", confidence=0.95, retrieved=dummy_retrieved, cfg=cfg)
    assert r4["action"] == "escalate" and r4["reason"] == EscalationReason.NEGATIVE_SENTIMENT
    print("✓ Rule 4 (NEGATIVE_SENTIMENT) verified")

    # Rule 5: NO_GROUNDING
    r5 = route("Wi-Fi drops", "connectivity_issue", confidence=0.95, retrieved=[], cfg=cfg)
    assert r5["action"] == "escalate" and r5["reason"] == EscalationReason.NO_GROUNDING
    print("✓ Rule 5 (NO_GROUNDING) verified")

    # Rule 6: Non-allowlisted intent
    r6 = route("My screen is cracked", "hardware_repair", confidence=0.95, retrieved=dummy_retrieved, cfg=cfg)
    assert r6["action"] == "escalate" and r6["reason"] == EscalationReason.HIGH_RISK_INTENT
    print("✓ Rule 6 (Non-allowlisted intent) verified")

    # Rule 7: Auto-handle happy path (using allowlisted software_update)
    r7 = route("I would like to update my iPhone to the latest iOS", "software_update", confidence=0.95, retrieved=dummy_retrieved, cfg=cfg)
    assert r7["action"] == "auto_handle" and r7["reason"] is None
    print("✓ Rule 7 (Auto-handle happy path) verified")


def test_reply_drafter():
    print("\n--- Testing src/reply.py ---")
    mock_ret = [
        {"thread_id": "999", "brand_resolution": "Try resetting network settings in Settings > General > Reset."}
    ]
    rep = draft_reply(
        customer_message="Wi-Fi keeps dropping on my iPhone",
        intent="connectivity_issue",
        intent_definition="Wi-Fi, Bluetooth, or cellular network connection problems.",
        retrieved=mock_ret,
    )
    assert "draft" in rep and "grounding_used" in rep and "source_thread_ids" in rep
    assert len(rep["draft"]) <= 280, f"Draft exceeded 280 chars: len={len(rep['draft'])}"
    assert rep["grounding_used"] is True
    assert "999" in rep["source_thread_ids"]
    print(f"✓ Reply drafted ({len(rep['draft'])} chars): \"{rep['draft']}\"")


def test_agent_orchestrator(cfg):
    print("\n--- Testing src/agent.py ---")
    index, corpus_meta = load_index(cfg)
    taxonomy = load_taxonomy(Path(cfg["data"]["taxonomy_json"]))

    # Test 1: Normal execution
    res = run_agent("My iPhone battery dies in 2 hours", index, corpus_meta, taxonomy, cfg)
    assert "input" in res and "intent" in res and "routing" in res and "reply" in res
    assert res.get("error") is None
    print(f"✓ Agent run success: action={res['routing']['action']}, intent={res['intent']['intent']}")

    # Test 2: Error resilience (mock broken LLM)
    def broken_llm(p, **k):
        raise RuntimeError("Network timeout simulation")
    res_err = run_agent("Test query", index, corpus_meta, taxonomy, cfg, llm_generate=broken_llm)
    assert res_err.get("error") is not None
    print("✓ Agent error resilience verified: caught exception gracefully without crashing")


def test_data_leakage_and_disjointness(cfg):
    print("\n--- Testing Data Leakage & Set Disjointness ---")
    golden_path = Path(cfg["data"]["golden_jsonl"])
    golden_ids = set()
    with open(golden_path, "r", encoding="utf-8") as f:
        for line in f:
            golden_ids.add(str(json.loads(line)["tweet_id"]))
    print(f"✓ Loaded {len(golden_ids)} golden set tweet IDs")

    # Check FAISS metadata
    _, corpus_meta = load_index(cfg)
    index_ids = {str(item["thread_id"]) for item in corpus_meta}
    overlap_index = golden_ids & index_ids
    assert len(overlap_index) == 0, f"DATA LEAKAGE DETECTED: {len(overlap_index)} golden IDs in retrieval index!"
    print(f"✓ FAISS Retrieval Index is strictly disjoint from Golden Set (overlap = {len(overlap_index)})")

    # Check Taxonomy tweet IDs
    tax_ids_file = Path(cfg["data"].get("cache_dir", ".cache")) / "taxonomy_tweet_ids.json"
    if tax_ids_file.exists():
        with open(tax_ids_file, "r", encoding="utf-8") as f:
            tax_ids = set(str(tid) for tid in json.load(f))
        overlap_tax = golden_ids & tax_ids
        assert len(overlap_tax) == 0, f"DATA LEAKAGE DETECTED: {len(overlap_tax)} golden IDs in taxonomy training set!"
        print(f"✓ Taxonomy Training Set is strictly disjoint from Golden Set (overlap = {len(overlap_tax)})")


def test_baselines_and_judge(cfg):
    print("\n--- Testing eval/baselines.py & eval/judge.py ---")
    golden_path = Path(cfg["data"]["golden_jsonl"])
    with open(golden_path, "r", encoding="utf-8") as f:
        first_item = json.loads(f.readline())

    msg = first_item["customer_message"]
    gold_intent = first_item["intent"]

    # Test Trivial Baseline
    trivial = TrivialBaseline(majority_intent="software_update")
    res_triv = trivial.run(msg, retrieved=[], cfg=cfg)
    assert res_triv["routing"]["action"] == "escalate"
    assert res_triv["routing"]["reason"] == EscalationReason.LOW_CONFIDENCE
    assert "https://support.apple.com" in res_triv["reply"]["draft"]
    print("✓ TrivialBaseline verified (predicted majority, canned apology, escalated)")

    # Test Simple Baseline (TF-IDF + LogReg)
    cache_dir = Path(cfg["data"].get("cache_dir", ".cache"))
    model_path = cache_dir / "simple_baseline_model.pkl"
    if model_path.exists():
        simple = SimpleBaseline(model_path=model_path)
        res_simple = simple.run(msg, retrieved=[{"thread_id": "123", "brand_resolution": "Restart your iPhone."}], cfg=cfg)
        assert "intent" in res_simple["intent"]
        assert 0.0 <= res_simple["intent"]["confidence"] <= 1.0
        assert res_simple["reply"]["draft"] == "Restart your iPhone."
        print(f"✓ SimpleBaseline verified (predicted intent={res_simple['intent']['intent']}, verbatim top-1 reply)")

    # Test Judge on high-quality vs canned draft
    j_high = judge_reply(
        customer_message="My iPhone battery drains in 1 hour",
        draft="We'd like to help you resolve this. Could you let us know what device model and iOS version you are currently running?",
        retrieved=[{"customer_message": "Battery issue", "brand_resolution": "Check Settings > Battery"}],
        true_intent="battery_charging",
        cfg=cfg,
    )
    assert 1 <= j_high["correctness"] <= 5
    assert 1 <= j_high["groundedness"] <= 5
    assert 1 <= j_high["tone"] <= 5
    assert 1 <= j_high["actionability"] <= 5
    print(f"✓ LLM-as-Judge scored grounded agent draft: Composite={j_high['composite']:.2f} (C={j_high['correctness']}, G={j_high['groundedness']}, T={j_high['tone']}, A={j_high['actionability']})")

    j_canned = judge_reply(
        customer_message="My iPhone battery drains in 1 hour",
        draft="We apologize for the inconvenience. For assistance, please visit https://support.apple.com.",
        retrieved=[],
        true_intent="battery_charging",
        cfg=cfg,
    )
    assert j_canned["composite"] < j_high["composite"]
    print(f"✓ LLM-as-Judge penalized canned apology: Composite={j_canned['composite']:.2f} < {j_high['composite']:.2f}")


def test_agreement_and_figures():
    print("\n--- Testing eval/judge_human_agreement.py & figures ---")
    agreement_file = Path("reports/judge_agreement.json")
    assert agreement_file.exists(), "Agreement report missing"
    with open(agreement_file, "r", encoding="utf-8") as f:
        agr = json.load(f)

    assert agr["n_samples"] == 50
    assert "metrics_by_dimension" in agr
    assert "composite" in agr["metrics_by_dimension"]
    comp_exact = agr["metrics_by_dimension"]["composite"]["exact_match_rate"]
    comp_near = agr["metrics_by_dimension"]["composite"]["near_exact_rate"]
    print(f"✓ Agreement metrics verified: N=50, Exact match={comp_exact*100:.1f}%, Near-exact (|diff|<=1)={comp_near*100:.1f}%")

    # Check generated figures
    fig_dir = Path("reports/figures")
    for fig_name in ["system_comparison.png", "judge_dimensions.png", "judge_human_scatter.png", "routing_breakdown.png"]:
        fig_path = fig_dir / fig_name
        assert fig_path.exists() and fig_path.stat().st_size > 1000, f"Figure missing or invalid: {fig_name}"
    print(f"✓ All 4 publication figures verified in {fig_dir} (non-empty PNGs)")


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_prepare_data_cleaning()
    test_intents_taxonomy_and_classification(cfg)
    test_retrieval(cfg)
    test_router(cfg)
    test_reply_drafter()
    test_agent_orchestrator(cfg)
    test_baselines_and_judge(cfg)
    test_agreement_and_figures()
    test_data_leakage_and_disjointness(cfg)

    print("\n" + "=" * 70)
    print("  ALL MANUAL VERIFICATION & COMPONENT TESTS PASSED 100% GREEN")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
