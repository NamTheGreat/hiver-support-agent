# Implementation Plan: Hiver Customer Support Agent (AppleSupport)

A minimalist, high-rigor customer support AI agent for Twitter/X customer support threads, evaluated on the AppleSupport slice of the Kaggle Customer Support on Twitter dataset.

Built with strict adherence to **Ponytail** engineering principles: standard library and native primitives first, zero framework bloat (no LangChain/LlamaIndex), auditable rules-based routing, reproducible evaluation, and deep failure analysis.

---

## User Review Required

> [!IMPORTANT]
> **API Key & Offline Execution**: The pipeline is designed to use OpenAI-compatible models (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`). However, per prompt requirements, `python run_all.py` must reproduce all numbers from scratch on `data/sample_subsample.csv` alone with no paid API required by default. We will implement `src/llm.py` to seamlessly connect to the OpenAI API when credentials are provided, while including a deterministic local fallback mechanism for offline/grader execution.

> [!IMPORTANT]
> **Git Policy Enforcement**: Per system user rules, `git commit` and `git push` are never executed by the assistant unless explicitly requested in a dedicated prompt. We will create and track all files (including `data/sample_subsample.csv`) ready for user git operations.

---

## Proposed Phases & Architecture

```
                                      [ Customer Tweet ]
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │   src/intents.py  │  SentenceTransformers
                                    │ Intent Classifier │  (MiniLM) + Few-shot LLM
                                    └─────────┬─────────┘
                                              │
                         ┌────────────────────┴───────────────────┐
                         ▼                                        ▼
               ┌───────────────────┐                    ┌───────────────────┐
               │   src/router.py   │                    │  src/retrieve.py  │
               │  Auditable Rules  │                    │   FAISS Index     │
               │  (5 Typed Reasons)│                    │ (Top-3 Cosine IP) │
               └─────────┬─────────┘                    └─────────┬─────────┘
                         │                                        │
                         └────────────────────┬───────────────────┘
                                              ▼
                                    ┌───────────────────┐
                                    │    src/reply.py   │  Grounding Context
                                    │   Reply Drafter   │  ≤ 280 chars, brand voice
                                    └─────────┬─────────┘
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │   src/agent.py    │  Unified Output Object
                                    │ Orchestrator Exec │  (Draft + Routing Action)
                                    └───────────────────┘
```

### Phase Breakdown

1. **Phase 1 — Data Preparation (`src/prepare_data.py`)**:
   - Ingest raw `data/twcs.csv` or `data/sample_subsample.csv`.
   - Filter AppleSupport interactions, reconstruct multi-turn dialogue trees via `in_response_to_tweet_id`, cap at 10 turns.
   - Clean texts (strip leading @mentions, URLs, normalize whitespace, keep casing).
   - Filter noise: remove bare "please DM us" redirects, non-English tweets (`langdetect`), empty customer queries.
   - Extract sample subsample (5,000–10,000 tweets) for grader standalone execution. Output `data/apple_threads.jsonl`.

2. **Phase 2 — Intent Taxonomy (`src/intents.py`)**:
   - Message embeddings via `all-MiniLM-L6-v2` with SHA-256 disk caching under `.cache/`.
   - Cluster first with HDBSCAN; if noise > 30%, fall back to k-means across range `[5, 8]` selected by silhouette score.
   - LLM cluster titling (snake_case + 1-line definition) using top-10 central exemplars; merge similar clusters (cosine > 0.85); append `"other"`.
   - Save `data/intent_taxonomy.json`.
   - Intent classification function with calibrated confidence bands (≥0.90 unambiguous, 0.60–0.89 plausible, <0.60 ambiguous/other).

3. **Phase 3 — Golden Evaluation Set (`eval/build_golden.py`)**:
   - Stratified sampling (180 target examples) across intents + 20% intentionally hard/ambiguous cases (<20 chars or confidence <0.65).
   - Calendar spread across ≥3 months; strictly disjoint from taxonomy clustering tweets.
   - Export `data/golden/golden_eval.jsonl` with `{id, tweet_id, customer_message, intent, escalate, escalation_reason, reference_reply_ok, labeller_note}`.
   - `data/golden/labelling_note.md` detailing methodology and candidate-labeller biases.

4. **Phase 4 — Retrieval System (`src/retrieve.py`)**:
   - Resolution corpus: paired `(customer_message, brand_resolution)` from historical resolved threads (min resolution len 20 chars).
   - FAISS `IndexFlatIP` over L2-normalized embeddings (cosine similarity).
   - Metadata storage in JSON; retrieval threshold filtering (`min_similarity=0.40`). Log retrieved IDs at `INFO` level.

5. **Phase 5 — Router (`src/router.py`)**:
   - 100% auditable rule-based routing with typed enum:
     - `LOW_CONFIDENCE`: classifier confidence < 0.80.
     - `OTHER_INTENT`: intent is `"other"`.
     - `HIGH_RISK_INTENT`: high-risk intent + safety keywords (hacked, stolen, fraud, legal, lawsuit).
     - `NEGATIVE_SENTIMENT`: VADER compound score < -0.5.
     - `NO_GROUNDING`: retrieval returned zero similar threads.
   - Conservative fallback: any intent outside allowlist escalates with `HIGH_RISK_INTENT`.

6. **Phase 6 — Reply Drafter (`src/reply.py`)**:
   - Apple Support persona prompt: empathetic, concise, professional, ≤280 chars, no hallucinations or false promises.
   - Incorporates top-3 retrieved resolutions. If ungrounded, explicitly prompts for generic safe guidance.

7. **Phase 7 — Agent Orchestrator (`src/agent.py`)**:
   - Chains: `classify -> route -> retrieve -> draft`.
   - Always retrieves and drafts even if escalated (providing human assist draft with `for_human_review: true`).
   - Resilient exception handling; never crashes the pipeline.

8. **Phase 8 — Baselines (`eval/baselines.py`)**:
   - **Trivial**: Majority-class intent predictor + fixed static apology link + always escalates (`LOW_CONFIDENCE`).
   - **Simple**: TF-IDF (1-2 ngrams) + LogisticRegression (trained strictly on non-golden retrieval corpus) + verbatim top-1 historical reply + Phase 5 routing rules. Persisted in `.cache/`.

9. **Phase 9 — Evaluation Harness (`eval/harness.py`)**:
   - Runs Trivial, Simple, and Agent on identical golden dataset.
   - Computes: Intent Accuracy, Macro-F1, per-class P/R/F1, Routing Accuracy/F1 and breakdown by reason, Judge metrics.
   - Saves `reports/results/{system}_metrics.json` and detailed per-example CSV.

10. **Phase 10 — LLM-as-Judge (`eval/judge.py`)**:
    - 4-dimension rubric (1–5):
      1. Correctness (issue resolution)
      2. Groundedness (consistency with historical resolutions)
      3. Tone (Apple brand voice, empathy)
      4. Actionability (concrete next steps)
    - Returns JSON with composite score and 1-sentence critique on the weakest dimension.

11. **Phase 11 — Human-Judge Agreement Calibration (`eval/judge_human_agreement.py`)**:
    - Hand-score 50 golden examples in `eval/human_scores_50.json`.
    - Calculate Pearson $r$, Spearman $\rho$, exact match %, and near-exact ($\le 1$) agreement.
    - Deep analysis of top 2–3 divergence cases. Save to `reports/judge_agreement.json`.

12. **Phase 12 — One-Command Reproduction (`run_all.py` & `Makefile`)**:
    - End-to-end execution script with wall-clock timing per stage; target < 15 minutes on subsample.
    - Generates visual figures (`reports/figures/`) for confusion matrix, radar chart, baseline comparisons.

13. **Phase 13 — Unit & Integration Testing (`tests/`)**:
    - `test_router.py`: Unit tests covering every single routing rule and escalation reason.
    - `test_intents.py`: Taxonomy validation, JSON schemas, confidence bounds.
    - `test_end_to_end.py`: End-to-end flow over 5 distinct customer scenarios with mocked LLM/retrieval.
    - Zero network, zero paid API required; passes `pytest tests/ -x`.

14. **Phase 14 — Rigorous Documentation & Reports**:
    - `reports/REPORT.md`: ≤ 6 pages covering Problem Framing, Method, Results Table, Judge Validation, Failure Analysis, "What is misleading about my headline number?", Next Steps.
    - `reports/decision_log.md`: 10–15 rationale bullets for every technical trade-off.
    - `reports/failure_analysis.md`: Top 5 failure cases with actual tweets, failure mode, and hypotheses.
    - `reports/dev_log.md`: Chronological log for each phase (Tried / Result / Errors / Next).
    - `README.md`: Reproduction instructions, performance numbers, runtime benchmarks, citations.

---

## Verification Plan

### Automated Tests
- `pytest tests/ -x`: Run completely offline with mocked endpoints to verify router rules, intent classification, and end-to-end orchestration.
- `python run_all.py`: Execute full reproduction pipeline end-to-end, measuring runtime and verifying output JSON/CSV metrics.

### Manual Inspection & Sanity Checks
- Verify `data/sample_subsample.csv` contains valid reconstructed threads for AppleSupport.
- Check `data/intent_taxonomy.json` for semantic clarity and non-overlapping cluster boundaries.
- Inspect `eval/human_scores_50.json` vs LLM judge scores to ensure authentic statistical correlation.
- Review all generated figures in `reports/figures/`.
