# Labelling Methodology and Dataset Notes

## Sampling Strategy
The golden evaluation dataset comprises 180 customer inquiries sampled from the reconstructed AppleSupport Twitter threads. To avoid data contamination and temporal retrieval leakage, the sampling procedure enforced three strict constraints:
1. **Disjoint Partitioning:** Zero tweet IDs in the golden set overlap with tweets used for empirical taxonomy clustering.
2. **Temporal Stratification:** Candidate inquiries span three distinct calendar months (October, November, and December 2017).
3. **Stratified Difficulty:** Approximately 80% of samples reflect well-formed inquiries distributed across the discovered taxonomy intents, while 20% are intentionally selected edge cases (<20 characters, highly ambiguous queries, or multi-issue complaints).

## Labelling Procedure
All labels were assigned directly by the candidate using `data/intent_taxonomy.json` as the authoritative guideline. For each query, the primary intent was assigned based on root cause. The escalation decision (`yes`/`no`) was annotated according to whether the issue can be safely resolved with autonomous technical guidance without human oversight. Inquiries involving high-risk keywords (such as 'hacked', 'stolen', 'unauthorized'), billing transactions, or out-of-scope 'other' requests were marked for escalation.

## Known Biases and Limitations
Because this dataset was created and reviewed by a single candidate labeller, inter-annotator agreement metrics (such as Cohen's Kappa) are not reported and represent valuable future work. Furthermore, the deliberate over-representation of ambiguous and terse customer queries (20% of the set) creates an artificially challenging distribution compared to live production volume, where straightforward frequently asked questions occur with higher base rates.
