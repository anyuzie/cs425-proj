# Internal stance-aware M4 experiment

Implementation follows the authorized stance v3 instruction sheet. Stance is inferred at runtime inside the new independent `minilm_stance` algorithm, solely from raw `ComponentInput.user_sentences`. No dataset stance properties or evaluation annotations are used. `minilm@mean_tasks_v1`, cosine M1, the metric definitions and production defaults are preserved.

Variants:

- `positive_internal_v1`: explicit positive activity phrases only.
- `stance_internal_v1`: separate positive/negative means; score = positive - negative, fixed lambda=1.

Both variants share the pinned model, validated corpus cache and maximum clipped task-cosine matching. See [PROTOCOL.md](PROTOCOL.md) for the prespecified aggregation, empty-evidence convention and equal-weight RRF k=60 fusion. Rules are conservative and incomplete; no independent stance labels were supplied, so polarity quality is unverified. Generic task evidence remains provisional; explicit per-phrase polarity/task traces are separate report outputs.

Run from the repository root after restoring the intended **120-record canonical** `data/dev_profiles.jsonl`:

```powershell
.\.venv\Scripts\python.exe scripts/run_stance_experiment.py --audit-only
.\.venv\Scripts\python.exe scripts/run_stance_experiment.py
```

The runner checks the record count and immutable dataset hash, produces review-only extraction artifacts, and executes five controlled conditions. Complete evaluations append once to the central history and clean notebook history. Fresh report directories are required; existing runs are never overwritten. `--expected-count` may be changed only when the user explicitly authorizes a different cohort. No alternative or modified stance dataset is read, searched or substituted.

Evaluation is complete on the restored **120-profile canonical dataset**, whose fingerprint exactly matches the original dot_vs_cosine_dev120 and combinations_dev120 experiments. All five conditions are registered once in both histories. See [DECISION.md](DECISION.md), [summary.csv](summary.csv), [comparisons.json](comparisons.json), and [PRIOR_DATASET_AUDIT.md](PRIOR_DATASET_AUDIT.md).

Extraction covers 97/120 profiles (90 with positives, 38 with negatives, 31 with both); signed stance has 23 unusable queries. No independent review labels are available, so polarity quality remains unverified. The blank review template is never read by inference. Frozen semantic scores and ranks reproduce all 45,360 occupation/profile results from the earlier matching cosine+dense experiment exactly.

Validation: 8 new stance tests, 13 dense tests, 10 architecture tests and 7 configuration tests passed (38 total). A first text inspection exposed a background/causal-clause scope issue, corrected by segmenting causal clauses before preference extraction. Exact same-activity contradictions are withheld; paraphrased contradictions and omitted implicit preferences remain limitations requiring independent review. The source is not a trained stance model and must not be treated as verified human annotation.
