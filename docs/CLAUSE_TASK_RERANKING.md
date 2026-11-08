# Sentence-to-task nearest-neighbour reranking (`clause_tasks_v1`)

## What changes

`context_tasks_v1` compares the *whole* user history with *whole* occupation
documents (description + all tasks, 224-token chunks, mean chunk cosine). One
strong match is diluted on both sides. This treatment instead matches each user
sentence against each individual O*NET task statement:

```
for each candidate occupation c (dot-product top 100, unchanged)
    for each user sentence u           (frozen M4 regex splitter)
        match(u, c) = max over c's tasks t of cos(u, t)
score(c) = mean over sentences of sign(u) * match(u, c)
```

Candidates are then sorted by `score` inside the original dot-product groups of
10 (ties: ascending code). Ranks 101-378 and all structured scores are untouched.
The formula is M4's `minilm@mean_tasks_v1` aggregation; the new element is using
it as a second-stage reranker rather than standalone or through RRF.

| Variant key | Encoder | `sign(u)` |
|---|---|---|
| `minilm_groups10` | pinned MiniLM-L6-v2 | always +1 |
| `mpnet_groups10` | pinned MPNet-base-v2 | always +1 |
| `minilm_groups10_signed` | pinned MiniLM-L6-v2 | -1 for explicit dislikes |
| `mpnet_groups10_signed` | pinned MPNet-base-v2 | -1 for explicit dislikes |

`sign` uses the existing conservative `preference()` rule from
`recommender/utterance/context_matching.py`. Only `explicit_negative` flips;
unknown, mixed or uncertain sentences stay +1, so signed and unsigned variants
differ in exactly one variable.

Code: `recommender/semantic/clause_tasks.py` (scorer and frozen configurations),
wired through `SemanticMatchingPipeline`. Saved selections:
`configs/ranking/clause_tasks_<variant key>_v1.json`. Earlier variants, the
default ranking and `context_tasks_v1` are unchanged.

## Measuring it

```powershell
.\.venv\Scripts\python.exe -m recommender.experiments.clause_tasks
```

This reruns a fresh dot-product control (and checks it reproduces the recorded
metrics), runs all four treatments through the unchanged M2 evaluator on the
120 development/oracle profiles, and writes `reports/ranking/clause_tasks_v1/`:

- per treatment: predictions, per-profile metrics, manifest, latency (standard M2 artifacts);
- `comparison_vs_weighted_dot_product.json` for every treatment;
- `comparison_vs_context_tasks_v1.json` (unsigned) or `comparison_vs_unsigned.json` (signed),
  the single-variable control;
- `README.md` with all tables, generated from those artifacts;
- four rows appended to `reports/ranking/semantic_experiments_consolidated.csv`
  (pass `--no-csv` to skip).

It refuses to overwrite a completed run. The permanent ten metrics are unchanged.
The report adds **reranker diagnostics**, derived only from saved target ranks:

| Diagnostic | Meaning |
|---|---|
| Hit@5 ceiling | Share of profiles whose target is in the dot-product top 10. Groups of 10 preserve membership, so no reranker can exceed this Hit@5. |
| First-block top-5 / top-1 conversion | Among those profiles, how often the reranker puts the target in the top 5 / at rank 1. Isolates second-stage quality from the first stage. |
| Mean within-block position | Target position 1-10 inside its group, for targets in the top 100. Random ordering expects 5.5; lower is better. |
| Moved up / down / unchanged, mean rank change | Direction of target movement versus the dot product. |

## Trying it manually

```powershell
.\.venv\Scripts\python.exe -m app.ranking_terminal --reranking clause-minilm
```

Choices: `clause-minilm`, `clause-mpnet`, `clause-minilm-signed`,
`clause-mpnet-signed`. The shortcut builds exactly the evaluated configuration.
For each of the top 10 occupations the terminal prints the dot-product rank, the
final rank and the direction moved, and every sentence with its best-matching
task and cosine; dislikes are flagged `[dislike: subtracts]` in signed variants.

The terminal ranks each entry independently from extractor RIASEC values (not
oracle values), and its ranking sentences come from the M4 splitter, not the
Stanza clauses shown in the classifier section below. The first entry encodes
the shortlist's task statements (slower with MPNet); later entries reuse them.
Model locations default to the existing local snapshots; override with
`CONTEXT_TASK_MINILM_MODEL` / `CONTEXT_TASK_MPNET_MODEL`.

## Result (8 November 2026)

All four treatments regress. Versus the dot product: Hit@5 .133–.158 (control .167),
MRR .101–.116 (control .142), NDCG@10 .176–.178 (control .196). Versus the
matching chunk-mean `context_tasks_v1` variants they are also lower; MPNet's Hit@5
difference (-.075, CI [-.142, -.008]) excludes zero. Explicit-dislike signing
changes little (MPNet +.025 Hit@5 over unsigned, CI spans zero). The mean
within-group target position is 5.83–6.06, worse than random ordering (5.5).
A single best-matching task appears to reward occupations with one generically
similar task, which removes the difference between candidates. Full tables are in
`reports/ranking/clause_tasks_v1/README.md`.

## Limits

Development/oracle evaluation only. Cosine similarity is topical association,
not suitability. The preference rule misses mixed or indirect dislikes (e.g.
"What I don't like is routine admin like filing forms" is unknown, so unsigned).
