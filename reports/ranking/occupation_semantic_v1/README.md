# Occupation semantic retrieval v1 (opt-in M4 dual encoder)

9 November 2026. **Update (same day): fine-tuned + cosine RRF fusion (`configs/ranking/occupation_semantic_fusion_v1.json`) is now the live default by explicit owner decision; see [default decision](../occupation_semantic_default/DECISION.md).** The text below describes the experiment as evaluated. Implementation started by Codex (8 November: preparation, training, indexing, runtime wiring); frozen evaluation, acceptance, leakage-test correction and this report completed 9 November.

## What it is

`occupation_dual_encoder` is registered as an M4 algorithm with variants `pretrained_v1` and `finetuned_v1` (`recommender/occupation_semantic.py`). A shared-weight MiniLM-L6 encoder (`sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243`) embeds the newline-joined chronological **user-only** text (assistant turns excluded, first 256 wordpieces) and scores all 378 eligible occupations by cosine against a precomputed L2-normalised occupation index. It consumes no relevance labels, RIASEC values or other component outputs. Fusion is the existing equal-weight RRF k=60 in the composition layer.

Semantic scoring runs for any nonempty text, including background-only statements with zero extracted preferences or RIASEC dimensions; it does not inherit the preference extractor's gate. Empty text returns `unusable`; a missing checkpoint/index returns `unavailable` with the exact path. Load-time validation rejects index/checkpoint, catalog, split-data, source-data or implementation-hash mismatches.

**Cosine here is topic/background association, not a career preference or probability.** Nonsense text still yields a full ranking (see acceptance cases); a nonempty ranking does not establish relevance.

## Data and leakage controls

| Item | Value |
|---|---|
| Sources | `data/occupations.csv`, `data/tasks.csv`, `data/work_context/onet_31_0/prepared/occupation_work_context.jsonl` (hashes in `split_manifest.json`) |
| Source units | 29,074 (8,243 task, 20,831 context) → 7,500 unique statements |
| Context text | element name + modal reported category description; numeric means never verbalised; 1,764 suppressed rating rows excluded |
| Grouping | transitive char 3–5-gram TF-IDF cosine ≥ .85; groups indivisible across splits |
| Split (groups) | train 3,470 / validation 1,598 / test 1,133 / occupation-held-out 903 (38 occupations) |
| Fine-tuning used | 768 train groups (`max_train_groups` bound), 3 epochs, CPU |
| dev_profile | overlap audit only (0 overlapping groups); never training data |
| Index text | title 24 / description 64 / tasks 112 / context 48 wordpiece budgets; all non-train sources and near copies removed. Truncation: tasks 272/378, description 46/378, context 1/378 |

Leakage check (`tests/test_occupation_semantic.py`): **no test query appears, even as a substring, in either index.** Three short non-test statements are substrings of distinct, longer training tasks ("Serve on committees." ⊂ "Serve on committees to promote…"): two validation (already used for selection) and one held-out diagnostic query ("Participate in campus and community events."). Excluding the held-out case leaves held-out MRR unchanged to three decimals. The test pins these three so any new leak fails.

The test split is effectively **task-only** (1,130 task groups, 3 context groups): context statements are shared across many occupations and collapse into few, large groups. Context retrieval is therefore not evaluated meaningfully here.

## Frozen results (test opened once, 9 November)

Binary multi-owner task-ownership proxy (mean 1.1 owners/query), full 378 depth, same pool and queries for all modes. Control receives RIASEC read by DeBERTa from the task sentence (`no_signal` on 24/1,133). **This is formal task retrieval, not conversational career relevance.**

| Split | Mode | MRR | Hit@1 | Hit@5 | Hit@10 | NDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| test (1,133) | RIASEC cosine control | .032 | .004 | .033 | .064 | .028 | .268 |
| | pretrained | .483 | .350 | .635 | .745 | .535 | .908 |
| | **fine-tuned** | **.518** | **.379** | **.679** | **.775** | **.571** | **.933** |
| | fine-tuned + cosine RRF | .201 | .084 | .286 | .486 | .247 | .889 |
| occupation held-out (903) | control | .055 | .019 | .061 | .084 | .035 | .243 |
| | pretrained | .364 | .260 | .482 | .570 | .399 | .732 |
| | **fine-tuned** | **.418** | **.296** | **.548** | **.653** | **.458** | **.794** |
| | fusion | .174 | .063 | .257 | .443 | .204 | .780 |

Paired group bootstrap (2,000 resamples, seed 425), MRR differences with 95% CI:

- fine-tuned − pretrained: test **+.035 [+.020, +.049]**; held-out **+.054 [+.040, +.067]**
- fusion − fine-tuned: test −.317 [−.340, −.295]; held-out −.244 [−.269, −.219]
- fusion − control: test +.169 [+.155, +.182]

Validation (selection only): pretrained .489 → epoch 3 .519. Train-catalog lookup MRR .696 (includes training text; not generalisation) — a visible train/validation gap.

Interpretation: fine-tuning gives a small, consistent gain, including on occupations never seen in training. RRF with RIASEC cosine hurts on this proxy, as expected: a single task sentence carries little RIASEC signal, and equal-weight RRF lets that weak ranking pull semantic hits down. This does **not** show fusion is worse for real conversations, where RIASEC accumulates over turns; that needs conversational labels.

### Error analysis

Most fine-tuned misses at Hit@10 retrieve plausible occupations the proxy labels count as wrong, because each task statement has typically one owner: "Analyze existing development or manufacturing procedures and suggest improvements" (gold Mechatronics Engineers) → Manufacturing/Industrial/Chemical Engineers; "Analyze competitive market strategies…" (gold Business Intelligence Analysts) → Market Research Analysts, Marketing Managers. Metrics therefore understate topical relevance and cannot distinguish good from bad among plausible neighbours.

Targeted acceptance cases show expected limitations of contrastive retrieval:

- **Negation is not handled:** "I do not want to teach mathematics." ranks Mathematical Science Teachers first (same as the positive sentence). Preference polarity belongs to the preference branch, not this component.
- **Contradiction** ("I enjoy laboratory work but I do not want laboratory work.") returns laboratory occupations.
- **Vague/nonsense** text returns confident-looking rankings (designers, intelligence analysts).
- **Background only** works as intended: "My degree is in chemistry and I have experience analysing laboratory samples." → Chemists, Chemistry Teachers, Biochemists, with 0 extracted preferences; "I studied information systems, cloud computing and artificial intelligence." → Computer and Information Research Scientists, Computer Science Teachers, Computer and Information Systems Managers. RIASEC control for both returns management occupations from the one-dimensional I tie.

XGBoost reranking with semantic features is **deferred**: the existing preference XGBoost schema cannot consume these features, and no career-relevance labels exist; training on task-ownership labels would not measure career-preference quality.

## Acceptance evidence (`acceptance.json`, 9 November)

Actual local models; no mocks.

- **Component/pipeline:** 9 cases × 4 modes. For every semantic mode, pipeline per-occupation semantic score/rank equals direct component invocation, and a repeated pipeline call reproduces the ranking exactly. Background/no-preference cases have 0 preference records yet rank. One-positive-I input gives 378 distinct semantic scores (no code-sorted ties).
- **HTTP (real uvicorn TCP server, System + DeBERTa/NLI/T5):** pretrained, fine-tuned (twice, restarted), fusion. 3-turn conversation: `user_sentences` equals the cumulative user turns at each turn; M4 status `usable`; five recommendation cards at the end. Cold start ≈ 9.1 s; warm turn 0.13–0.26 s.
- **Terminal vs web:** `app.ranking_terminal --text … --json` top 100 equals the web ranking trace for the same first turn, in every mode.
- **Restart:** fine-tuned ranking identical across server restarts.
- **Browser:** **pending** — no browser automation was used. Manual steps below.
- **Unit tests:** `python -m unittest discover -s tests -k occupation_semantic` (leakage, masking, index compatibility, statuses, preference-independent dispatch) pass; the full suite has only the pre-existing errors listed in the default decision.

Pre-existing, unrelated: the old M4 three-way dense cache is stale in the current environment and fails with the original HEAD registry too (`preexisting_m4_cache_check.json`); old M4 files are unchanged.

## Reproduce (PowerShell, repository root)

```powershell
$env:HF_HUB_OFFLINE = '1'; $env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pip install -r requirements-occupation-semantic.txt
# Each stage refuses to overwrite existing outputs; use a new config/version to rerun.
.\.venv\Scripts\python.exe -m training.occupation_semantic prepare  --config configs/occupation_semantic_v1.json
.\.venv\Scripts\python.exe -m training.occupation_semantic train    --config configs/occupation_semantic_v1.json --smoke
.\.venv\Scripts\python.exe -m training.occupation_semantic train    --config configs/occupation_semantic_v1.json   # --resume to continue
.\.venv\Scripts\python.exe -m training.occupation_semantic index    --config configs/occupation_semantic_v1.json
.\.venv\Scripts\python.exe -m training.occupation_semantic evaluate --config configs/occupation_semantic_v1.json   # test opened once
.\.venv\Scripts\python.exe scripts/check_occupation_semantic.py --port 8017
```

Checkpoints and indexes live in `artifacts/ranking/occupation_semantic_v1/` (gitignored, as is `per_query.json`).

## Try it

Terminal (independent entries; modes `control`, `pretrained`, `finetuned`, `fusion`):

```powershell
.\.venv\Scripts\python.exe -m app.ranking_terminal --terminal-config configs/ranking_terminal/occupation_semantic_finetuned_v1.json
# one-shot, machine-readable:
.\.venv\Scripts\python.exe -m app.ranking_terminal --terminal-config configs/ranking_terminal/occupation_semantic_fusion_v1.json --text "I studied information systems, cloud computing and artificial intelligence." --json --top-k 10
```

Website:

```powershell
$env:CM_CONFIG = 'configs/system.occupation-semantic-finetuned-v1.json'   # or -pretrained-, -fusion-, -control-
.\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8010
```

Manual browser check: open http://127.0.0.1:8010/, start a researcher (debug) session, send "My degree is in chemistry and I have experience analysing laboratory samples.", then "I also studied statistics.", then "I have no additional preferences." Expect: researcher diagnostics show `occupation_semantic` with M4 `usable`, the fine-tuned checkpoint/index identity, `xgboost_invoked: false`, and zero preference records; the final five cards carry the semantic-association explanation ("Related to the topics and background you described; this does not establish a career preference."). The policy still asks RIASEC probing questions in between, and the final reply wording still says "based on the preferences you described" — a policy-text mismatch not changed in this increment. Do not expect one specific correct occupation.
