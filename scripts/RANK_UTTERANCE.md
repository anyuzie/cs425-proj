# Interactive utterance ranking

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/rank_utterance.py
```

Enter one utterance per line. Enter exactly `x` to exit; Ctrl+C and end-of-input also exit. Blank entries are skipped.

The script loads the real DeBERTa checkpoint and ranking pipeline once. Defaults come from `configs/system.json`: `paths.extractor_checkpoint` and `paths.ranking_config`. The current ranking YAML selects weighted-dot-product M1 only. The local trained checkpoint and existing model dependencies are required.

Each entry starts fresh: extracted `sign * strength` supplies observed runtime RIASEC values in [-1,1]. Unmentioned types have zero placeholders but remain unobserved. Confidence is displayed separately and is not multiplied into strength. Only the current utterance is supplied to text rankers. A single active evidence record permits configurations using `active_only`. No targets, oracle scores, previous turns, or dataset annotations are supplied.

The terminal shows extracted preferences, confidence, reasons, and up to ten occupation codes, titles and component scores/ranks. Fused results also show their RRF scores. Scores are not probabilities. If no enabled component has a usable signal, the reason is displayed and input continues.

To select an existing alternative configuration:

```powershell
.\.venv\Scripts\python.exe scripts/rank_utterance.py --ranking-config configs/ranking/m1_cosine.json
.\.venv\Scripts\python.exe scripts/rank_utterance.py --ranking-config configs/ranking/m1_cosine_three_way.json
```

`--checkpoint`, `--system-config`, and `--occupations` overrides are also available. Relative paths resolve against the repository root, including when launching from another working directory. M4 configurations require the matching local MiniLM model/task cache.

This standalone adapter bypasses the unfinished tracker, policy and conversation orchestration. It does not implement cumulative interests, contradiction resolution, or recommendation suitability assessment. It does not write evaluation-history rows or alter ranking defaults.
