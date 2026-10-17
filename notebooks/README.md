# Ranking diagnostic notebook

Open [ranking_diagnostic_analysis.ipynb](ranking_diagnostic_analysis.ipynb). It is saved with executed tables, interpretations and rendered charts for all A1-D4 sections. The analysis uses 60 development profiles, the 378-occupation pool, seven matching saved M4 evaluations, and the existing pinned MiniLM cache. It reads no held-out profiles and changes no ranking implementation, metric, previous experiment or history row.

## Re-execute from the repository root

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-diagnostics.txt
python -m ipykernel install --prefix .venv --name ranking-diagnostics --display-name "Ranking diagnostics"
```

Select that environment in your notebook editor and run all cells with the repository root as the working directory. In VS Code, set the notebook file root to the workspace folder if it otherwise starts in `notebooks/`.

Alternatively, execute headlessly from the repository root:

```python
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("notebooks/ranking_diagnostic_analysis.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook, kernel_name="ranking-diagnostics", timeout=600,
    resources={"metadata": {"path": str(Path.cwd())}},
).execute()
nbformat.write(notebook, path)
```

All seven M4 artifact directories must remain available under `reports/ranking/m4/`. Shared profile/input/pool/configuration hashes and recomputed evaluator metrics are validated before joining. A missing or stale MiniLM cache is explicitly reported: saved task/score analyses still run, but live per-sentence winner diagnostics remain unavailable. The notebook never downloads a model or rebuilds task embeddings.

## Outputs

`reports/ranking/diagnostics/` contains the consolidated profile table, all per-profile metrics, subgroup/bootstrap tables, fusion regressions, provisional failure indicators, improvement priorities, sentence/task diagnostics, numerical distributions, figures, computed findings summary, and integrity/provenance manifest. Unretrieved BM25 ranks are missing values; zero BM25 scores remain valid zeros. The value 379 is used only as a plotting sentinel and is never a rank supplied to the evaluator.

Specificity review uses two stages:

1. Give an independent reviewer only `specificity_blind_inputs.csv`. The automated proposals are explicitly provisional and use only text, not targets, vectors, metadata, or predictions.
2. Record text-only human categories/rationales in `specificity_human_reviews.csv` before revealing targets. Only verified entries with a reviewer ID count as human reviews. Then use `specificity_annotations.csv` for separate target compatibility review.

The notebook preserves the human review file across reruns. An empty review file means **zero verified reviews**; no verified-specificity performance chart or human-ground-truth claim is fabricated. Sample selection uses metadata/results for reproducible coverage, but those fields never enter the text-only classification function.

Exploratory thresholds and the priority-weight rubric are declared in the notebook. Their indicators are not causal diagnoses or expected treatment effects. Recommended experiments are proposals only; this deliverable stops before M5 or implementation of any improvement.
