# Fine-tuned NLI deployment

The production presets select `models/nli-probing-v1/checkpoint`. Model weights
are shared separately and are not included in Git. Obtain the team's original
probing-v1 checkpoint and copy these files into that directory:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

From the repository root, verify the files against the committed sizes and
SHA-256 hashes before starting the application:

```powershell
python scripts/check_nli_checkpoint.py
python scripts/check_nli_checkpoint.py --smoke
python run.py
```

Use the project's Python environment for these commands. The smoke check loads
the real checkpoint and prints probabilities for one sentence pair; it is not
an accuracy evaluation. Do not substitute stock weights under this folder name.
Missing local checkpoints deliberately fail startup rather than silently loading
another model.

`configs/nli/probing_v1.json` records checkpoint hashes, training provenance,
selection and limitations. The selected first epoch was trained from
`cross-encoder/nli-deberta-v3-base`. Its training policy includes tensions between
interests and career preferences worth asking about. However, the live candidate
rule remains unchanged: only opposite-sign readings of the same RIASEC type
with both confidences above 0.4 reach NLI. A contradiction probability strictly
above 0.6 confirms a candidate under the default configuration. Cross-domain
statements alone therefore do not trigger a challenge.

The upstream model uses synthetic additions and previously inspected legacy
test predictions. Its manifest does not establish held-out conversational
accuracy. The extractor, candidate rule, tracker/action priorities and ranking
algorithms are separate from this checkpoint selection.
