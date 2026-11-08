# Fine-tuned NLI deployment

The real app presets select **NLI v2**, the `1.75e-5` fine-tune with a `0.90`
contradiction cutoff. Copy the team's shared `models/nli-probing-v2/checkpoint/`
folder to the same path in your checkout. Git tracks the manifest, and the
team shares these checkpoint files separately:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

From the repository root, verify the files against the committed sizes and
SHA-256 hashes before starting the application:

```powershell
python scripts/check_nli_checkpoint.py --smoke
python run.py
```

Use the project's Python environment. The smoke check verifies the file hashes,
then loads the checkpoint and prints probabilities and the confirmation decision
for one sentence pair. See [the app test conversations](RUNNING_APP.md#try-these-conversations)
to try the new configuration end to end.

`configs/nli/probing_v2.json` records hashes and training provenance. The deployed
checkpoint is epoch 1 from the original train/dev split at `1.75e-5`. The grouped
five-fold experiment used separately trained checkpoints to select the settings.
At `0.90` it produced 96.70% pooled accuracy, 96.60% recall, 9 misses and 9 false
alarms after correcting one food-stall label during scoring. The development
pool includes generated examples, and the team selected the threshold using
these validation results.

In the app, only opposite-sign readings of the same RIASEC type with both
confidences above 0.4 reach NLI. A contradiction probability strictly above
`settings.confirm_threshold` confirms a candidate.

To compare with v1, copy a system preset and set `paths.nli_model` to
`models/nli-probing-v1/checkpoint` and `settings.confirm_threshold` to `0.60`.
Select that preset using `CM_CONFIG`. Verify its checkpoint with
`python scripts/check_nli_checkpoint.py --manifest configs/nli/probing_v1.json`.
