# Extractor wiring follow-up

The M5 application already selects `deberta` and `configured` ranking in `configs/system.json`. The extractor is constructed once by `pipeline.registry`, then `System.step` routes its `Reading` through the selected tracker before invoking ranking and policy. No ranking algorithms, defaults, metrics, frozen artifacts or historical evaluation rows were changed in this follow-up.

## Adapter corrections

- Map all 45 output logits by semantic names in the checkpoint's `id2label`. The historical teammate config (`15edf27^:model/config.json`) orders direction classes negative/neutral/positive, while `data/labels.json` requires neutral/positive/negative. Positional decoding would reinterpret preference signs. Unknown or incomplete head labels now fail explicitly.
- Use the shared neutral label index when assigning zero strength. The previous positive-index check incorrectly zeroed positive predictions. Active strengths retain the existing 0.2/0.55/1.0 mapping and checkpoint temperatures.
- Load torch/transformers lazily; construct the model once in evaluation mode and run forward passes under inference mode. Local checkpoint loading does not try a remote model identifier when a folder is missing.
- Validate checkpoint head count, labelled output indices, maximum input length and finite positive calibration temperatures. Paired context/user input uses tokenizer truncation to respect the checkpoint's token limit even with long user statements.

## Running the real-checkpoint integration check

The current branch includes the inference adapter, but **no local checkpoint weights**. `models/extractor/` does not exist. The tracked extractor commit `15edf27` removed the earlier `model/` tokenizer/config assets; its current tree contains no model checkpoint. Model folders/weights are also gitignored. Updating this branch alone therefore does not install the trained weights.

Provide the complete exported Hugging Face checkpoint directory: config with `extractor` calibration and all 45 semantic `id2label` entries, tokenizer assets, and trained weights (including shard/index files if applicable). Put files directly under `models/extractor/` or set `paths.extractor_checkpoint` to another local directory.

```powershell
python scripts/check_m5_extractor.py --checkpoint C:/path/to/checkpoint
```

The smoke harness loads the **real extractor** and the configured M5 ranker, runs three chronological turns, checks occupation eligibility/uniqueness, and saves input/profile/ranking traces plus checkpoint/source hashes. It uses explicitly mocked tracker/NLI/policy components because their production implementations remain unfinished. Its condition is `mock_tracker`, even though the extractor is real. No accuracy metrics or evaluation-history entries are produced.

The default ranking selection remains M1-only. The same smoke harness can exercise the existing optional three-way pipeline:

```powershell
python scripts/check_m5_extractor.py --checkpoint C:/path/to/checkpoint --ranking-config configs/ranking/m4_three_way.json --output reports/ranking/m5/extractor_three_way_smoke.json
```

Each output must be fresh. Preparing the existing pinned M4 cache is required for three-way execution. The production application still requires the real tracker, NLI and policy; the smoke harness does not install test stand-ins into application defaults.

## Validation and remaining dependency

Adapter tests cover semantic head reordering, positive/negative strength propagation into M5's current tracker vector and top-100 result, contextual tokenization, inference mode, lazy imports, unlabelled-head rejection and a clear missing-checkpoint error. The original M5 integration and ranking regression tests remain in the full suite.

Full discovery ran **159 tests: 158 passed, one error** from the existing production-startup test requiring the missing extractor checkpoint. All four new checkpoint-adapter tests pass. See [test output](extractor-test-results.txt). The original M5 report retains its historical test count.

Actual trained-model inference has **not** been validated: the real smoke command exits clearly at the missing checkpoint before creating a report. No output or model quality result is fabricated. The original frozen `baseline_v1` remains unchanged; this follow-up is not a replacement freeze.
