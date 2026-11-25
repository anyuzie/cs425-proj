# Synthetic continuation curriculum v1

Built by `tools/build_continuation_data.py`; the untested lower-replay subset was corrected by `research_20261109/stratify_replay.py`. Both versions are preserved. No reserved final content was read to author this curriculum.

The 36 authored activities are allocated before expansion: 24 training, six validation, six auxiliary audit, balanced across RIASEC slots. Each group expands to three actions with three initial questions and six retries (27 rows). CHALLENGE requests a choice between stored statements; PROBE separates enjoyment from familiarity. Long task phrases, curly quotations, an em dash and colloquial stored context are included. `last_user` and `known_interest` stay empty. No reply interpretation is added.

New training targets use wording families 0–3; auxiliary validation/audit targets use families 4–5. Previous-question families and the structured schema are shared. This is a limited family split, not proof of semantic independence. Global RIASEC labels and some legacy question families necessarily overlap the incumbent's training. All data are synthetic and AI-reviewed; no independent human review has occurred.

`full_replay` has 648 new + 864 original training rows; `lower_replay_stratified` has the same 648 new + 216 replay rows (six per slot/action/retry stratum). Both use the same 162 new + 216 original validation rows. The original replay's repeated inputs with different targets are retained. New-only audit contains 162 rows, separate from the reserved final set. Counts refer to template-expanded rows, not independent participants.

Construction and token audits record hashes, row/unique-input counts and action/retry distributions. Exact and canonicalized overlaps are zero between training, validation, auxiliary audit and known development inputs. All new activity strings are disjoint from original training/validation activity strings. Surface and template similarity remains; no embedding-based semantic leakage detector was run. All generated targets pass the existing unchanged format/focus/action/novelty checks, and all rows fit 256 input/64 target tokens. This automated check is not a human quality score.

Replay ratios change row/token/step counts: 189 steps per epoch for full replay versus 108 for lower replay and original-data control, at batch eight. Full replay versus original control also changes the validation set, so validation losses across those conditions cannot select a winner or establish a causal data effect. Full-replay LR variants share data, order seed, batch, epoch and method. Lower replay versus full replay changes both data exposure and step budget. LoRA versus full also changes LR and parameterization.

`lower_replay` (without the `_stratified` suffix) is an untested construction retained for provenance. Its first-rows replay selection overrepresented earlier slots; it is not used in model experiments.
