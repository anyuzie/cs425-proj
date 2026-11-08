# Core functionality verification

Checked 21 October 2026. Core prototype paths are implemented and connected;
this is a functional check, not a model-quality acceptance result.

| Function | Implementation and evidence |
| --- | --- |
| Extract preferences | Teammate's real DeBERTa checkpoint produces six RIASEC stance/reason readings; actual-model conversations completed. |
| Keep state and evidence | Cumulative confidence-weighted merging, original utterances, source turns and profile history; contract and dialogue integration tests. |
| Detect contradictions | Opposite-sign candidates plus real NLI confirmation; accepted preference preserved until a targeted answer. Original main rule: both confidences > 0.4, then NLI > 0.6. |
| Choose actions | OPENING, CLARIFY, PROBE, CHALLENGE, ABSTAIN and RECOMMEND are wired. Controlled tests cover neutral retries, persistent/recurrent conflicts and resolutions. |
| Stop conversations | Minimum five turns and stable complete profile; hard cap ten; tests cover both stopping and no-probing control. |
| Generate wording | Local fine-tuned T5 loaded in the HTTP app; invalid/repeated wording falls back to templates. Exact quotations inserted by code. |
| Rank occupations | Original main default: structured weighted-dot-product scoring through configs/ranking.yaml. Existing retrieval/fusion variants are retained, not selected by default. |
| Show results and citations | Live HTTP conversation ended at turn seven with five occupations. All fifteen displayed task citations were checked against their occupation in `data/tasks.csv`. |
| Browser/session support | Page returned HTTP 200; health shows real components and 378 eligible occupations. API tests cover sessions, validation, duplicate messages, debug/study presentation and logs. |
| Evaluate dialogue | New `eval.dialogue` uses real System with paired full/no-probing development simulations; six conversations completed. Labels remain outside System inputs. |

Scope reversion: extraction, contradiction logic, ranking defaults, and their
original tests follow main `7e065d6`. Earlier verification counts below are
superseded by the current scope-reversion test records. Real model loading and
the tracker/action/T5 implementation remain integrated.

## Remaining limits

- No full held-out evaluation, human study, quality tuning or latency sign-off was
  performed. Neither arm placed its target in the top five in the three-profile
  development smoke sample; this is too small for a comparative conclusion.
- The extractor can infer a preference from vague replies such as "hmm". The
  neutral-retry logic works when neutral predictions are supplied, but prediction
  mistakes still affect state. T5 wording can be awkward.
- Raw text retrieval preserves negation in user sentences but BM25 and this dense
  baseline do not reliably interpret negative preferences. Task citations are
  retrieved evidence, not proof of personal suitability.
- State is per-session in memory. Original user sentences are retained for
  retrieval and transcripts; no separate persistent education/background profile
  was added. Model checkpoints and dense assets are local and ignored by Git.
- Mixed/contextual contradiction resolution remains deliberately deferred; the
  current CHALLENGE follows the proposal's choice between statements.

See [RUNNING_APP.md](RUNNING_APP.md) for launch, asset preparation, configuration
alternatives and evaluator commands.
