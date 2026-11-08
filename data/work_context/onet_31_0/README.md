# O*NET 31.0 Work Context sidecar

Retrieved from the official O*NET 31.0 Database. Existing occupations.csv was
built from that release (data/scripts/build_onet.py). This sidecar does not change
the existing CSV, annotation labels, classifier checkpoint, or ranking behaviour.

## Coverage

- All 1,016 local occupation codes and titles match the official 31.0 reference.
- 911 occupations have Work Context data; 105 have no data in this source.
- Of 378 currently ranking-eligible occupations, 372 have context data; six do not.
- 305,389 original rating rows describe 57 distinct context elements on four scales.
- 11,491 ratings have suppression flags; they remain preserved, not discarded.

Missing data means unknown, not zero or absence of a working condition. No crosswalk,
fuzzy occupation join, imputation, dominant-category selection or ranking score is added.

## Files

- raw/work_context.csv: exact downloaded ratings, statistics and reliability flags.
- raw/work_context_categories.csv: response-category meanings.
- raw/scales_reference.csv: scale names and numeric ranges.
- raw/content_model_reference.csv: context names and definitions/questions.
- raw/occupation_data.csv: reference taxonomy and titles used to verify the join.
- prepared/occupation_work_context.jsonl: one record per local occupation, grouped
  context elements with every rating and source row number. Unmatched context
  availability is explicit via has_context_data=false and an empty context array.
- prepared/context_units.csv: 57 short context names and source definitions.
- prepared/occupation_coverage.csv: per-occupation match and coverage audit.
- prepared/coverage_report.json: counts, mismatches and missing occupation list.
- source_manifest.json: pinned release, URLs, download sizes and SHA-256 fingerprints.

The rating records retain source numeric strings, sample sizes, standard errors,
confidence intervals, dates, domain sources, suppression flags and scale IDs.
Category descriptions and scale bounds are added from the official references.
The JSONL is a structural transformation, not new model predictions or human labels.

## How to interpret context

An attribute's presence does not mean a job has a high level of that condition.
All rated variables must be interpreted with their values and categories. For
example, a public-speaking item asks how frequently public speaking is required;
it is not simply evidence that public speaking is a frequent activity. Preserve
those frequencies before deriving matching text. Never compare CT (1–3), CX
(1–5), and category percentages (0–100) as interchangeable numeric scores.

Context units are already short enough for the next preparation stage. For mixed
task prose, extract activity-plus-object and context spans separately while keeping
their parent source and relationship, e.g. “repair machinery outdoors” → activity
“repair machinery” and context “outdoors”. Do not split down to verbs alone or turn
all conditions into universal properties of an occupation. This retrieval does
not automatically resegment or relabel existing activity examples.

Reproduce with retrieve.py followed by prepare.py in the same folder, using the
project's Python environment. verify.py checks full source-field preservation,
coverage and unchanged source occupations.csv.

## Attribution

Includes information from the O*NET 31.0 Database by the U.S. Department of Labor,
Employment and Training Administration (USDOL/ETA), used under CC BY 4.0:
https://www.onetcenter.org/database.html
https://creativecommons.org/licenses/by/4.0/
https://www.onetcenter.org/license_db.html

Raw downloads are unchanged. Derived files group the source data by occupation,
add reference descriptions and coverage metadata, and link to the local dataset.
These transformations are project changes, not endorsed or tested by USDOL/ETA.
