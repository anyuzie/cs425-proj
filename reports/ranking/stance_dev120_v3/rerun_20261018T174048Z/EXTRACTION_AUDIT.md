# Text-only extraction audit

{
  "profiles": 120,
  "clauses": 699,
  "usable_profiles": 97,
  "polarity_quality": "UNVERIFIED: no independent labels supplied",
  "profile_counts": {
    "positive": 90,
    "negative": 38,
    "unknown": 118,
    "usable": 97,
    "mixed_polarity": 31,
    "explicit_negation": 81,
    "uncertain": 16,
    "background_or_third_person": 43,
    "contradicted": 0,
    "familiarity_without_preference": 4
  },
  "clause_polarities": {
    "unknown": 517,
    "positive": 135,
    "negative": 47
  },
  "rules": {
    "no_explicit_preference": 438,
    "activity_preference_predicate": 27,
    "third_party_or_background_scope": 49,
    "explicit_first_person": 139,
    "explicit_negative_predicate": 16,
    "uncertain_or_negated_dislike": 24,
    "unclear_activity_scope": 6
  }
}

Counts are automatic rule outputs, not independent ground truth. Familiarity and third-person counts are conservative lexical proxies. Exact normalized activity conflicts are withheld; paraphrase conflicts may be missed. Unknown includes bare mentions, unclear scope and uncertainty. Review spans and their full source context before interpreting labels. The review template has intentionally blank human judgments; precision/recall cannot be reported. Review files are never read by inference.
