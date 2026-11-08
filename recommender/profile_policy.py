"""Ranking-owned Stage A: frozen transforms of the tracker vector before scoring.

Mirrors ``evidence_filter``: a policy changes only the structured input that
components receive; components (dot product, cosine, ...) are unchanged. Text is
never altered. Policies read the tracker's per-slot reason; they never read
hidden slots, targets or relevance labels.

Hypothesis for the familiarity policies (fixed before running): an unconfirmed
familiarity-based slot ("I did a lot of X at school") is weaker evidence of
interest than an enjoyment-based slot. At a single turn the policy cannot tell
misleading from genuine familiarity, so its net effect is expected to be mixed;
subgroup results show how much is left for probing to resolve.
"""
from copy import deepcopy

POLICIES = {
    "raw": None,
    # Unconfirmed familiarity treated as no evidence: the type becomes unobserved.
    "familiarity_excluded_v1": {"reason": "familiarity", "action": "exclude"},
    # Unconfirmed familiarity kept at half weight; the type stays observed.
    "familiarity_half_v1": {"reason": "familiarity", "action": "scale", "factor": 0.5},
}


def validate_policy(name):
    if name not in POLICIES:
        raise ValueError(f"profile_policy must be one of {sorted(POLICIES)}")
    return name


def apply_policy(name, interests, observed, slot_reasons):
    """Return (interests, observed, diagnostics); inputs are not mutated."""
    rule = POLICIES[validate_policy(name)]
    interests, observed = dict(interests), set(observed)
    if rule is None:
        return interests, observed, {"profile_policy": name, "changed_types": []}
    if slot_reasons is None:
        raise ValueError(f"profile_policy {name} requires tracker slot_reasons")
    changed = sorted(t for t in observed if slot_reasons.get(t) == rule["reason"])
    for t in changed:
        if rule["action"] == "exclude":
            observed.discard(t)
            interests[t] = 0.0
        else:
            interests[t] *= rule["factor"]
    return interests, observed, {"profile_policy": name, "rule": deepcopy(rule), "changed_types": changed}
