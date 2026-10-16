"""M2 exact-target and graded metrics. Labels stay on the evaluator side."""

import math

METRICS = ("hit_at_5", "mrr", "ndcg_at_10")
GRADES = {"primary_short": 3, "primary_long": 2, "supplemental": 1}


def relevance_labels(target: str | None, related: dict, pool: set[str]) -> dict[str, int]:
    labels = {}
    for tier, grade in GRADES.items():
        codes = related.get(tier, [])
        if not isinstance(codes, list) or any(not isinstance(c, str) for c in codes):
            raise ValueError(f"invalid relevance list: {tier}")
        for code in codes:
            if code in pool:
                labels[code] = max(labels.get(code, 0), grade)
    if target in pool:
        labels[target] = 4
    return labels


def evaluate_ranking(codes: list[str], target: str | None, related: dict,
                     pool: set[str]) -> dict:
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate recommendations")
    if not set(codes) <= pool:
        raise ValueError("recommendations outside eligible pool")
    labels = relevance_labels(target, related, pool)
    target_rank = codes.index(target) + 1 if target in codes else None
    def dcg(grades):
        return sum((2 ** grade - 1) / math.log2(rank + 1)
                   for rank, grade in enumerate(grades, 1))
    ideal = dcg(sorted(labels.values(), reverse=True)[:10])
    actual = dcg([labels.get(code, 0) for code in codes[:10]])
    return {
        "hit_at_5": int(target_rank is not None and target_rank <= 5),
        "mrr": 1 / target_rank if target_rank else 0.0,
        "ndcg_at_10": actual / ideal if ideal else 0.0,
        "target_rank": target_rank,
        "target_status": "missing" if not target else "eligible" if target in pool else "ineligible",
        "zero_idcg": ideal == 0,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot evaluate an empty profile set")
    return {
        "profile_count": len(rows),
        **{key: sum(row[key] for row in rows) / len(rows) for key in METRICS},
        "missing_targets": sum(row["target_status"] == "missing" for row in rows),
        "ineligible_targets": sum(row["target_status"] == "ineligible" for row in rows),
        "zero_idcg_cases": sum(row["zero_idcg"] for row in rows),
        "unranked_targets": sum(row["target_rank"] is None for row in rows),
    }
