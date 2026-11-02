"""M2 exact-target and graded metrics. Labels stay on the evaluator side."""

import math

LEGACY_METRICS = ("hit_at_5", "mrr", "ndcg_at_10")
PRE_RECALL_METRICS = ("hit_at_1", "hit_at_5", "hit_at_10", "mrr", "ndcg_at_5", "ndcg_at_10")
RECALL_METRICS = ("recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50")
METRICS = PRE_RECALL_METRICS + RECALL_METRICS
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


def hit_at(target_rank: int | None, k: int) -> int:
    return int(target_rank is not None and target_rank <= k)


def dcg(grades, k: int) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 1)
               for rank, grade in enumerate(list(grades)[:k], 1))


def ndcg_at(codes: list[str], labels: dict[str, int], k: int) -> float:
    ideal = dcg(sorted(labels.values(), reverse=True), k)
    actual = dcg([labels.get(code, 0) for code in codes[:k]], k)
    return actual / ideal if ideal else 0.0


def evaluate_ranking(codes: list[str], target: str | None, related: dict,
                     pool: set[str]) -> dict:
    if len(codes) != len(set(codes)):
        raise ValueError("duplicate recommendations")
    if not set(codes) <= pool:
        raise ValueError("recommendations outside eligible pool")
    labels = relevance_labels(target, related, pool)
    target_rank = codes.index(target) + 1 if target in codes else None
    ideal = dcg(sorted(labels.values(), reverse=True), 10)
    return {
        **{f"hit_at_{k}": hit_at(target_rank, k) for k in (1, 5, 10)},
        "mrr": 1 / target_rank if target_rank else 0.0,
        **{f"ndcg_at_{k}": ndcg_at(codes, labels, k) for k in (5, 10)},
        **{f"recall_at_{k}": len(set(codes[:k]) & labels.keys()) / len(labels) if labels else 0.
           for k in (5, 10, 20, 50)},
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
