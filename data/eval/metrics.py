"""Reference metrics. `ranking` is the system's ordered list of SOC codes (Job Zone 4-5 pool)."""
import math

def grades(profile):
    g = {}
    for c in profile["related"]["supplemental"]: g[c] = 1
    for c in profile["related"]["primary_long"]: g[c] = 2
    for c in profile["related"]["primary_short"]: g[c] = 3
    g[profile["target"]] = 4
    return g

def hit_at(ranking, profile, k=5):
    return int(profile["target"] in ranking[:k])

def related_hit_at(ranking, profile, k=5):
    ok = {profile["target"], *profile["related"]["primary_short"]}
    return int(any(c in ok for c in ranking[:k]))

def twin_hit_at(ranking, profile, k=5):
    ok = {profile["target"], *profile["riasec_twins"]}       # jobs with a near-identical RIASEC profile
    return int(any(c in ok for c in ranking[:k]))

def mrr(ranking, profile):
    return 1 / (ranking.index(profile["target"]) + 1) if profile["target"] in ranking else 0.0

def ndcg_at(ranking, profile, k=10):
    g = grades(profile)
    reachable = {profile["target"], *[c for v in profile["related_in_pool"].values() for c in v]}
    dcg = sum((g.get(c, 0) if c in reachable else 0) / math.log2(i + 2) for i, c in enumerate(ranking[:k]))   # out-of-pool codes earn nothing, so NDCG <= 1
    ideal = sorted((g[c] for c in reachable if c in g), reverse=True)[:k]   # IDCG over codes the system can return
    idcg = sum(v / math.log2(i + 2) for i, v in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def evaluate(rankings, profiles):
    """rankings: {profile_id: [codes]}; returns the mean of every metric."""
    out = {"hit@5": 0, "hit@3": 0, "related_hit@5": 0, "twin_hit@5": 0, "mrr": 0, "ndcg@10": 0}
    for p in profiles:
        r = rankings[p["id"]]
        out["hit@5"] += hit_at(r, p, 5); out["hit@3"] += hit_at(r, p, 3)
        out["related_hit@5"] += related_hit_at(r, p); out["twin_hit@5"] += twin_hit_at(r, p)
        out["mrr"] += mrr(r, p); out["ndcg@10"] += ndcg_at(r, p)
    return {k: v / len(profiles) for k, v in out.items()}
