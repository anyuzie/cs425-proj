"""Assemble v4 profiles: new dev D061-D120 (appended to dev_profiles.jsonl) and test v4 V001-V120 (profiles_v4.jsonl).
Also adds `pattern` and `nearest_by_profile` to every profile in all three files."""
import json, csv, os, glob, numpy as np
D = "/home/claude/data"; W = f"{D}/work/v4"; OUT = f"{D}/out"
T = ["R", "I", "A", "S", "E", "C"]
NM = dict(zip(["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"], T)); LN = {v: k for k, v in NM.items()}
L = lambda p: [json.loads(l) for l in open(p)]
occ = {o["code"]: o for o in csv.DictReader(open(f"{OUT}/occupations.csv"))}
pool = [c for c, o in occ.items() if o["z_R"] and o["job_zone"] in ("4.0", "5.0")]
Z = np.array([[float(occ[c][f"z_{t}"]) for t in T] for c in pool])
rel = {}
for r in csv.DictReader(open(f"{OUT}/related_occupations.csv")): rel.setdefault(r["code"], []).append(r)
bank = json.load(open(f"{OUT}/question_bank.json"))
level = lambda z: "pos" if z >= 1 else "pos_mild" if z >= 0.4 else "dontmind" if z > -0.4 else "neg_mild" if z > -0.8 else "neg"

def pattern_of(h):
    v = sorted(h.values(), reverse=True)
    if v[0] < 0.75: return "weak_uncertain"
    if sum(x >= 0.8 for x in v) >= 4 and v[-1] > -0.3: return "generalist"
    if v[0] >= 1.0 and sum(x <= -1.0 for x in v) >= 2 and v[1] >= 1.0: return "conflicting"
    if v[2] >= 1.0 and v[0] - v[2] <= 0.35: return "three_balanced"
    if v[1] >= 1.0 and v[0] - v[1] <= 0.6: return "two_strong"
    return "single_dominant"

def cos_top(vec, k=5):
    v = np.array([vec[t] for t in T]); c = Z @ v / (np.linalg.norm(Z, axis=1) * np.linalg.norm(v) + 1e-9)
    return c, [pool[i] for i in np.argsort(-c)[:k]]

def enrich(p):
    R_ = rel.get(p["target"], [])
    p["related"] = {k: [x["related_code"] for x in R_ if x["tier"] == tier] for k, tier in
                    (("primary_short", "Primary-Short"), ("primary_long", "Primary-Long"), ("supplemental", "Supplemental"))}
    p["related_in_pool"] = {k: [c for c in v if c in pool] for k, v in p["related"].items()}
    c, _ = cos_top(p["target_profile"])
    p["riasec_twins"] = [pool[i] for i in np.argsort(-c) if c[i] > 0.98 and pool[i] != p["target"]]
    p["nearest_by_profile"] = cos_top(p["hidden_slots"])[1]          # the 5 pool jobs closest to this person
    p.setdefault("pattern", pattern_of(p["hidden_slots"]))
    p["top_type"] = max(T, key=lambda t: p["hidden_slots"][t])
    return p

def build_new():
    m1 = json.load(open(f"{W}/tier1_meta.json")); m2 = {x["id"]: x for x in json.load(open(f"{W}/tier2_meta.json"))}
    t1 = {}; [t1.update(json.load(open(f))) for f in sorted(glob.glob(f"{W}/t1_out_*.json"))]
    d2 = {}; [d2.update({x["id"]: x for x in json.load(open(f))}) for f in sorted(glob.glob(f"{W}/t2_out_*.json"))]
    res = json.load(open(f"{W}/resolutions.json")); rating = json.load(open(f"{W}/ratings.json"))
    prof = []
    for pid in list(m1) + list(m2):
        if pid in m1:
            m = m1[pid]; p = {"id": pid, "tier": 1, "style": m["style"], "background": m["background"], "text": t1[pid],
                              "planted": [], "contradiction": None, "latent_reveal": ""}
        else:
            m = m2[pid]; d = d2[pid]; p = {"id": pid, "tier": 2, "style": "full", "background": False, "plan": m["plan"], "text": d["text"],
                                           "planted": d["planted"], "contradiction": d["contradiction"], "latent_reveal": d.get("latent_reveal", "")}
        p.update({"hidden_slots": m["hidden_slots"], "target": m["target"], "target_profile": m["target_profile"], "pattern": m["pattern"]})
        planted = {NM[x.split(":")[1]] for x in p["planted"]}
        rs = {NM.get(k, k): v for k, v in res[pid].items()}
        for t in T:     # a strong type the text never mentions: answer like the CLARIFY simulator (by band)
            if t not in planted and abs(p["hidden_slots"][t]) >= 0.8 and rating[pid][t] == 0:
                rs[t] = bank["ANSWERS"][LN[t]][0][level(p["hidden_slots"][t])]
        p["resolutions"] = rs
        prof.append(enrich(p))
    return prof

if __name__ == "__main__":
    new = build_new()
    dev_old = [enrich(p) for p in L(f"{OUT}/dev_profiles.jsonl") if int(p["id"][1:]) <= 60]
    test_v3 = [enrich(p) for p in L(f"{OUT}/profiles.jsonl")]
    dev = dev_old + sorted([p for p in new if p["id"][0] == "D"], key=lambda p: p["id"])
    v4 = sorted([p for p in new if p["id"][0] == "V"], key=lambda p: p["id"])
    for name, rows in (("profiles.jsonl", test_v3), ("dev_profiles.jsonl", dev), ("profiles_v4.jsonl", v4)):
        with open(f"{OUT}/{name}", "w") as f:
            for p in rows: f.write(json.dumps(p) + "\n")
        print(name, len(rows))
    agree = sum(pattern_of(p["hidden_slots"]) == p["pattern"] for p in new); print("designed pattern = rule pattern:", agree, "/", len(new))
