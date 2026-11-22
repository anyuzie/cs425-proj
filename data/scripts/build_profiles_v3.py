"""Assemble the v3 test/dev profiles from the blind drafts."""
import json, csv, os, numpy as np
W, OUT = os.environ.get("V3DIR", "/home/claude/data/work/v3"), "/home/claude/data/out"
T = ["R", "I", "A", "S", "E", "C"]
NM = dict(zip(["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"], T)); LN = {v: k for k, v in NM.items()}
m1 = json.load(open(f"{W}/tier1_meta.json")); t1 = json.load(open(f"{W}/tier1_texts.json"))
m2 = {x["id"]: x for x in json.load(open(f"{W}/tier2_meta.json"))}; d2 = {x["id"]: x for x in json.load(open(f"{W}/tier2_drafts.json"))}
res = json.load(open(f"{W}/resolutions.json"))
rk = json.load(open(f"{W}/rate_key.json")); ro = json.load(open(f"{W}/rate_out.json")); rating = {rk[q]: v for q, v in ro.items()}
bank = json.load(open(f"{OUT}/question_bank.json"))
occ = {o["code"]: o for o in csv.DictReader(open(f"{OUT}/occupations.csv"))}
pool = [c for c, o in occ.items() if o["z_R"] and o["job_zone"] in ("4.0", "5.0")]
Z = np.array([[float(occ[c][f"z_{t}"]) for t in T] for c in pool])
rel = {}
for r in csv.DictReader(open(f"{OUT}/related_occupations.csv")): rel.setdefault(r["code"], []).append(r)
level = lambda z: "pos" if z >= 1 else "pos_mild" if z >= 0.4 else "dontmind" if z > -0.4 else "neg_mild" if z > -0.8 else "neg"
prof = []
for pid in list(m1) + list(m2):
    if pid in m1:
        m = m1[pid]; p = {"id": pid, "tier": 1, "style": m["style"], "background": m["background"], "text": t1[pid],
                          "planted": [], "contradiction": None, "latent_reveal": ""}
    else:
        m = m2[pid]; d = d2[pid]; p = {"id": pid, "tier": 2, "style": "full", "background": False, "plan": m["plan"], "text": d["text"],
                                       "planted": d["planted"], "contradiction": d["contradiction"], "latent_reveal": d["latent_reveal"]}
    p.update({"hidden_slots": m["hidden_slots"], "target": m["target"], "target_profile": m["target_profile"]})
    planted = {NM[x.split(":")[1]] for x in p["planted"]}
    rs = {NM[k]: v for k, v in res[pid].items()}
    # a strong type the text never mentions has nothing to restate: answer like the CLARIFY simulator (by band)
    for t in T:
        if t not in planted and abs(p["hidden_slots"][t]) >= 0.8 and rating[pid][t] == 0:
            rs[t] = bank["ANSWERS"][LN[t]][0][level(p["hidden_slots"][t])]
    p["resolutions"] = rs
    if p.get("contradiction"): p["contradiction"]["slot"] = LN.get(p["contradiction"]["slot"], p["contradiction"]["slot"])  # full type name
    L2N = True
    R_ = rel.get(p["target"], [])
    p["related"] = {k: [x["related_code"] for x in R_ if x["tier"] == tier] for k, tier in
                    (("primary_short", "Primary-Short"), ("primary_long", "Primary-Long"), ("supplemental", "Supplemental"))}
    p["related_in_pool"] = {k: [c for c in v if c in pool] for k, v in p["related"].items()}
    v = np.array([p["target_profile"][t] for t in T]); c = Z @ v / (np.linalg.norm(Z, axis=1) * np.linalg.norm(v) + 1e-9)
    p["riasec_twins"] = [pool[i] for i in np.argsort(-c) if c[i] > 0.98 and pool[i] != p["target"]]
    prof.append(p)
if os.environ.get("V3DIR"):          # extra dev profiles: written separately, merged into dev_profiles.jsonl below
    with open(f"{W}/dev_extra.jsonl", "w") as f:
        for p in sorted(prof, key=lambda p: p["id"]): f.write(json.dumps(p) + "\n")
    print("dev_extra", len(prof))
else:
  for name, pre in (("profiles.jsonl", "P"), ("dev_profiles.jsonl", "D")):
    sel = sorted([p for p in prof if p["id"].startswith(pre)], key=lambda p: p["id"])
    extra = f"/home/claude/data/work/v3extra/dev_extra.jsonl"
    if pre == "D" and os.path.exists(extra): sel += [json.loads(l) for l in open(extra)]
    with open(f"{OUT}/{name}", "w") as f:
        for p in sel: f.write(json.dumps(p) + "\n")
    print(name, len(sel))
json.dump(rating, open(f"{W}/ratings.json", "w"))
