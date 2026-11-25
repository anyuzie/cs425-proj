"""v4 profiles: balanced by the STUDENT's top z-type (hidden_slots), built around six patterns.
- dev: +60 students (D061-D120) so dev reaches 20 per top type; test v4: 120 new students (V001-V120), 20 per top type.
- Target = a degree-level job (Job Zone 4-5) where the student's top type is in the job's top 3 z-scores and z >= 0.3.
  No degree-level job has Realistic as its top z-score, so R-led students get the most Realistic degree jobs (engineers etc.).
- hidden_slots = the job's z-row reshaped by the pattern (+ small noise), with the chosen type always on top.
- Jobs: each used once across v4; dev and test never share a job; test v4 avoids every earlier dev job; new dev avoids every test job.
Writers never see the job title (same rule as v3)."""
import pandas as pd, numpy as np, json, os
rng = np.random.default_rng(2028)
T = ["R", "I", "A", "S", "E", "C"]
OUT = "/home/claude/data/work/v4"; os.makedirs(OUT, exist_ok=True)
D = "/home/claude/data"
L = lambda p: [json.loads(l) for l in open(p)]
df = pd.read_csv(f"{D}/out/occupations.csv")
df = df[df.has_riasec & (df.job_zone >= 4) & (df.n_tasks >= 8) & ~df.Title.str.contains("All Other")].reset_index(drop=True)
Z = df[[f"z_{t}" for t in T]].to_numpy()
rank = np.argsort(np.argsort(-Z, axis=1), axis=1)            # rank[i, t] = position of type t in job i (0 = top)

test_v3 = {p["target"] for p in L(f"{D}/out/profiles.jsonl")}
dev_old = {p["target"] for p in L(f"{D}/out/dev_profiles.jsonl")}
early = set(json.load(open(f"{D}/work/tier1_targets.json")).values()) | {x["target"] for x in json.load(open(f"{D}/work/tier2_inputs.json"))}
for f in ["work/old_profiles/profiles.jsonl", "work/old_profiles/dev_profiles.jsonl"]:
    if os.path.exists(f"{D}/{f}"): early |= {p["target"] for p in L(f"{D}/{f}")}

cur = pd.Series([max(T, key=lambda t: p["hidden_slots"][t]) for p in L(f"{D}/out/dev_profiles.jsonl")]).value_counts().to_dict()
DEV_NEED = {t: max(0, 20 - cur.get(t, 0)) for t in T}
extra = 60 - sum(DEV_NEED.values())                           # if a type is already over 20, spread the rest evenly
for t in sorted(T, key=lambda t: cur.get(t, 0)):
    if extra <= 0: break
    DEV_NEED[t] += 1; extra -= 1
TEST_NEED = {t: 20 for t in T}
PATTERNS = ["single_dominant", "two_strong", "three_balanced", "weak_uncertain", "conflicting", "generalist"]
PLANS = ["contradiction_like", "contradiction_dislike", "familiarity_misleading", "familiarity_genuine", "both"]

used = set()
def pick(t, banned, n):
    i = T.index(t); out = []
    for tier in (0, 1, 2):                                   # prefer jobs where t is the job's own top type
        for fresh in (True, False):
            c = [k for k in rng.permutation(len(df)) if rank[k, i] == tier and Z[k, i] >= 0.3 and df.code[k] not in used
                 and df.code[k] not in banned and ((df.code[k] not in early | test_v3 | dev_old) or not fresh)]
            for k in c:
                if len(out) == n: return out
                out.append(k); used.add(df.code[k])
    assert len(out) == n, (t, len(out), n)
    return out

def shape(z, t, pattern):
    i = T.index(t); h = z.copy(); others = [j for j in np.argsort(-z) if j != i]; low = list(np.argsort(z))
    U = lambda a, b: float(rng.uniform(a, b))
    if pattern == "single_dominant":
        h = 0.5 * z; h[i] = max(1.8, z[i] + 0.6); h[others] = np.minimum(h[others], 0.6)
    elif pattern == "two_strong":
        s2 = others[0]; h = 0.6 * z; h[i] = max(1.6, z[i] + 0.4); h[s2] = max(1.3, h[i] - U(0.1, 0.3))
        rest = others[1:]; h[rest] = np.minimum(h[rest], 0.6)
    elif pattern == "three_balanced":
        s2, s3 = others[:2]; h = 0.6 * z; h[i] = U(1.3, 1.6); h[s2] = h[i] - U(0.05, 0.15); h[s3] = h[i] - U(0.12, 0.25)
        rest = others[2:]; h[rest] = np.minimum(h[rest], 0.4)
    elif pattern == "weak_uncertain":
        h = np.clip(0.25 * z, -0.45, 0.35); h[i] = min(0.7, h.max() + U(0.2, 0.35))
    elif pattern == "conflicting":
        s2 = others[0]; l1, l2 = [j for j in low if j not in (i, s2)][:2]
        h = 0.6 * z; h[i] = max(1.7, z[i] + 0.5); h[s2] = U(1.3, 1.55); h[l1] = min(z[l1], -1.3); h[l2] = min(z[l2], -1.2)
    elif pattern == "generalist":
        h = np.clip(1.0 + 0.25 * (z - z.mean()), 0.6, 1.25); h[i] = h.max() + 0.15
    h = h + rng.normal(0, 0.1 if pattern != "weak_uncertain" else 0.05, 6)
    if pattern == "weak_uncertain": h = np.clip(h, -0.5, 0.7)
    if np.argmax(h) != i: h[i] = h.max() + 0.12
    h = np.clip(h, -2.5, 3.0)
    return {T[j]: round(float(h[j]), 2) for j in range(6)}

def build(need, prefix, start, banned):
    rows = []
    for t in T:
        ks = pick(t, banned, need[t])
        for n, k in enumerate(ks):
            pat = PATTERNS[n % 6]
            rows.append({"t": t, "k": int(k), "pattern": pat})
    order = list(rng.permutation(len(rows))); rows = [rows[j] for j in order]
    # 40% Tier 2, never on weak_uncertain (nothing strong enough to plant on)
    n2 = round(0.4 * len(rows)); t2 = [r for r in rows if r["pattern"] != "weak_uncertain"][:n2]
    t1n = t2n = 0; out1, out2 = {}, []
    for idx, r in enumerate(rows):
        k = r["k"]; z = Z[k]; h = shape(z, r["t"], r["pattern"]); pid = f"{prefix}{start + idx:03d}"
        tp = {t: round(float(z[j]), 2) for j, t in enumerate(T)}
        base = {"id": pid, "target": df.code[k], "target_profile": tp, "hidden_slots": h, "pattern": r["pattern"], "top_type": r["t"]}
        if r in t2:
            srt = sorted(T, key=lambda t: h[t])
            ok = {"contradiction_like": any(h[t] >= 0.6 for t in T), "familiarity_genuine": any(h[t] >= 0.6 for t in T),
                  "contradiction_dislike": any(h[t] <= -0.6 for t in T), "familiarity_misleading": any(h[t] <= -0.3 for t in T),
                  "both": any(h[t] >= 0.6 for t in T) and any(h[t] <= -0.3 for t in T)}
            plan = next(PLANS[(t2n + j) % 5] for j in range(5) if ok[PLANS[(t2n + j) % 5]])
            out2.append({**base, "plan": plan, "lowest": srt[:2], "highest": srt[::-1][:2],
                         "neg_strong": [t for t in T if h[t] <= -0.6], "pos_strong": [t for t in T if h[t] >= 0.6]}); t2n += 1
        else:
            messy = r["pattern"] in ("single_dominant", "two_strong", "weak_uncertain") and t1n % 3 == 0
            out1[pid] = {**base, "style": "messy" if messy else "full", "background": t1n % 4 == 1}; t1n += 1
    return out1, out2

dev1, dev2 = build(DEV_NEED, "D", 61, banned=test_v3)
tst1, tst2 = build(TEST_NEED, "V", 1, banned=dev_old | {x["target"] for x in list(dev1.values()) + dev2})
tier1 = {**dev1, **tst1}; t2 = dev2 + tst2
json.dump(tier1, open(f"{OUT}/tier1_meta.json", "w"), indent=1)
json.dump(t2, open(f"{OUT}/tier2_meta.json", "w"), indent=1)
json.dump({k: {**v["hidden_slots"], "pattern": v["pattern"], "style": v["style"], "background": v["background"]} for k, v in tier1.items()},
          open(f"{OUT}/tier1_numbers.json", "w"), indent=1)
json.dump([{k: v for k, v in x.items() if k not in ("target", "target_profile", "top_type")} for x in t2],
          open(f"{OUT}/tier2_inputs_blind.json", "w"), indent=1)
allp = list(tier1.values()) + t2
for pre in ("D", "V"):
    s = [p for p in allp if p["id"][0] == pre]
    print(pre, len(s), "tier1", sum(1 for p in s if "style" in p), "| top:", pd.Series([max(T, key=lambda t: p["hidden_slots"][t]) for p in s]).value_counts().to_dict(),
          "| patterns:", pd.Series([p["pattern"] for p in s]).value_counts().to_dict())
print("dev need", DEV_NEED, "| reused earlier jobs:", sum(1 for p in allp if p["target"] in early | test_v3 | dev_old), "| unique jobs", len({p["target"] for p in allp}))
