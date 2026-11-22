"""v3 test/dev set.
- Targets: Job Zone 4-5 pool, stratified by each job's raw dominant RIASEC type (so Realistic jobs are included).
- People are not job averages: hidden_slots = target z-row + N(0, 0.3) personal variation. The target stays the job.
- Tier 2 writers never see the job title or description (no vocabulary leak).
- Plants are two-sided: contradictions whose genuine side is a like OR a dislike, and familiarity cases where the
  background is misleading (low true score) OR genuine (high true score, probing should keep it).
- 1 in 4 Tier 1 profiles mention a background that matches a genuine interest, so 'mentions family/course' is not a cue.
New seed so nothing tuned earlier has seen these targets."""
import pandas as pd, numpy as np, json, os
import sys
EXTRA = len(sys.argv) > 1 and sys.argv[1] == "extra"
OUT = "/home/claude/data/work/v3extra" if EXTRA else "/home/claude/data/work/v3"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2027 if EXTRA else 2026)
T = ["R", "I", "A", "S", "E", "C"]
df = pd.read_csv("/home/claude/data/out/occupations.csv")
df = df[df.has_riasec & (df.job_zone >= 4) & (df.n_tasks >= 8) & ~df.Title.str.contains("All Other")].copy()
old_targets = set()
for f in ["/home/claude/data/out/profiles.jsonl", "/home/claude/data/out/dev_profiles.jsonl",
          "/home/claude/data/work/old_profiles/profiles.jsonl", "/home/claude/data/work/old_profiles/dev_profiles.jsonl"]:
    if os.path.exists(f): old_targets |= {json.loads(l)["target"] for l in open(f)}
if EXTRA:   # v2 targets (the set before v3) were looked at during development too
    old_targets |= set(json.load(open("/home/claude/data/work/tier1_targets.json")).values())
    old_targets |= {x["target"] for x in json.load(open("/home/claude/data/work/tier2_inputs.json"))}
df = df[~df.code.isin(old_targets)]                       # never reuse a target that earlier checks looked at
df["dom"] = df[T].idxmax(axis=1)
per = {t: list(rng.permutation(df.index[df.dom == t])) for t in T}
order = []
while any(per.values()):                                   # round-robin over dominant types
    for t in rng.permutation(T):
        if per[t]: order.append(per[t].pop())
picked = df.loc[order[:40 if EXTRA else 120]]
def zrow(r): return {t: float(r[f"z_{t}"]) for t in T}
def person(r):
    return {t: round(v + float(rng.normal(0, 0.3)), 2) for t, v in zrow(r).items()}

rows = list(picked.iterrows())
test_t1, test_t2, dev_t1, dev_t2 = (([], [], rows[0:24], rows[24:40]) if EXTRA else (rows[0:60], rows[60:100], rows[100:112], rows[112:120]))
tier1, t2in = {}, []
for ids, chunk in (([f"P{i+1:03d}" for i in range(60)], test_t1), ([f"D{i+(21 if EXTRA else 1):03d}" for i in range(len(dev_t1))], dev_t1)):
    for pid, (_, r) in zip(ids, chunk):
        n = int(pid[1:])
        tier1[pid] = {"hidden_slots": person(r), "target": r.code, "target_profile": {t: round(v, 2) for t, v in zrow(r).items()},
                      "style": "messy" if n % 3 == 0 else "full", "background": n % 4 == 1}
plans = ["contradiction_like", "contradiction_dislike", "familiarity_misleading", "familiarity_genuine", "both"]
for ids, chunk in (([f"P{61+i:03d}" for i in range(40)], test_t2), ([f"D{(45 if EXTRA else 13)+i:03d}" for i in range(len(dev_t2))], dev_t2)):
    for k, (pid, (_, r)) in enumerate(zip(ids, chunk)):
        h = person(r); srt = sorted(T, key=lambda t: h[t])
        t2in.append({"id": pid, "target": r.code, "target_profile": {t: round(v, 2) for t, v in zrow(r).items()},
                     "hidden_slots": h, "plan": plans[k % 5],
                     "lowest": srt[:2], "highest": srt[::-1][:2],
                     "neg_strong": [t for t in T if h[t] <= -0.6], "pos_strong": [t for t in T if h[t] >= 0.6]})
json.dump({k: {**v["hidden_slots"], "style": v["style"], "background": v["background"]} for k, v in tier1.items()},
          open(f"{OUT}/tier1_numbers.json", "w"), indent=1)
json.dump(tier1, open(f"{OUT}/tier1_meta.json", "w"), indent=1)
# what the Tier 2 writer sees: numbers and plan only, never the title or description
json.dump([{k: v for k, v in x.items() if k not in ("target", "target_profile")} for x in t2in], open(f"{OUT}/tier2_inputs_blind.json", "w"), indent=1)
json.dump(t2in, open(f"{OUT}/tier2_meta.json", "w"), indent=1)
print("dominant types:", picked.dom.value_counts().to_dict(), "| zones:", picked.job_zone.value_counts().to_dict(),
      "| majors:", picked.code.str[:2].nunique())
print("plans:", pd.Series([x["plan"] for x in t2in]).value_counts().to_dict())
old = set()
for f in ["/home/claude/data/out/profiles.jsonl", "/home/claude/data/out/dev_profiles.jsonl"]:
    old |= {json.loads(l)["target"] for l in open(f)}
print("overlap with previous targets:", len(old & set(picked.code)))
