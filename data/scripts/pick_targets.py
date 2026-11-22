"""Pick target occupations for Tier 1 (60), Tier 2 (40), dev (20).
Targets restricted to O*NET Job Zones 4-5 (usually need a bachelor's degree or more), to match
SMU students; spread across SOC major groups by round-robin.
Writes the agent inputs: tier1_numbers.json holds ONLY ids + six numbers + a style flag (no titles)."""
import pandas as pd, numpy as np, json, os
OUT = "/home/claude/data/work"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(425)
T = ["R", "I", "A", "S", "E", "C"]
df = pd.read_csv("/home/claude/data/out/occupations.csv")
df = df[df.has_riasec & (df.job_zone >= 4) & (df.n_tasks >= 8) & ~df.Title.str.contains("All Other")].copy()
df["major"] = df.code.str[:2]
groups = {g: list(rng.permutation(sub.index)) for g, sub in df.groupby("major")}
order = []
while any(groups.values()):
    for g in rng.permutation(sorted(groups)):      # new group order each round
        if groups[g]: order.append(groups[g].pop())
picked = df.loc[order[:120]]
t1, t2 = picked.iloc[0:120:2][:60], picked.iloc[1:120:2][:40]
dev = picked.drop(t1.index).drop(t2.index).iloc[:20]
def zrow(r): return {t: round(float(r[f"z_{t}"]), 2) for t in T}

tier1 = [{"id": f"P{i+1:03d}", "hidden_slots": zrow(r), "target": r.code} for i, (_, r) in enumerate(t1.iterrows())]
dev1 = [{"id": f"D{i+1:03d}", "hidden_slots": zrow(r), "target": r.code} for i, (_, r) in enumerate(dev.iloc[:12].iterrows())]
# one in three Tier 1 profiles is a short, messy first turn (P003, P006, ... and D003, D006, ...)
style = lambda pid: "messy" if int(pid[1:]) % 3 == 0 else "full"
json.dump({p["id"]: {**p["hidden_slots"], "style": style(p["id"])} for p in tier1 + dev1},
          open(f"{OUT}/tier1_numbers.json", "w"), indent=1)
json.dump({p["id"]: p["target"] for p in tier1 + dev1}, open(f"{OUT}/tier1_targets.json", "w"), indent=1)

plans = ["contradiction", "familiarity", "both"]
tier2 = []
for i, (_, r) in enumerate(list(t2.iterrows()) + list(dev.iloc[12:].iterrows())):
    pid = f"P{61+i:03d}" if i < 40 else f"D{13+i-40:03d}"
    z = zrow(r); low = sorted(T, key=lambda t: z[t])
    tier2.append({"id": pid, "target": r.code, "title": r.Title, "description": r.Description,
                  "hidden_slots": z, "plan": plans[i % 3],
                  "familiarity_slot_candidates": low[:2], "top_types": sorted(T, key=lambda t: -z[t])[:3]})
json.dump(tier2, open(f"{OUT}/tier2_inputs.json", "w"), indent=1)
print(len(tier1), len(dev1), len(tier2), "| majors t1:", t1.major.nunique(), "t2:", t2.major.nunique(),
      "| zones:", picked.job_zone.value_counts().to_dict())
print("sample:", "; ".join(t1.Title.head(12)))
