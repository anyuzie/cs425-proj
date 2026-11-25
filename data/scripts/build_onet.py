"""O*NET 31.0 -> occupations.csv, tasks.csv, activities.jsonl"""
import pandas as pd, json, os
D = "/home/claude/data/onet"; OUT = "/home/claude/data/out"; os.makedirs(OUT, exist_ok=True)
T = ["R", "I", "A", "S", "E", "C"]
NAME = {"Realistic": "R", "Investigative": "I", "Artistic": "A", "Social": "S", "Enterprising": "E", "Conventional": "C"}

occ = pd.read_csv(f"{D}/occupation_data.csv")
ci = pd.read_csv(f"{D}/career_interest_types.csv")
oi = ci[ci["Scale ID"] == "OI"]                      # numeric scores only; IH rows are high-point codes
piv = oi.pivot_table(index="O*NET-SOC Code", columns="Element Name", values="Data Value").rename(columns=NAME)[T]
z = (piv - piv.mean()) / piv.std(ddof=0)
z.columns = [f"z_{c}" for c in T]
tasks = pd.read_csv(f"{D}/task_statements.csv")
titles = pd.read_csv(f"{D}/job_titles.csv")
alt = titles.groupby("O*NET-SOC Code")["Job Title"].apply(lambda s: " | ".join(s)).rename("alt_titles")
tk = tasks.groupby("O*NET-SOC Code")["Task"].apply(lambda s: " ".join(s)).rename("tasks_text")

df = occ.set_index("O*NET-SOC Code").join(piv).join(z).join(alt).join(tk)
df["n_tasks"] = tasks.groupby("O*NET-SOC Code").size()
df["has_riasec"] = df["R"].notna()
df["bm25_doc"] = (df["Title"] + " " + df["alt_titles"].fillna("").str.replace(" | ", " ", regex=False)
                  + " " + df["Description"] + " " + df["tasks_text"].fillna(""))
jz = pd.read_csv("/home/claude/data/onet2/job_zones.csv").set_index("O*NET-SOC Code")["Job Zone"].rename("job_zone")
df = df.join(jz)
rel = pd.read_csv("/home/claude/data/onet2/related_occupations.csv")
rel.rename(columns={"O*NET-SOC Code": "code", "Related O*NET-SOC Code": "related_code", "Relatedness Tier": "tier", "Index": "rank"})[
    ["code", "related_code", "tier", "rank"]].to_csv(f"{OUT}/related_occupations.csv", index=False)
df.drop(columns=["tasks_text"]).reset_index().rename(columns={"O*NET-SOC Code": "code"}).to_csv(f"{OUT}/occupations.csv", index=False)
tasks.rename(columns={"O*NET-SOC Code": "code"})[["code", "Title", "Task ID", "Task", "Task Type"]].to_csv(f"{OUT}/tasks.csv", index=False)

# 188 activities. Career Type rows carry their type; Specific Area rows mapped to one RIASEC type
# (LLM-assigned, still to be reviewed; the 31.0 file labels these by Specific Area, not RIASEC).
AREA = {
 "Mechanics/Electronics": "R", "Construction/Woodwork": "R", "Transportation/Machine Operation": "R",
 "Physical/Manual Labor": "R", "Protective Service": "R", "Agriculture": "R", "Nature/Outdoors": "R",
 "Animal Service": "R", "Athletics": "R", "Engineering": "R",
 "Physical Science": "I", "Life Science": "I", "Medical Science": "I", "Social Science": "I",
 "Humanities": "I", "Mathematics/Statistics": "I", "Information Technology": "I",
 "Visual Arts": "A", "Applied Arts and Design": "A", "Performing Arts": "A", "Music": "A",
 "Creative Writing": "A", "Media": "A", "Culinary Art": "A",
 "Teaching/Education": "S", "Social Service": "S", "Health Care Service": "S", "Religious Activities": "S",
 "Personal Service": "S", "Professional Advising": "S", "Human Resources": "S",
 "Business Initiatives": "E", "Sales": "E", "Marketing/Advertising": "E", "Finance": "E",
 "Management/Administration": "E", "Public Speaking": "E", "Politics": "E", "Law": "E",
 "Accounting": "C", "Office Work": "C",
}
a = pd.read_csv(f"{D}/interests_illustrative_activities.csv")
with open(f"{OUT}/activities.jsonl", "w") as f:
    for i, r in a.iterrows():
        n = r["Element Name"]; ty = NAME.get(n) or AREA[n]
        f.write(json.dumps({"seed_id": f"IA_{i+1:03d}", "activity": r["Activity"], "area": n,
                            "interest_type": r["Interest Type"], "type": ty}) + "\n")
print("occupations", len(df), "with RIASEC", int(df.has_riasec.sum()), "| tasks", len(tasks), "| titles", len(titles))
print("activities per type", a["Element Name"].map(lambda n: NAME.get(n) or AREA[n]).value_counts().to_dict())
print(z.describe().loc[["mean", "std"]].round(2))
