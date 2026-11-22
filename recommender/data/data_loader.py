"""Validate static ranking inputs and audit split metadata without benchmarking."""

import csv
import hashlib
import json
import math
from pathlib import Path

from recommender.schemas import TYPES, Occupation, validate_vector

FILES = ("occupations.csv", "tasks.csv", "related_occupations.csv", "labels.json",
         "dev_profiles.jsonl", "profiles.jsonl")
TIERS = ("Primary-Short", "Primary-Long", "Supplemental")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path, columns: set[str]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(columns <= set(reader.fieldnames or []), f"{path.name}: missing columns {sorted(columns - set(reader.fieldnames or []))}")
        return list(reader)


def read_profiles(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                require(isinstance(row, dict), "profile must be an object")
                rows.append(row)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path.name}:{number}: {error}") from error
    return rows


def finite_number(value: str, context: str) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{context}: invalid number {value!r}") from error
    require(math.isfinite(result), f"{context}: non-finite number")
    return result


def load_eligible_occupations(path: Path) -> list[Occupation]:
    """Load RIASEC ranking inputs only; keep supplied z-scores and exclude unrated rows."""
    rows = read_csv(Path(path), {"code", "Title", "job_zone", *(f"z_{t}" for t in TYPES)})
    codes, eligible = set(), []
    for row in rows:
        code = row["code"]
        require(isinstance(code, str) and bool(code.strip()) and code not in codes,
                f"occupations.csv: empty or duplicate code {code!r}")
        codes.add(code)
        require(isinstance(row["Title"], str) and bool(row["Title"].strip()), f"{code}: missing title")
        require(all(isinstance(row[key], str) for key in ("job_zone", *(f"z_{t}" for t in TYPES))),
                f"{code}: malformed occupation row")
        values = [row[f"z_{t}"].strip() for t in TYPES]
        require(not any(values) or all(values), f"{code}: incomplete RIASEC z-scores")
        z_scores = {t: finite_number(row[f"z_{t}"], f"{code}/z_{t}") for t in TYPES} if all(values) else None
        zone = finite_number(row["job_zone"], f"{code}/job_zone") if row["job_zone"].strip() else None
        require(zone is None or zone in {1, 2, 3, 4, 5}, f"{code}: invalid job zone")
        if zone in {4, 5} and z_scores is not None:
            eligible.append(Occupation(code, row["Title"], z_scores, row.get("bm25_doc")))
    require(bool(eligible), "occupations.csv: no eligible occupations")
    return eligible


def audit_data(data_dir: Path, expected_counts: dict[str, int] | None = None) -> dict:
    data_dir = Path(data_dir)
    for name in FILES:
        require((data_dir / name).is_file(), f"missing dataset: {name}")
    labels = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    require(labels.get("types") == list(TYPES), "labels.json: RIASEC order must be R,I,A,S,E,C")
    columns = {"code", "Title", "job_zone", "has_riasec", "bm25_doc", *TYPES, *(f"z_{t}" for t in TYPES)}
    occupations = read_csv(data_dir / "occupations.csv", columns)
    codes, eligible = set(), set()
    missing = {key: 0 for key in sorted(columns)}
    rated = 0
    for row in occupations:
        code = row["code"]
        require(bool(code) and code not in codes, f"occupations.csv: empty or duplicate code {code!r}")
        codes.add(code)
        for key in missing:
            missing[key] += int(not row[key].strip())
        require(bool(row["Title"].strip()), f"{code}: missing title")
        require(row["has_riasec"].lower() in {"true", "false"}, f"{code}: invalid has_riasec")
        has_scores = row["has_riasec"].lower() == "true"
        raw = [row[t].strip() for t in TYPES]
        zs = [row[f"z_{t}"].strip() for t in TYPES]
        require((all(raw) and all(zs)) if has_scores else not any(raw + zs), f"{code}: incomplete/inconsistent RIASEC scores")
        if has_scores:
            rated += 1
            for t in TYPES:
                value = finite_number(row[t], f"{code}/{t}")
                require(1 <= value <= 7, f"{code}/{t}: raw score outside [1,7]")
                finite_number(row[f"z_{t}"], f"{code}/z_{t}")
        zone = finite_number(row["job_zone"], f"{code}/job_zone") if row["job_zone"].strip() else None
        require(zone is None or zone in {1, 2, 3, 4, 5}, f"{code}: invalid job zone")
        if zone in {4, 5} and has_scores:
            require(bool(row["bm25_doc"].strip()), f"{code}: eligible occupation has no bm25_doc")
            eligible.add(code)
    tasks = read_csv(data_dir / "tasks.csv", {"code", "Task ID", "Task"})
    task_keys, task_codes = set(), set()
    for row in tasks:
        key = (row["code"], row["Task ID"])
        require(row["code"] in codes, f"tasks.csv: unknown occupation {row['code']}")
        require(bool(row["Task ID"]) and bool(row["Task"].strip()), "tasks.csv: missing task ID/text")
        require(key not in task_keys, f"tasks.csv: duplicate task key {key}")
        task_keys.add(key)
        task_codes.add(row["code"])
    related = read_csv(data_dir / "related_occupations.csv", {"code", "related_code", "tier", "rank"})
    relation_keys = set()
    for row in related:
        require(row["code"] in codes and row["related_code"] in codes, "related_occupations.csv: unknown occupation")
        require(row["tier"] in TIERS, f"related_occupations.csv: invalid tier {row['tier']}")
        key = (row["code"], row["related_code"], row["tier"])
        require(key not in relation_keys, f"related_occupations.csv: duplicate relation {key}")
        relation_keys.add(key)
        require(row["rank"].isdigit() and int(row["rank"]) > 0, "related_occupations.csv: invalid rank")
    splits, id_sets, target_sets = {}, {}, {}
    for name in ("dev_profiles.jsonl", "profiles.jsonl"):
        profiles = read_profiles(data_dir / name)
        ids, targets = set(), set()
        schema = set().union(*(set(p) for p in profiles))
        for p in profiles:
            require({"id", "text", "hidden_slots", "target", "related"} <= set(p), f"{name}: missing profile fields")
            require(isinstance(p["id"], str) and bool(p["id"]) and p["id"] not in ids, f"{name}: invalid/duplicate profile ID")
            ids.add(p["id"])
            require(isinstance(p["text"], str) and bool(p["text"].strip()), f"{name}: empty profile text")
            validate_vector(p["hidden_slots"], runtime=False)
            require(p["target"] in eligible, f"{name}/{p['id']}: target is missing or ineligible")
            targets.add(p["target"])
            require(isinstance(p["related"], dict), f"{name}: invalid related labels")
            for tier in ("primary_short", "primary_long", "supplemental"):
                relations = p["related"].get(tier)
                require(isinstance(relations, list) and all(isinstance(c, str) and c in codes for c in relations), f"{name}/{p['id']}: invalid related labels {tier}")
        id_sets[name], target_sets[name] = ids, targets
        splits[name] = {"count": len(profiles), "unique_targets": len(targets), "eligible_targets": len(profiles),
                        "fields": sorted(schema), "has_ordered_turns": all(isinstance(p.get("turns"), list) and bool(p["turns"]) for p in profiles),
                        "has_evidence_states": all(isinstance(p.get("evidence"), list) and bool(p["evidence"]) for p in profiles)}
    require(not (id_sets["dev_profiles.jsonl"] & id_sets["profiles.jsonl"]), "profile IDs overlap across splits")
    counts = {"occupations": len(occupations), "rated": rated, "eligible": len(eligible), "tasks": len(tasks),
              "eligible_tasks": sum(t["code"] in eligible for t in tasks), "related": len(related),
              "dev": splits["dev_profiles.jsonl"]["count"], "held_out": splits["profiles.jsonl"]["count"]}
    for key, expected in (expected_counts or {}).items():
        require(counts[key] == expected, f"{key}: expected {expected}, got {counts[key]}")
    return {"counts": counts, "missingness": missing, "eligible_without_tasks": sorted(eligible - task_codes),
            "splits": splits, "target_overlap_count": len(target_sets["dev_profiles.jsonl"] & target_sets["profiles.jsonl"]),
            "sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in FILES}}
