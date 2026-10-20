"""Read-only diagnostic helpers for the executed analysis notebook.

Uses production scoring/metrics; writes only to reports/ranking/diagnostics.
Specificity proposals are deterministic text-only heuristics, never human labels.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from recommender.bm25_ranker import BM25Ranker, BASELINE_PARAMETERS as BM25_PARAMETERS, tokenize
from recommender.components import ComponentInput
from recommender.data_loader import load_eligible_occupations, read_profiles
from recommender.evaluation import METRICS, aggregate, evaluate_ranking, relevance_labels
from recommender.schemas import TYPES, OracleInput
from recommender.structured import StructuredRanker

SEED = 425
RUNS = {"m1": "structured", "m3": "bm25", "m4": "dense", "m1_m3": "hybrid",
        "m1_m4": "structured_dense", "m3_m4": "bm25_dense", "fusion": "three_way"}
NAMES = {"m1": "M1", "m3": "BM25", "m4": "MiniLM", "m1_m3": "M1+BM25",
         "m1_m4": "M1+MiniLM", "m3_m4": "BM25+MiniLM", "fusion": "Three-way RRF"}
OUT = Path("reports/ranking/diagnostics")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_fingerprints():
    paths = [Path("data/dev_profiles.jsonl"), Path("data/occupations.csv"), Path("data/tasks.csv"),
             Path("data/labels.json"), Path("reports/ranking/evaluation_history.csv")]
    for folder in ("reports/ranking/m2", "reports/ranking/m3", "reports/ranking/m4", "recommender",
                   "artifacts/ranking/minilm_mean_tasks_v1", "artifacts/ranking/models/all-MiniLM-L6-v2_1110a243"):
        paths.extend(p for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".cache" not in p.parts)
    return {p.as_posix(): sha(p) for p in sorted(set(paths))}


def interval(values, seed=SEED, samples=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def subgroup_metrics(long, column):
    rows = []
    for (group, method), part in long.groupby([column, "method"], dropna=False, observed=True):
        for metric in ("hit_at_5", "mrr", "ndcg_at_10"):
            low, high = interval(part[metric])
            rows.append({"attribute": column, "group": str(group), "method": method,
                         "n": len(part), "metric": metric, "mean": part[metric].mean(),
                         "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def load_analysis():
    profiles = read_profiles(Path("data/dev_profiles.jsonl"))
    ids = [p["id"] for p in profiles]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development profile IDs")
    labels = json.loads(Path("data/labels.json").read_text(encoding="utf-8"))
    if labels["types"] != list(TYPES):
        raise ValueError("Unexpected RIASEC dimension ordering")
    occupations = load_eligible_occupations(Path("data/occupations.csv"))
    occupation = {o.code: o for o in occupations}
    pool = set(occupation)
    catalog = pd.read_csv("data/occupations.csv", dtype={"code": str})
    known_codes = set(catalog.code)
    integrity = {"profiles": len(ids), "unique_ids": len(set(ids)), "eligible_occupations": len(pool),
                 "missing_targets": 0, "ineligible_targets": 0, "unknown_related_codes": 0,
                 "related_codes_outside_pool": 0, "nonfinite_or_missing_riasec": 0,
                 "missing_text": 0, "excluded_profiles": 0}
    for p in profiles:
        integrity["missing_targets"] += int(not p.get("target"))
        integrity["ineligible_targets"] += int(bool(p.get("target")) and p["target"] not in pool)
        integrity["missing_text"] += int(not isinstance(p.get("text"), str) or not p["text"].strip())
        for field in ("hidden_slots", "target_profile"):
            vector = p.get(field, {})
            invalid = set(vector) != set(TYPES) or any(not isinstance(vector.get(t), (int, float)) or not np.isfinite(vector[t]) for t in TYPES)
            integrity["nonfinite_or_missing_riasec"] += int(invalid)
        for codes in p.get("related", {}).values():
            integrity["unknown_related_codes"] += sum(c not in known_codes for c in codes)
            integrity["related_codes_outside_pool"] += sum(c not in pool for c in codes)
    if integrity["nonfinite_or_missing_riasec"] or integrity["unknown_related_codes"]:
        raise ValueError(f"Invalid source data; no silent exclusions: {integrity}")
    requests = None
    manifests, predictions, metric_rows, grades = {}, {}, [], []
    pmap = {p["id"]: p for p in profiles}
    reference = None
    for method, folder in RUNS.items():
        directory = Path("reports/ranking/m4") / folder
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        signature = {key: manifest[key] for key in ("split", "input_mode", "pool", "dataset_sha256", "inputs_sha256", "evaluator_version", "metric_definitions", "eligibility", "ranking_depth")}
        if signature["split"] != "development" or signature["input_mode"] != "oracle" or set(signature["pool"]) != pool:
            raise ValueError(f"Incompatible split/mode/pool: {method}")
        if reference is not None and signature != reference:
            raise ValueError(f"Incompatible run snapshots: {method}")
        reference = signature
        for path, fingerprint in manifest["dataset_sha256"].items():
            if sha(path) != fingerprint:
                raise ValueError(f"Dataset differs from saved run: {path}")
        for path, fingerprint in manifest.get("auxiliary_dataset_sha256", {}).items():
            if sha(path) != fingerprint:
                raise ValueError(f"Auxiliary dataset differs: {path}")
        if sha(directory / "inputs.json") != manifest["inputs_sha256"] or sha(directory / "config.json") != manifest["config_sha256"]:
            raise ValueError(f"Saved config/input hash mismatch: {method}")
        inputs = json.loads((directory / "inputs.json").read_text())
        if {r["profile_id"] for r in inputs} != set(ids) or len(inputs) != len(ids):
            raise ValueError("Prediction input profile set mismatch")
        for r in inputs:
            p = pmap[r["profile_id"]]
            if r["interests"] != p["hidden_slots"] or r["user_sentences"] != [p["text"]]:
                raise ValueError("Saved input is not the frozen development request")
        if requests is not None and inputs != requests:
            raise ValueError("Methods received different requests")
        requests = inputs
        resolved = manifest["resolved_ranking"]
        enabled = {key for key, value in resolved["components"].items() if value is not None}
        expected = {"m1": {"m1"}, "m3": {"m3"}, "m4": {"m4"}, "m1_m3": {"m1", "m3"},
                    "m1_m4": {"m1", "m4"}, "m3_m4": {"m3", "m4"}, "fusion": {"m1", "m3", "m4"}}[method]
        if enabled != expected or resolved["fusion"] != {"algorithm": "rrf", "k": 60}:
            raise ValueError(f"Unexpected component/fusion selection: {method}")
        for key, value in resolved["components"].items():
            if value and (value["algorithm"], value["variant"]) != {"m1": ("weighted_dot_product", "baseline_v1"), "m3": ("bm25", "baseline_v1"), "m4": ("minilm", "mean_tasks_v1")}[key]:
                raise ValueError("Unrecognized scoring variant")
        rows = {pid: [] for pid in ids}
        with (directory / "predictions.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                if r["profile_id"] not in rows or r["run_id"] != manifest["run_id"] or r["code"] not in pool:
                    raise ValueError("Invalid joined prediction")
                for signal in r["signals"].values():
                    if not np.isfinite(signal["score"]):
                        raise ValueError("Nonfinite component score")
                keep = {key: r[key] for key in ("code", "rank", "signals", "rrf_score")}
                keep["evidence"] = r["evidence"] if r["rank"] <= 3 or r["code"] == pmap[r["profile_id"]]["target"] else []
                rows[r["profile_id"]].append(keep)
        saved_rows = json.loads((directory / "per_profile.json").read_text())
        saved = {r["profile_id"]: r for r in saved_rows}
        if len(saved_rows) != len(ids) or set(saved) != set(ids):
            raise ValueError("Missing/duplicate per-profile metric records")
        recomputed = []
        for pid, items in rows.items():
            items.sort(key=lambda r: r["rank"])
            if [r["rank"] for r in items] != list(range(1, len(items) + 1)):
                raise ValueError("Noncontiguous saved ranks")
            p = pmap[pid]
            codes = [r["code"] for r in items]
            result = evaluate_ranking(codes, p.get("target"), p["related"], pool)
            for metric in METRICS:
                if not np.isclose(result[metric], saved[pid][metric], atol=1e-12, rtol=0):
                    raise ValueError("Saved metrics do not match existing evaluator")
            relevant = relevance_labels(p.get("target"), p["related"], pool)
            metric_rows.append({"profile_id": pid, "method": method, **result,
                                "relevant_top5": sum(relevant.get(c, 0) > 0 for c in codes[:5]),
                                "relevant_top10": sum(relevant.get(c, 0) > 0 for c in codes[:10])})
            for rank, code in enumerate(codes[:10], 1):
                grades.append({"profile_id": pid, "method": method, "rank": rank, "code": code, "grade": relevant.get(code, 0)})
            recomputed.append(result)
        summary = json.loads((directory / "metrics.json").read_text())
        if any(not np.isclose(aggregate(recomputed)[m], summary[m], atol=1e-12, rtol=0) for m in METRICS):
            raise ValueError("Aggregate metric mismatch")
        manifests[method], predictions[method] = manifest, rows
    history = pd.read_csv("reports/ranking/evaluation_history.csv", keep_default_na=False)
    if history.run_id.duplicated().any() or any(m["run_id"] not in set(history.run_id) for m in manifests.values()):
        raise ValueError("Missing/duplicate evaluation-history registrations")
    return profiles, occupation, catalog, manifests, predictions, pd.DataFrame(metric_rows), pd.DataFrame(grades), integrity


DOMAIN_CUES = {"healthcare": {"patients", "medication", "nursing", "clinic", "diagnose"},
               "education": {"teach", "teaching", "classroom", "lessons", "curriculum"},
               "software": {"software", "programming", "coding", "debugging", "code"},
               "finance": {"accounts", "invoices", "budgets", "reconciled", "bookkeeping"},
               "engineering": {"circuits", "machinery", "repairs", "machines", "equipment"},
               "creative": {"sketch", "posters", "videos", "designing", "drawing"}}


def domain_cues(text):
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    matches = {domain: sorted(words & tokens) for domain, words in DOMAIN_CUES.items() if words & tokens}
    return matches


def text_only_proposal(text):
    """Blind stage: receives no target, scores, metadata or retrieval results."""
    lower = text.lower()
    cues = domain_cues(text)
    exposure = any(word in lower for word in ("my dad", "my mum", "my family", "grew up", "my uncle"))
    uncertainty = any(word in lower for word in ("assumed", "assume", "just what", "maybe", "so i guess", "natural that"))
    count = sum(map(len, cues.values()))
    if exposure and uncertainty:
        return "misleading", "Exposure and uncertainty markers co-occur; possible distracting background, not a verified false preference."
    if any(len(words) >= 2 for words in cues.values()):
        return "specific", f"At least two explicit domain/task cue tokens in one family: {cues}; role identity remains unverified."
    if count:
        return "partially_specific", f"Some domain/task cues narrow a family: {cues}; insufficient evidence of a unique role."
    return "broad", "No predefined domain/task cues; primarily general preferences under this limited heuristic rubric."


def make_profile_table(profiles, occupation, predictions):
    ranker = StructuredRanker()
    lexical = BM25Ranker(list(occupation.values()), BM25_PARAMETERS)
    z = np.array([[occupation[c].z_scores[t] for t in TYPES] for c in sorted(occupation)])
    codes = sorted(occupation)
    znorm = np.linalg.norm(z, axis=1)
    rows = []
    for p in profiles:
        u = np.array([p["hidden_slots"][t] for t in TYPES])
        runtime = ranker.rank_oracle(OracleInput.from_profile(p))
        saved = predictions["m1"][p["id"]]
        if [(r.code, r.signals["structured"].score) for r in runtime.occupations] != [(r["code"], r["signals"]["structured"]["score"]) for r in saved]:
            raise ValueError("Current M1 differs from saved baseline")
        target = occupation.get(p.get("target"))
        actual = np.array([target.z_scores[t] for t in TYPES]) if target else np.full(6, np.nan)
        target_rows = {m: next((r for r in items[p["id"]] if r["code"] == p.get("target")), None) for m, items in predictions.items()}
        row = {"profile_id": p["id"], "target_code": p.get("target"), "target_title": target.title if target else None,
               "text": p["text"], "text_style": p.get("style", "unavailable"), "tier": p.get("tier"),
               "background": p.get("background"), "has_contradiction": bool(p.get("contradiction")),
               "contradiction_slot": (p.get("contradiction") or {}).get("slot"),
               "contradiction_genuine": (p.get("contradiction") or {}).get("genuine"),
               "profile_plan": p.get("plan", "unavailable"),
               "planted_evidence": json.dumps(p.get("planted", [])),
               "listed_twin_codes": json.dumps(p.get("riasec_twins", [])),
               "has_familiarity": any("familiarity" in x for x in p.get("planted", [])),
               "riasec_twin_count": len(p.get("riasec_twins", [])),
               "word_count": len(re.findall(r"\b[\w']+\b", p["text"])), "character_count": len(p["text"]),
               "domain_cue_count": sum(map(len, domain_cues(p["text"]).values())),
               "dominant_absolute": TYPES[int(np.argmax(np.abs(u)))],
               "dominant_positive": TYPES[int(np.argmax(u))] if np.any(u > 0) else "no_positive"}
        # Use the same production sentence splitter even if the optional model is absent.
        row["sentence_count"] = sum(any(c.isalnum() for c in m.group()) for m in re.finditer(r"[^.!?\n]+[.!?]*", p["text"]))
        tokens = re.findall(r"\b[\w']+\b", p["text"].lower())
        row["unique_token_count"] = len(set(tokens))
        row["lexical_diversity"] = len(set(tokens)) / len(tokens) if tokens else np.nan
        row["target_profile_max_abs_difference"] = float(np.max(np.abs(actual - np.array([p["target_profile"][t] for t in TYPES]))))
        row["target_cosine"] = float(np.dot(u, actual) / (np.linalg.norm(u) * np.linalg.norm(actual))) if np.linalg.norm(u) and np.linalg.norm(actual) else np.nan
        row["user_target_l2"] = float(np.linalg.norm(u - actual))
        row["target_vector_norm"] = float(np.linalg.norm(actual))
        winner = runtime.occupations[0]
        row["m1_winner_norm"] = float(np.linalg.norm([winner.occupation_z[t] for t in TYPES]))
        for m, item in target_rows.items():
            row[f"{m}_target_rank"] = item["rank"] if item else np.nan
        row["m1_target_score"] = target_rows["m1"]["signals"]["structured"]["score"] if target_rows["m1"] else np.nan
        scores = {o.code: o.signals["structured"].score for o in runtime.occupations}
        competitors = [v for c, v in scores.items() if c != p.get("target")]
        row["m1_best_competitor_score"] = max(competitors) if competitors else np.nan
        row["m1_margin"] = row["m1_target_score"] - row["m1_best_competitor_score"]
        near = [(c, v) for c, v in scores.items() if abs(v - row["m1_target_score"]) <= .1]
        row["score_band_count"] = len(near)
        row["target_rank_within_score_band"] = 1 + sum(v > row["m1_target_score"] or (v == row["m1_target_score"] and c < p.get("target", "")) for c, v in near) if near else np.nan
        cosines = z @ actual / (znorm * np.linalg.norm(actual)) if np.linalg.norm(actual) else np.full(len(z), np.nan)
        row["calculated_cosine_neighbors"] = sum(c != p.get("target") and similarity >= .99 for c, similarity in zip(codes, cosines))
        output = lexical.score(ComponentInput(user_sentences=(p["text"],)))
        if [(r.code, r.score) for r in output.occupations] != [(r["code"], r["signals"]["lexical"]["score"]) for r in predictions["m3"][p["id"]]]:
            raise ValueError("Actual BM25 does not reproduce saved predictions")
        query = set(tokenize(p["text"]))
        doc = set(tokenize(target.bm25_doc)) if target else set()
        matched = sorted(query & doc)
        indexed = query & lexical._idf.keys()
        row["m3_target_token_overlap"] = len(matched)
        row["m3_query_coverage"] = len(matched) / len(query) if query else np.nan
        row["m3_weighted_overlap"] = sum(lexical._idf[t] for t in matched) / sum(lexical._idf[t] for t in sorted(indexed)) if indexed else np.nan
        row["m3_oov_share"] = len(query - lexical._idf.keys()) / len(query) if query else np.nan
        row["m3_matching_terms"] = ", ".join(matched)
        row["m3_generic_matched_terms"] = ", ".join(t for t in matched if len(lexical._postings[t]) / len(occupation) >= .25)
        bm25_score = {r.code: r.score for r in output.occupations}
        row["m3_target_score"] = bm25_score.get(p.get("target"), 0.0) if target else np.nan
        row["m3_competitor_score"] = max((v for c, v in bm25_score.items() if c != p.get("target")), default=0.0)
        row["m3_margin"] = row["m3_target_score"] - row["m3_competitor_score"]
        competitor = next((r for r in output.occupations if r.code != p.get("target")), None)
        row["m3_competitor_code"] = competitor.code if competitor else None
        row["m3_competitor_token_overlap"] = len(query & set(tokenize(occupation[competitor.code].bm25_doc))) if competitor else np.nan
        dense = target_rows["m4"]
        row["m4_target_score"] = dense["signals"]["semantic"]["score"] if dense else np.nan
        row["m4_target_task_similarity"] = max((e["match_score"] for e in dense["evidence"]), default=np.nan) if dense else np.nan
        row["m4_target_best_task"] = dense["evidence"][0]["text"] if dense and dense["evidence"] else None
        best = next((r for r in predictions["m4"][p["id"]] if r["code"] != p.get("target")), None)
        row["m4_competitor_score"] = best["signals"]["semantic"]["score"] if best else np.nan
        row["m4_competitor_code"] = best["code"] if best else None
        row["m4_competitor_best_task"] = best["evidence"][0]["text"] if best and best["evidence"] else None
        row["m4_margin"] = row["m4_target_score"] - row["m4_competitor_score"]
        row["has_twins"] = row["riasec_twin_count"] > 0
        rows.append(row)
    return pd.DataFrame(rows), lexical


def semantic_diagnostics(profiles, occupation, table, manifests):
    """Observe sentence winners using actual encoder/index; do not redefine ranking."""
    results, sentence_rows = [], []
    try:
        from recommender.dense_config import BASELINE_PARAMETERS
        from recommender.dense_ranker import DenseRanker, sentences, validated_vectors
        dense = DenseRanker.from_cache(list(occupation.values()), BASELINE_PARAMETERS)
    except (ImportError, ValueError, OSError) as error:
        return pd.DataFrame(), pd.DataFrame(), f"Live semantic diagnostics unavailable: {error}. Saved task/score diagnostics remain available."
    saved_metadata = manifests["m4"]["component_metadata"]["m4"]
    if dense.metadata["artifact_sha256"] != saved_metadata["artifact_sha256"]:
        raise ValueError("Current semantic cache differs from analyzed run")
    indexed = table.set_index("profile_id")
    for p in profiles:
        pid = p["id"]
        text = sentences((p["text"],))
        actual = dense.score(ComponentInput(user_sentences=(p["text"],)))
        target = next((r for r in actual.occupations if r.code == p.get("target")), None)
        if target and (target.rank != indexed.loc[pid, "m4_target_rank"] or not np.isclose(target.score, indexed.loc[pid, "m4_target_score"], atol=1e-6, rtol=0)):
            raise ValueError("Live semantic output differs from saved score/rank")
        if not target:
            continue
        query = validated_vectors(dense.encoder.encode(text), len(text), dense.index.d)
        distances, ids = dense.index.search(query, dense.index.ntotal)
        by_task = np.empty((len(text), len(dense.tasks)), dtype="float32")
        for i in range(len(text)):
            by_task[i, ids[i]] = distances[i]
        target_indices = [i for i, task in enumerate(dense.tasks) if task.code == p["target"]]
        target_scores = by_task[:, target_indices]
        winners = [target_indices[i] for i in target_scores.argmax(axis=1)]
        counts = Counter(winners)
        best_sentence_scores = target_scores.max(axis=1)
        if not np.isclose(best_sentence_scores.astype("float64").mean(), target.score, atol=1e-6, rtol=0):
            raise ValueError("Sentence diagnostic does not reconstruct actual aggregation")
        results.append({"profile_id": pid, "target_task_count": len(target_indices),
                        "distinct_winning_tasks": len(counts), "dominant_task_share": max(counts.values()) / len(text),
                        "target_sentence_max_spread": float(np.ptp(best_sentence_scores)),
                        "best_task_minus_aggregate": float(best_sentence_scores.max() - target.score)})
        competitor_code = indexed.loc[pid, "m4_competitor_code"]
        competitor_indices = [i for i, task in enumerate(dense.tasks) if task.code == competitor_code]
        competitor_scores = by_task[:, competitor_indices]
        competitor_winners = [competitor_indices[i] for i in competitor_scores.argmax(axis=1)]
        results[-1]["competitor_dominant_task_share"] = max(Counter(competitor_winners).values()) / len(text)
        for sentence, idx, similarity in zip(text, winners, best_sentence_scores):
            sentence_rows.append({"profile_id": pid, "role": "target", "sentence": sentence, "task_id": dense.tasks[idx].task_id,
                                  "task": dense.tasks[idx].text, "similarity": float(similarity)})
        for sentence, idx, similarity in zip(text, competitor_winners, competitor_scores.max(axis=1)):
            sentence_rows.append({"profile_id": pid, "role": "best competitor", "sentence": sentence,
                                  "task_id": dense.tasks[idx].task_id, "task": dense.tasks[idx].text, "similarity": float(similarity)})
    return pd.DataFrame(results), pd.DataFrame(sentence_rows), "Actual pinned model/cache reused; task embeddings were not rebuilt."


def select_sample(table, n=18):
    """Deterministic greedy stratification; selection metadata never enters judge."""
    features = {}
    for row in table.to_dict("records"):
        features[row["profile_id"]] = {f"style:{row['text_style']}", f"tier:{row['tier']}",
            f"dominant:{row['dominant_positive']}", f"contradiction:{row['has_contradiction']}",
            f"familiarity:{row['has_familiarity']}", f"bm25_hit:{row['m3_target_rank'] <= 5}",
            f"dense_hit:{row['m4_target_rank'] <= 5}"}
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(sorted(features)).tolist()
    selected, covered = [], set()
    while len(selected) < min(n, len(shuffled)):
        candidate = max((pid for pid in shuffled if pid not in selected), key=lambda pid: len(features[pid] - covered))
        selected.append(candidate)
        covered.update(features[candidate])
    return selected
