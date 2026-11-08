"""Run the prespecified text-only stance experiment on unchanged canonical dev data."""

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.data_loader import read_profiles, load_eligible_occupations
from recommender.evaluate import run, save_json
from recommender.history import append_history
from recommender.compare import paired_comparison
from recommender.evaluation import METRICS, aggregate, evaluate_ranking
from recommender.task_preferences import TaskPreferenceExtractor

OUT = ROOT / 'reports/ranking/stance_dev120_v3'
DATA = ROOT / 'data/dev_profiles.jsonl'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(profiles):
    """Review-only text extraction; no annotation fields passed to extractor."""
    extractor = TaskPreferenceExtractor()
    records, profile_rows = [], []
    for p in profiles:
        evidence = extractor.extract((p['text'],))
        usable = [e for e in evidence if e.status == 'usable']
        counts = Counter(e.polarity for e in usable)
        profile_rows.append({'profile_id': p['id'], 'positive': counts['positive'],
            'negative': counts['negative'], 'unknown': sum(e.polarity == 'unknown' for e in evidence),
            'usable': bool(usable), 'mixed_polarity': bool(counts['positive'] and counts['negative']),
            'explicit_negation': bool(re.search(r"\b(?:not|don't|isn't|never|no thanks)\b", p['text'], re.I)),
            'uncertain': any(e.rule == 'uncertain_or_negated_dislike' for e in evidence),
            'background_or_third_person': any(e.rule == 'third_party_or_background_scope' for e in evidence),
            'contradicted': any(e.status == 'unresolved' for e in evidence),
            'familiarity_without_preference': bool(re.search(r"\b(?:stud(?:y|ied)|interned|worked|grew up|family|my dad|my mum)\b", p['text'], re.I)) and not usable})
        records.extend({'profile_id': p['id'], **asdict(e)} for e in evidence)
    write_csv(OUT / 'extracted_spans.csv', records)
    write_csv(OUT / 'extraction_coverage.csv', profile_rows)
    # Blank labels are an independent review template, never inferred truth.
    sample, seen = [], Counter()
    for record in records:
        if seen[record['rule']] < 5:
            sample.append({**record, 'reviewer': '', 'review_polarity': '', 'review_activity_scope': '', 'review_notes': ''})
            seen[record['rule']] += 1
    write_csv(OUT / 'independent_review_template.csv', sample)
    summary = {'profiles': len(profiles), 'clauses': len(records),
               'usable_profiles': sum(r['usable'] for r in profile_rows),
               'polarity_quality': 'UNVERIFIED: no independent labels supplied',
               'profile_counts': {k: sum(bool(r[k]) for r in profile_rows) for k in profile_rows[0] if k != 'profile_id'},
               'clause_polarities': dict(Counter(r['polarity'] for r in records)),
               'rules': dict(Counter(r['rule'] for r in records))}
    save_json(OUT / 'extraction_audit.json', summary)
    (OUT / 'EXTRACTION_AUDIT.md').write_text(
        '# Text-only extraction audit\n\n'+json.dumps(summary, indent=2)+
        '\n\nCounts are automatic rule outputs, not independent ground truth. Familiarity and third-person counts are conservative lexical proxies. Exact normalized activity conflicts are withheld; paraphrase conflicts may be missed. Unknown includes bare mentions, unclear scope and uncertainty. Review spans and their full source context before interpreting labels. The review template has intentionally blank human judgments; precision/recall cannot be reported. Review files are never read by inference.\n', encoding='utf-8')
    return summary, profile_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-count', type=int, default=120,
                        help='Change only with explicit user authorization for another cohort')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    import os
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    original = sha(DATA)
    profiles = read_profiles(DATA)
    if len(profiles) != args.expected_count or len({p['id'] for p in profiles}) != len(profiles):
        (OUT / 'FEASIBILITY.md').write_text(
            f'# Feasibility gate\n\nBLOCKED: canonical data/dev_profiles.jsonl has {len(profiles)} records; required count={args.expected_count}. SHA-256={original}. No dataset was substituted or modified. No evaluations or history rows were created by this attempt. Restore the intended canonical dataset or explicitly authorize a different cohort.\n', encoding='utf-8')
        parser.exit(1, f'Dataset count gate: found {len(profiles)}, expected {args.expected_count}.\n')
    summary, coverage = audit(profiles)
    assert sha(DATA) == original
    feasible = summary['usable_profiles'] / len(profiles) >= 0.10
    (OUT / 'FEASIBILITY.md').write_text(
        f'# Feasibility\n\nCanonical profiles={len(profiles)}; SHA-256={original}. Usable profile coverage={summary["usable_profiles"]}/{len(profiles)}. Operational negligible-signal gate is <10% profile coverage, fixed before results. Coverage gate={"PASS" if feasible else "STOP"}. Polarity/scope accuracy is UNVERIFIED; no independent review labels exist. No suitability or novelty claims. Rules inspect raw user text only; scoring never receives annotations.\n', encoding='utf-8')
    if not feasible:
        parser.exit(1, 'Negligible extraction signal; collect independent annotations before tuning.\n')
    if args.audit_only:
        print(json.dumps(summary, indent=2))
        return
    conditions = {'frozen_m4': 'm4_dense', 'positive_only': 'stance_positive',
                  'signed_stance': 'stance_signed', 'cosine': 'm1_cosine',
                  'cosine_stance': 'cosine_stance'}
    # Archive the implemented source, not alternate evaluation datasets.
    source_dir = OUT / 'source_snapshot'
    source_dir.mkdir(exist_ok=True)
    for directory in ('recommender',):
        for p in Path(directory).glob('*.py'):
            (source_dir / p.name).write_bytes(p.read_bytes())
    metrics = {}
    for name, config in conditions.items():
        assert sha(DATA) == original, 'Canonical dataset changed'
        print('Starting '+name, flush=True)
        metrics[name] = run(Path('configs/m2.json'), OUT / name,
                            ranking_config=Path('configs/ranking') / (config+'.json'))
        append_history(OUT / name, history=Path('notebooks/ranking_evaluation_history_clean.csv'))
        print(name, metrics[name], flush=True)
    # Verify rankings and saved metrics independently; inactive profiles remain
    # in the denominator with empty rankings.
    pool = {o.code for o in load_eligible_occupations(Path('data/occupations.csv'))}
    status_counts, grouped = {}, []
    pmap = {p['id']: p for p in profiles}
    coverage_map = {r['profile_id']: r for r in coverage}
    for name in conditions:
        folder = OUT / name
        by_id = {pid: [] for pid in pmap}
        for line in (folder / 'predictions.jsonl').read_text().splitlines():
            row = json.loads(line)
            by_id[row['profile_id']].append(row)
        per_profile = json.loads((folder / 'per_profile.json').read_text())
        recomputed = []
        status_counts[name] = dict(Counter(r['status'] for r in per_profile))
        for row in per_profile:
            pid = row['profile_id']; p = pmap[pid]
            items = sorted(by_id[pid], key=lambda r: r['rank'])
            result = evaluate_ranking([r['code'] for r in items], p['target'], p['related'], pool)
            for m in METRICS:
                assert abs(result[m]-row[m]) < 1e-12
            recomputed.append(result)
            c = coverage_map[pid]
            group = 'mixed' if c['mixed_polarity'] else 'positive_only' if c['positive'] else 'negative_only' if c['negative'] else 'no_usable'
            grouped.append({'condition': name, 'profile_id': pid, 'polarity_group': group, **{m: result[m] for m in METRICS}})
        for m in METRICS:
            assert abs(aggregate(recomputed)[m]-metrics[name][m]) < 1e-12
    comparisons = {}
    for control, treatment in [('frozen_m4', 'positive_only'), ('positive_only', 'signed_stance'),
                               ('frozen_m4', 'signed_stance'), ('cosine', 'cosine_stance')]:
        summary, diffs = paired_comparison(OUT / control, OUT / treatment)
        key = treatment+'_vs_'+control
        comparisons[key] = summary
        write_csv(OUT / (key+'_paired.csv'), diffs)
    save_json(OUT / 'comparisons.json', comparisons)
    save_json(OUT / 'status_counts.json', status_counts)
    write_csv(OUT / 'subgroup_per_profile.csv', grouped)
    write_csv(OUT / 'summary.csv', [{'condition': k, **{m:v[m] for m in METRICS}} for k,v in metrics.items()])
    # Diagnostic replay uses identical raw text only, never label fields.
    from recommender.ranker import RankingPipeline
    from recommender.components import ComponentInput
    pipeline = RankingPipeline(config=Path('configs/ranking/stance_signed.json'))
    component = pipeline._components['m4']
    traces, inventory = [], []
    for p in profiles:
        ranking = component.score(ComponentInput(user_sentences=(p['text'],)))
        traces.append({'profile_id': p['id'], 'status': ranking.status, **component.last_trace})
        inventory.extend({'profile_id': p['id'], 'code': r.code, 'task_count': len(component.indices[r.code]), 'score': r.score} for r in ranking.occupations)
    save_json(OUT / 'stance_traces.json', traces)
    write_csv(OUT / 'task_inventory_scores.csv', inventory)
    # Rank correlation is descriptive; task-count dependence is not adjusted.
    import pandas as pd
    inv = pd.DataFrame(inventory)
    corr = float(inv[['task_count','score']].corr(method='spearman').iloc[0,1]) if not inv.empty else None
    save_json(OUT / 'task_inventory_bias.json', {'pooled_spearman_task_count_score': corr,
        'method': 'descriptive pooled profile-occupation pairs; correlated observations; no aggregation change',
        'task_covered_occupations':len(component.codes), 'eligible_occupations':len(pool),
        'missing_task_codes': sorted(pool-set(component.codes))})
    assert sha(DATA) == original
    table = '| Condition | Hit@5 | MRR | NDCG@10 |\n|---|---:|---:|---:|\n'
    for name,v in metrics.items(): table += f'| {name} | {v["hit_at_5"]:.6f} | {v["mrr"]:.6f} | {v["ndcg_at_10"]:.6f} |\n'
    (OUT / 'DECISION.md').write_text(
        '# Stance experiment results\n\n'+table+
        '\nKeep production defaults unchanged. Stance is inferred inside the independent M4 component, never supplied by a dataset property. All five conditions use unchanged canonical dev data and matching oracle inputs/pool/metrics. New history rows were appended; historical results and the frozen baseline are preserved.\n\nPolarity quality is UNVERIFIED without independent labels. Review the blank independent_review_template.csv before accuracy claims or further tuning. Exact activity contradictions are withheld, paraphrases can be missed, and scoped rules have limited coverage. This is static synthetic evaluation, not demonstrated skills, career suitability, novel occupation discovery or multi-turn improvement.\n\nFull six metrics are in summary.csv; comparisons.json contains paired profile bootstrap intervals and per-profile wins/losses. Status counts include unusable queries; subgroup_per_profile.csv permits observed-polarity breakdowns. Task inventory association and top-ten signed task traces are diagnostic artifacts.\n', encoding='utf-8')
    print('Completed experiment: '+str(OUT), flush=True)


if __name__ == '__main__':
    main()
