"""Reproducible development-only M6 stress test; never actual-tracker evidence.

Run from the repository root: python scripts/run_m6_experiment.py
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender.compare import paired_comparison
from recommender.evaluate import run, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reports/ranking/m6/run_v1'))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    source = Path('data/dev_profiles.jsonl')
    profiles = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    cases = []
    strata = ('superseded', 'disputed', 'all_active', 'empty_active')
    for i, profile in enumerate(profiles):
        # Fixed offset, independent of retrieval outcomes and target labels.
        stale = profiles[(i + len(profiles)//2) % len(profiles)]['text']
        texts = [stale, profile['text']]
        stratum = strata[i % len(strata)]
        final_statuses = {'superseded': ['superseded', 'active'],
                          'disputed': ['disputed', 'active'],
                          'all_active': ['active', 'active'],
                          'empty_active': ['disputed', 'disputed']}[stratum]
        # Synthetic bounded fixture vector, explicitly not extractor predictions.
        interests = {k: max(-1.0, min(1.0, v / 3.0)) for k, v in profile['hidden_slots'].items()}
        turns = []
        for turn in range(2):
            statuses = ['active'] if turn == 0 else final_statuses
            turns.append({'text': texts[turn], 'interests': interests,
                          'observed_types': list(interests), 'source_evidence': [
                              {'id': f'e{j+1}', 'turn': j+1, 'text': text, 'status': statuses[j]}
                              for j, text in enumerate(texts[:turn+1])]})
        cases.append({'id': profile['id'], 'stratum': stratum, 'input_mode': 'mock_tracker',
                      'upstream': {'extractor': 'test-synthetic-bounded-oracle',
                                   'tracker': 'test-manual-lifecycle', 'revision': 'm6-stress-v1'},
                      'turns': turns, 'target': profile['target'], 'related': profile['related']})
    dataset = root / 'profiles.jsonl'
    dataset.write_text(''.join(json.dumps(c) + '\n' for c in cases), encoding='utf-8')
    save_json(root / 'fixture_provenance.json', {
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'generator': str(Path(__file__)), 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'synthetic lifecycle stress test, not human annotations or real contradictions',
        'vector': 'hidden_slots / 3 clipped to [-1,1]; identical across turns and paired runs',
        'assignment': 'source order modulo four; distractor offset floor(N/2); no outcome selection'})
    evaluation = json.loads(Path('configs/m2.json').read_text())
    evaluation.update(run_id='m6_stress_v1', input_mode='mock_tracker', profiles=dataset.as_posix())
    evaluation_path = root / 'evaluation.json'
    save_json(evaluation_path, evaluation)
    results = {}
    for stack in ('hybrid', 'structured_dense', 'three_way'):
        config = json.loads(Path(f'configs/ranking/m4_{stack}.json').read_text())
        for policy in ('raw', 'active_only'):
            selection = deepcopy(config)
            selection['evidence_filter'] = policy
            selection_path = root / f'{stack}_{policy}.json'
            save_json(selection_path, selection)
            run(evaluation_path, root / f'{stack}_{policy}', ranking_config=selection_path)
        left, right = (root / f'{stack}_{p}' for p in ('raw', 'active_only'))
        assert (left / 'inputs.json').read_bytes() == (right / 'inputs.json').read_bytes()
        summary, differences = paired_comparison(left, right)
        summary['strata'] = {}
        for stratum in strata:
            ids = {c['id'] for c in cases if c['stratum'] == stratum}
            rows = [d for d in differences if d['profile_id'] in ids]
            summary['strata'][stratum] = {
                'count': len(rows),
                'rank_improved': sum(d['treatment_target_rank'] < d['control_target_rank'] for d in rows),
                'rank_regressed': sum(d['treatment_target_rank'] > d['control_target_rank'] for d in rows),
                'metric_differences': {m: sum(d[m] for d in rows)/len(rows) for m in summary['metrics']}}
        save_json(root / f'{stack}_comparison.json', summary)
        save_json(root / f'{stack}_paired_differences.json', differences)
        results[stack] = summary
    save_json(root / 'comparisons.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
