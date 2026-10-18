"""Check saved M6 paired inputs, structured invariance and fallback gates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender.ranker import RankingPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    cases = {c['id']: c for c in map(json.loads, (root/'profiles.jsonl').read_text().splitlines())}
    m1 = RankingPipeline(config=Path('configs/ranking/baseline_v1.json'))
    checked = {}
    for stack in ('hybrid', 'structured_dense', 'three_way'):
        left, right = (root/f'{stack}_{p}' for p in ('raw', 'active_only'))
        assert (left/'inputs.json').read_bytes() == (right/'inputs.json').read_bytes()
        a, b = (read(p/'turn_predictions.json') for p in (left, right))
        assert len(a) == len(b) == 2*len(cases)
        for raw, active in zip(a, b):
            assert (raw['profile_id'], raw['turn']) == (active['profile_id'], active['turn'])
            structured = lambda row: {o['code']: o['signals']['structured'] for o in row['occupations']}
            case = cases[raw['profile_id']]
            turn = case['turns'][raw['turn']-1]
            baseline = m1.rank(turn['interests'], set(turn['observed_types']), [], top_k=m1.eligible_count)
            expected_signals = {o.code: asdict(o)['signals']['structured'] for o in baseline.occupations}
            for row in (raw, active):
                assert all(signal == expected_signals[code] for code, signal in structured(row).items())
            if raw['turn'] == 1 or case['stratum'] == 'all_active':
                assert raw['occupations'] == active['occupations']
            if raw['turn'] == 2 and case['stratum'] == 'empty_active':
                turn = case['turns'][-1]
                expected = m1.rank(turn['interests'], set(turn['observed_types']), [], top_k=100)
                assert [o.code for o in expected.occupations] == [o['code'] for o in active['occupations']]
                assert active['query_diagnostics']['query_sentences'] == []
        checked[stack] = {'turn_pairs': len(a), 'identical_inputs': True,
                          'structured_invariant': True, 'all_active_identical': True,
                          'empty_active_m1_fallback': True}
    (root/'validation.json').write_text(json.dumps(checked, indent=2)+'\n')
    print(json.dumps(checked, indent=2))


if __name__ == '__main__':
    main()
