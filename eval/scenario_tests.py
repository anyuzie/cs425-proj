"""Repeatable HTTP acceptance scenarios against an already running real app."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def expectation_errors(result, expected):
    errors = []
    profile = result['trace']['profile']
    for key in ('action', 'done'):
        if key in expected and result[key] != expected[key]:
            errors.append(f"{key}: expected {expected[key]}, got {result[key]}")
    if result['action'] == expected.get('not_action'):
        errors.append(f"Unexpected repeated {result['action']}")
    signs = [(expected['slot'], expected['sign'])] if 'slot' in expected else []
    signs += [(s, 1) for s in expected.get('positive', [])]
    signs += [(s, -1) for s in expected.get('negative', [])]
    for slot, sign in signs:
        value = profile[slot]['value']
        if value is None or value * sign <= 0:
            errors.append(f"{slot}: expected sign {sign}, got {value}")
    if expected.get('empty_profile') and any(p['value'] is not None for p in profile.values()):
        errors.append('Inferred preferences from an input expected to be neutral/background')
    if expected.get('no_recommendations') and result['recommendations']:
        errors.append('Returned recommendations despite no stated occupational preference')
    if 'recommendations' in expected and len(result['recommendations'] or []) != expected['recommendations']:
        errors.append('Wrong recommendation count')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or ROOT/'reports/scenarios'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)

    def request(path, data=None):
        req = urllib.request.Request(args.url.rstrip('/')+path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    health = request('/api/health')[1]
    if health['components']['extractor'] != 'deberta':
        raise RuntimeError('This acceptance run requires the real extractor')
    with (ROOT/'data/tasks.csv').open(encoding='utf-8-sig', newline='') as f:
        tasks = {(r['code'], r['Task']) for r in csv.DictReader(f)}
    cases = json.loads((ROOT/'tests/conversation_cases.json').read_text(encoding='utf-8'))
    results = []
    for case in cases:
        status, result = request('/api/session', {'debug': True})
        assert status == 200 and result['turn'] == 0
        turns, failures = [], []
        for index, text in enumerate(case['messages'], 1):
            start = perf_counter()
            status, result = request('/api/chat', {'session_id': result['session_id'], 'text': text})
            elapsed = perf_counter()-start
            if status != 200:
                failures.append(f'HTTP {status}: {result}')
                break
            if result['turn'] != index or not result['reply'] or not 1 <= result['turn'] <= 10:
                failures.append('Invalid turn/reply/budget')
            for slot in result['trace']['profile'].values():
                value = slot['value']
                if value is not None and (not math.isfinite(value) or not -1 <= value <= 1):
                    failures.append('Invalid profile value')
            recs = result['recommendations'] or []
            if recs and (len(recs) != 5 or len({r['code'] for r in recs}) != 5):
                failures.append('Recommendation count/uniqueness failure')
            for rec in recs:
                if not rec['tasks'] or any((rec['code'], task) not in tasks for task in rec['tasks']):
                    failures.append('Missing or incorrectly attributed task citation')
            quotes = result.get('quotes')
            if quotes and (not any(quotes['now'] in m for m in case['messages'][:index]) or
                           ('earlier' in quotes and not any(quotes['earlier'] in m for m in case['messages'][:index]))):
                failures.append('Quoted text was not an exact user message')
            turns.append({'text': text, 'seconds': elapsed, 'response': result})
            if result['done']:
                break
        quality = expectation_errors(result, case['expect']) if status == 200 else []
        row = dict(id=case['id'], invariants=failures, behavioral_failures=quality, turns=turns)
        results.append(row)
        (output/f"{case['id']}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding='utf-8')
        print(case['id'], 'PASS' if not failures and not quality else 'FAIL', quality or failures, flush=True)

    # API boundary checks use a separate session so invalid requests cannot alter scenarios.
    _, a = request('/api/session', {'debug': True})
    _, b = request('/api/session', {'debug': False})
    sid = a['session_id']
    checks = {}
    for name, body, expected in [
        ('unknown_session', {'session_id':'missing', 'text':'hello'},404),
        ('blank', {'session_id':sid,'text':'  '},422),
        ('too_long', {'session_id':sid,'text':'x'*2001},422),
        ('missing_text', {'session_id':sid},422),
        ('invalid_message_id', {'session_id':sid,'text':'hello','message_id':'bad id'},422)]:
        checks[name] = request('/api/chat', body)[0] == expected
    checks['unsafe_participant'] = request('/api/session', {'participant_code':'../bad'})[0] == 422
    body = {'session_id':sid,'text':'I enjoy painting.','message_id':'once'}
    _, first = request('/api/chat',body)
    _, repeated = request('/api/chat',body)
    checks['duplicate_request'] = first == repeated and repeated['turn'] == 1
    _, isolated = request('/api/chat',{'session_id':b['session_id'],'text':'I enjoy repairing machines.'})
    checks['session_isolation'] = isolated['turn'] == 1 and isolated['trace'] is None
    completed = next((t['response'] for r in results for t in r['turns'] if t['response']['done']), None)
    checks['finished_session'] = completed is not None and request('/api/chat',{'session_id':completed['session_id'],'text':'more'})[0] == 409
    _, boundary = request('/api/session', {'debug':False})
    checks['max_length_accepted'] = request('/api/chat', {'session_id':boundary['session_id'], 'text':'I enjoy painting '+ 'x'*1983})[0] == 200
    _, concurrent = request('/api/session', {'debug':False})
    duplicate = {'session_id':concurrent['session_id'], 'text':'I enjoy teaching.', 'message_id':'concurrent-once'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(lambda _: request('/api/chat',duplicate), range(2)))
    checks['concurrent_duplicate'] = replies[0] == replies[1] and replies[0][0] == 200 and replies[0][1]['turn'] == 1
    times = sorted(t['seconds'] for r in results for t in r['turns'])
    summary = dict(health=health, cases=len(results), turns=len(times),
        invariant_failures=sum(bool(r['invariants']) for r in results),
        behavioral_failures={r['id']:r['behavioral_failures'] for r in results if r['behavioral_failures']},
        api_checks=checks, seconds={'median':times[len(times)//2], 'max':max(times)},
        scope='Sequential real-model development acceptance; not held-out quality or load testing')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print('Saved',output)
    return bool(summary['invariant_failures'] or summary['behavioral_failures'] or not all(checks.values()))


if __name__ == '__main__':
    raise SystemExit(main())
