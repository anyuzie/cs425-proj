"""Real-model repair regression; explicit annotations, no simulator-generated answers."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import types
import zipfile
import torch

from pipeline.extractor import DebertaExtractor

ROOT=Path(__file__).resolve().parents[1]


def evaluate(extractor):
    cases=json.loads((ROOT/'data/eval/dialogue_repairs.json').read_text())
    rows=[]
    for case in cases:
        reading=extractor.read(case['text'],case.get('context'))
        got={r.slot:r.sign for r in reading.slots}
        reasons={r.slot:r.reason for r in reading.slots}
        errors=[]
        if 'expected' in case and got!=case['expected']:
            errors.append(f"expected exactly {case['expected']}, got {got}")
        for slot,sign in case.get('required',{}).items():
            if got.get(slot)!=sign:errors.append(f'{slot}: expected {sign}, got {got.get(slot)}')
        for slot,reason in case.get('reasons',{}).items():
            if reasons.get(slot)!=reason:errors.append(f'{slot}: expected reason {reason}, got {reasons.get(slot)}')
        rows.append(dict(**case,reading=asdict(reading),passed=not errors,errors=errors))
        print(case['id'], 'PASS' if not errors else errors,flush=True)
    return dict(total=len(rows),passed=sum(r['passed'] for r in rows),cases=rows,
                caveat='AI-authored development regression annotations, not a blind human benchmark. Required slots allow extra predictions; expected maps require an exact match.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-source',type=Path,help='Preserved source ZIP for a paired adapter comparison with identical weights')
    args=parser.parse_args()
    torch.set_num_threads(4)
    extractor=DebertaExtractor(ROOT/'models/extractor',ROOT/'data/labels.json')
    result=evaluate(extractor)
    if args.baseline_source:
        module=types.ModuleType('pipeline.extractor_before')
        module.__package__='pipeline'
        with zipfile.ZipFile(args.baseline_source) as archive:
            source=archive.read('pipeline/extractor.py').decode('utf-8-sig')
        exec(compile(source,'preserved/pipeline/extractor.py','exec'),module.__dict__)
        extractor.read=types.MethodType(module.DebertaExtractor.read,extractor)
        result={'after':result,'before':evaluate(extractor),
                'scope':'Identical trained extractor weights; compare preserved and revised read adapters.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Saved',args.output)


if __name__=='__main__':main()
