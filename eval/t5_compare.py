"""Paired wording comparison on frozen real-pipeline turns and new synthetic test rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

from eval.t5_wording import AuditedT5, AuditedT5V2, profiles, summarize
from pipeline.t5 import T5Filler
from pipeline.t5_v2 import KNOWN

ROOT=Path(__file__).resolve().parents[1]


def legacy_prompt(fields):
    selected={k:fields[k] for k in ('action','type','reference','context','retry','previous_question')}
    selected['style']={'direct':'plain','scenario':'direct','workday':'reflective'}[fields['style']]
    return 'career question: '+json.dumps(selected,ensure_ascii=False)


def heldout(filler,rows,version):
    results=[]
    for row in rows:
        fields=json.loads(row['input'].split(': ',1)[1])
        text=row['input'] if version==2 else legacy_prompt(fields)
        started=perf_counter()
        raw=T5Filler.generate(filler,text)
        valid=filler.valid(raw,fields['action'],fields['previous_question'])
        results.append(dict(id=row['id'],action=row['action'],slot=row['slot'],input=text,
                            focus=fields['focus'],target=row['target'],raw=raw,surface_valid=valid,
                            exact_target=raw==row['target'],seconds=perf_counter()-started,
                            repeated_previous=bool(raw) and ' '.join(raw.lower().split()) in ' '.join(fields['previous_question'].lower().split())))
    return results


def test_summary(rows):
    return dict(total=len(rows),surface_valid=sum(r['surface_valid'] for r in rows),
                exact_target=sum(r['exact_target'] for r in rows),repeated_previous=sum(r['repeated_previous'] for r in rows),
                unique_raw=len({r['raw'] for r in rows if r['raw']}),
                actions={a:dict(total=sum(r['action']==a for r in rows),
                                surface_valid=sum(r['surface_valid'] for r in rows if r['action']==a)) for a in sorted({r['action'] for r in rows})},
                caveat='Synthetic unseen background groups; targets and linguistic patterns are template-expanded. Surface validity and exact matching are not semantic accuracy.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    bank=json.loads((ROOT/'data/question_bank.json').read_text())
    names=json.loads((ROOT/'data/labels.json').read_text())['type_names']
    checkpoint1=ROOT/'models/t5-questions-v1'
    old=AuditedT5(bank,names,checkpoint1);old.records=[]
    new=AuditedT5V2(bank,names,args.v2);new.records=[]
    old.torch.set_num_threads(4)
    comparisons=[]
    for persona in profiles():
        frozen=json.loads((ROOT/'reports/t5/current_20261021'/f"{persona['id']}.json").read_text())
        for turn in frozen['turns']:
            action=turn['wording']['action'];payload=dict(turn['wording']['payload'])
            old.fill(action,payload)
            known=next((s for s,v in sorted(turn['profile_after'].items(),key=lambda p:-(p[1]['value'] or 0))
                        if v['value'] is not None and v['value']>0 and v['reason']=='enjoyment' and s!=payload.get('slot')),None)
            payload.update(last_user=turn['user'][:240],known_interest=KNOWN[known] if known else '')
            new.fill(action,payload)
            comparisons.append(dict(persona=persona['id'],description=persona['description'],turn=turn['turn'],
                                    user=turn['user'],profile=turn['profile_after'],v1=dict(old.records[-1]),v2=dict(new.records[-1])))
        print('Replayed',persona['id'],flush=True)
    (args.output/'frozen_conversations.json').write_text(json.dumps(comparisons,indent=2,ensure_ascii=False),encoding='utf-8')
    styles=[]
    for slot in 'RIASEC':
        for index in range(3):
            payload=dict(slot=slot,index=index,clarify_index=1,last_user='I have described my other work interests.',
                         known_interest='creative work' if slot=='I' else 'research')
            old.fill('CLARIFY',payload);new.fill('CLARIFY',payload)
            styles.append(dict(slot=slot,style_index=index,v1=dict(old.records[-1]),v2=dict(new.records[-1])))
    (args.output/'styles.json').write_text(json.dumps(styles,indent=2,ensure_ascii=False),encoding='utf-8')
    source=ROOT.parent/'data/t5_questions_v2/test.jsonl'
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    tests={}
    for version,filler in ((1,old),(2,new)):
        generations=heldout(filler,rows,version)
        (args.output/f'heldout_v{version}.json').write_text(json.dumps(generations,indent=2,ensure_ascii=False),encoding='utf-8')
        tests[f'v{version}']=test_summary(generations)
        print('Heldout',version,json.dumps(tests[f'v{version}']),flush=True)
    result=dict(frozen={version:summarize([r[version] for r in comparisons]) for version in ('v1','v2')},
                fixed_focus_style_diversity={v:{slot:len({r[v]['raw'] for r in styles if r['slot']==slot and r[v]['source']=='t5'})
                                                for slot in 'RIASEC'} for v in ('v1','v2')},
                heldout=tests,test_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                checkpoint_sha256={f'v{v}':hashlib.sha256((p/'model.safetensors').read_bytes()).hexdigest()
                                   for v,p in ((1,checkpoint1),(2,args.v2))},
                scope='Compare each trained checkpoint with its matching production prompt. This is a model-plus-prompt comparison, not an isolated weight-only ablation.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
