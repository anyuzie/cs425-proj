"""Source-labelled task/context splits; metadata never enters model text."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent if (ROOT.parent/'data/tasks.csv').exists() else Path(r'C:\Users\aanny\Documents\GitHub\year5\cs425\project')
def normal(text):return re.sub(r'\W+',' ',text.casefold()).strip()
def write(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=PROJECT/'data/task_context_source_v1')
    args=parser.parse_args()
    output=args.output
    if output.exists():raise ValueError('Dataset directory already exists; refusing overwrite')
    task_path=PROJECT/'data/tasks.csv'
    context_path=PROJECT/'data/work_context/onet_31_0/compact/context_dimensions.csv'
    with task_path.open(encoding='utf-8-sig',newline='') as handle:tasks=list(csv.DictReader(handle))
    with context_path.open(encoding='utf-8-sig',newline='') as handle:contexts=list(csv.DictReader(handle))
    rng=random.Random(42)
    occupations=sorted({r['code'] for r in tasks});rng.shuffle(occupations)
    validation_codes=set(occupations[:round(.2*len(occupations))])
    families=sorted({r['context_id'] for r in contexts});rng.shuffle(families)
    validation_families=set(families[:round(.2*len(families))])
    proposed=[]
    for row in tasks:
        proposed.append({'example_id':'task:'+row['code']+':'+row['Task ID'],'text':row['Task'].strip(),
            'labels':{'task':1,'context':0},'source':'tasks.csv','occupation_code':row['code'],
            'source_id':row['Task ID'],'split':'validation' if row['code'] in validation_codes else 'train',
            'label_provenance':'source_label','training_eligible':True})
    for row in contexts:
        for field in ('context','description'):
            proposed.append({'example_id':'context:'+row['context_id']+':'+field,'text':row[field].strip(),
                'labels':{'task':0,'context':1},'source':'context_dimensions.csv','context_id':row['context_id'],
                'source_field':field,'split':'validation' if row['context_id'] in validation_families else 'train',
                'label_provenance':'source_label','training_eligible':True})
    grouped={}
    for row in proposed:
        if row['text']:grouped.setdefault(normal(row['text']),[]).append(row)
    conflicts=[];duplicates=[];overlaps=[];retained=[]
    for key,rows in grouped.items():
        if len({tuple(r['labels'].values()) for r in rows})>1:
            conflicts.extend(rows);continue
        if len({r['split'] for r in rows})>1:
            # Exclude all copies so neither occupation nor family grouping is violated.
            overlaps.extend(rows);continue
        retained.append(rows[0]);duplicates.extend(rows[1:])
    train=sorted([r for r in retained if r['split']=='train'],key=lambda r:r['example_id'])
    validation=sorted([r for r in retained if r['split']=='validation'],key=lambda r:r['example_id'])
    assert not {normal(r['text']) for r in train}&{normal(r['text']) for r in validation}
    assert not {r.get('occupation_code') for r in train if 'occupation_code'in r}&{r.get('occupation_code') for r in validation if 'occupation_code'in r}
    assert not {r.get('context_id') for r in train if 'context_id'in r}&{r.get('context_id') for r in validation if 'context_id'in r}
    assert len(families)==57
    output.mkdir(parents=True)
    write(output/'train.jsonl',train);write(output/'validation.jsonl',validation)
    write(output/'excluded_conflicts.jsonl',conflicts);write(output/'excluded_duplicates.jsonl',duplicates)
    write(output/'excluded_cross_split_duplicates.jsonl',overlaps)
    manifest={'seed':42,'labels':['task','context'],'input_fields':['text'],
        'source_labelling':'Task source -> [1,0]; context names/descriptions -> [0,1]',
        'train':{'examples':len(train),'sources':dict(Counter(r['source'] for r in train))},
        'validation':{'examples':len(validation),'sources':dict(Counter(r['source'] for r in validation))},
        'context_families':{'train':len(families)-len(validation_families),'validation':len(validation_families)},
        'task_occupation_groups':{'train':len(occupations)-len(validation_codes),'validation':len(validation_codes)},
        'excluded':{'conflicting_labels':len(conflicts),'duplicates':len(duplicates),'cross_split_duplicate_rows':len(overlaps)},
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (task_path,context_path)},
        'dataset_sha256':{name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in ('train.jsonl','validation.jsonl')},
        'limits':['Source labels are provisional, not manually reviewed clause labels',
                  'No training targets with simultaneous positives or simultaneous negatives',
                  'No conversational generalisation claim']}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
