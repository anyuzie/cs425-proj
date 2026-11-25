"""Evaluate a T5 checkpoint: joint checks on raw questions, reference-token accuracy, and (optionally)
the app suites (112 controlled dialogue cases, 23 catalogue tasks with a retry each).

Run from the repository root:
  python -m training.t5.evaluate --model <checkpoint> --benchmark data/t5/benchmarks/development_432.json \n      --decode data/t5/decode.json --output <new dir> --application-suites
"""
import argparse
import collections
import copy
import json
import time
from pathlib import Path
from training.t5.common import ROOT,read,write,sha,checks,load_filler


def evaluate(model,adapter,benchmark,output,device,threads,decode,application_suites=False,seed=42):
    f=load_filler(model,adapter,device,threads,decode);started=time.perf_counter()
    f.torch.manual_seed(seed)
    if f.torch.cuda.is_available():f.torch.cuda.manual_seed_all(seed)
    rows=read(benchmark)
    if not rows:raise ValueError('Benchmark is empty')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate case IDs')
    outputs=[];batch_seconds=[]
    for start in range(0,len(rows),8):
        batch=rows[start:start+8]
        encoded=f.tokenizer([r['input'] for r in batch],padding=True,return_tensors='pt',truncation=False)
        if encoded.input_ids.shape[1]>f.decoding['max_input_tokens']:
            # Keep the application limit: overlong inputs fall back rather than truncate.
            raw_texts=[]
            for row in batch:
                f.pending={};raw_texts.append(f.generate(row['input']))
        else:
            options={k:v for k,v in f.decoding.items() if k!='max_input_tokens'}
            if f.device=='cuda':f.torch.cuda.synchronize()
            generation_start=time.perf_counter()
            with f.torch.inference_mode():generated=f.model.generate(**encoded.to(f.device),**options)
            if f.device=='cuda':f.torch.cuda.synchronize()
            batch_seconds.append(time.perf_counter()-generation_start)
            raw_texts=[t.strip() for t in f.tokenizer.batch_decode(generated,skip_special_tokens=True)]
        for row,raw in zip(batch,raw_texts):
            fields=row.get('fields') or json.loads(row['input'].split(': ',1)[1])
            record=dict(id=row['id'],input=row['input'],target=row['target'],raw=raw,
                fields=fields,scores=checks(fields,raw),retry=fields['retry'],action=fields['action'])
            if 'payload' in row:
                payload=copy.deepcopy(row['payload'])
                if f.fields(row['action'],payload)!=fields:
                    raise ValueError('Prompt contract changed: revise/version the benchmark and rerun the baseline.')
                original=f.generate
                def replay(text,raw=raw,row=row):
                    if text!=row['input']:raise ValueError('Renderer prompt does not match frozen input')
                    f.pending.update(input=text,raw=raw);return raw
                f.generate=replay
                try:final=f.fill(row['action'],payload);audit=f.records[-1]
                finally:f.generate=original
                record.update(final=final,source=audit['source'])
            outputs.append(record)
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq
    encoded=[]
    for row in rows:
        item=f.tokenizer(row['input'],truncation=False)
        if len(item['input_ids'])>f.decoding['max_input_tokens']:continue
        item['labels']=f.tokenizer(text_target=row['target'],truncation=False)['input_ids'];encoded.append(item)
    loss=0.;correct=tokens=0
    with f.torch.inference_mode():
        for batch in DataLoader(encoded,batch_size=8,collate_fn=DataCollatorForSeq2Seq(f.tokenizer,model=f.model)):
            batch={k:v.to(f.device) for k,v in batch.items()};result=f.model(**batch)
            mask=batch['labels']!=-100;n=mask.sum().item()
            correct+=((result.logits.argmax(-1)==batch['labels'])&mask).sum().item();tokens+=n;loss+=result.loss.item()*n
    stats=dict(cases=len(rows),joint_pass=sum(r['scores']['joint'] for r in outputs),
        joint_rate=sum(r['scores']['joint'] for r in outputs)/len(rows),
        checks={key:sum(r['scores'][key] for r in outputs) for key in ('format','focus','novel','intent')},
        by_action={kind:dict(total=sum(r['action']==kind for r in outputs),passed=sum(r['action']==kind and r['scores']['joint'] for r in outputs)) for kind in ('CLARIFY','PROBE','CHALLENGE')},
        retries=dict(total=sum(r['retry'] for r in outputs),passed=sum(r['retry'] and r['scores']['joint'] for r in outputs)),
        sources=dict(collections.Counter(r.get('source','not_rendered') for r in outputs)),
        reference_tokens=dict(correct=correct,total=tokens,accuracy=correct/tokens if tokens else None,cross_entropy=loss/tokens if tokens else None),
        benchmark_sha256=sha(benchmark),device=f.device,decode=f.decoding,seed=seed,
        timing=dict(batched_generation_seconds=sum(batch_seconds),batch_seconds=batch_seconds,batch_size=8,
                    note='Batched throughput timing; not single-user reply latency. Overlimit per-row fallbacks excluded.'),
        hardware=dict(cuda_version=f.torch.version.cuda,cuda_device=f.torch.cuda.get_device_name(0) if f.device=='cuda' else None,
                      torch_threads=f.torch.get_num_threads()))
    if application_suites:
        from training.t5.controlled import controlled
        from pipeline.task_questions import load_tasks
        generic=controlled(f);tasks=[]
        for i,task in enumerate(load_tasks(ROOT/'data/tasks.csv',ROOT/'configs/question_tasks.json')):
            p=dict(slot=task['slot'],task=dict(task),index=i%3,clarify_index=i%3)
            f.fill('CLARIFY',p);first=dict(f.records[-1]);tasks.append(first)
            f.fill('ABSTAIN',dict(p,retry='CLARIFY',previous_question=first['final']));tasks.append(dict(f.records[-1]))
        def counts(records):
            return dict(total=len(records),sources=dict(collections.Counter(r['source'] for r in records)),
                retry_total=sum(r['action']=='ABSTAIN' for r in records),
                retry_accepted=sum(r['action']=='ABSTAIN' and r['source']=='t5' for r in records))
        stats['application_suites']=dict(generic=counts(generic),tasks=counts(tasks))
        write(Path(output)/'application_outputs.json',dict(generic=generic,tasks=tasks))
    stats['seconds']=time.perf_counter()-started
    write(Path(output)/'outputs.json',outputs);write(Path(output)/'metrics.json',stats)
    print(json.dumps(stats),flush=True)
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--adapter')
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--decode',type=Path,required=True);p.add_argument('--application-suites',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.output.exists():p.error('Evaluation output exists; use a new directory')
    evaluate(a.model,a.adapter,a.benchmark,a.output,a.device,a.threads,read(a.decode),a.application_suites,a.seed)
