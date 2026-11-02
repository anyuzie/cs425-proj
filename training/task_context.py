"""Train and evaluate source-labelled BERT using clause text exclusively."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

LABELS=('task','context')
def read(path):
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:raise ValueError('Empty dataset')
    for row in rows:
        if not isinstance(row.get('text'),str) or not row['text'].strip():raise ValueError('Nonempty text required')
        if set(row.get('labels',{}))!=set(LABELS) or any(row['labels'][k] not in (0,1) for k in LABELS):raise ValueError('Two binary labels required')
    return rows
def metrics(rows,predictions):
    result={}
    for label in LABELS:
        tp=sum(bool(r['labels'][label] and p['route_'+label]) for r,p in zip(rows,predictions))
        fp=sum(bool(not r['labels'][label] and p['route_'+label]) for r,p in zip(rows,predictions))
        fn=sum(bool(r['labels'][label] and not p['route_'+label]) for r,p in zip(rows,predictions))
        tn=len(rows)-tp-fp-fn
        result[label]={'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),
            'f1':2*tp/max(1,2*tp+fp+fn),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'support':tp+fn}
    result['macro_f1']=sum(result[k]['f1'] for k in LABELS)/2
    result['exact_match']=sum(all(bool(r['labels'][k])==p['route_'+k] for k in LABELS) for r,p in zip(rows,predictions))/len(rows)
    return result

def evaluate(checkpoint,validation,train,task_threshold=.5,context_threshold=.5):
    try:from task_context import TaskContextClassifier
    except ImportError:from recommender.utterance.task_context import TaskContextClassifier
    model=TaskContextClassifier(checkpoint,task_threshold=task_threshold,context_threshold=context_threshold)
    # Length sorting reduces padding; restore original order in saved predictions.
    order=sorted(range(len(validation)),key=lambda i:len(validation[i]['text']))
    predicted=model.predict([validation[i]['text'] for i in order])
    predictions=[None]*len(validation)
    for i,prediction in zip(order,predicted):predictions[i]=prediction
    majority={label:sum(r['labels'][label] for r in train)>len(train)/2 for label in LABELS}
    baseline=[{'route_'+label:majority[label] for label in LABELS} for row in validation]
    def distribution(values):
        values=sorted(values)
        return {'count':len(values),'minimum':values[0],'median':values[len(values)//2],
                'maximum':values[-1],'mean':sum(values)/len(values)} if values else {'count':0}
    report={'inputs':['text'],'validation_examples':len(validation),
        'thresholds':{'task':task_threshold,'context':context_threshold},
        'metrics':metrics(validation,predictions),'majority_baseline':{'prediction':majority,**metrics(validation,baseline)},
        'score_distributions':{label:{str(target):distribution([p[label+'_score'] for r,p in zip(validation,predictions)
            if r['labels'][label]==target]) for target in (0,1)} for label in LABELS},
        'routing_counts':dict(Counter('task_and_context' if p['route_task'] and p['route_context'] else
            'task' if p['route_task'] else 'context' if p['route_context'] else 'no_channel' for p in predictions)),
        'limits':['Source-based labels, not independently reviewed clause labels',
            'Only 11 held-out context families; 22 context texts',
            'Only [1,0] and [0,1] supervision; simultaneous routing and rejection need separate conversational evaluation',
            'Sigmoid outputs are not calibrated probabilities; no independent test set']}
    with (checkpoint/'validation_predictions.jsonl').open('w',encoding='utf-8') as handle:
        for row,prediction in zip(validation,predictions):
            correct=all(bool(row['labels'][label])==prediction['route_'+label] for label in LABELS)
            handle.write(json.dumps(dict(row,**prediction,exact_match=correct),ensure_ascii=False)+'\n')
    (checkpoint/'validation_errors.jsonl').write_text(''.join(json.dumps(dict(row,**p),ensure_ascii=False)+'\n'
        for row,p in zip(validation,predictions) if not all(bool(row['labels'][label])==p['route_'+label] for label in LABELS)),encoding='utf-8')
    (checkpoint/'validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True);parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--base-model',default='google-bert/bert-base-uncased')
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--max-length',type=int,default=128);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--task-threshold',type=float,default=.5);parser.add_argument('--context-threshold',type=float,default=.5)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite checkpoint directory')
    if args.epochs<1 or args.batch_size<1 or not 4<=args.max_length<=512:raise ValueError('Invalid training arguments')
    if not 0<args.task_threshold<1 or not 0<args.context_threshold<1:raise ValueError('Thresholds must lie between 0 and 1')
    train=read(args.train);validation=read(args.validation)
    normalize=lambda text:' '.join(''.join(c if c.isalnum() else ' ' for c in text.casefold()).split())
    if {normalize(r['text']) for r in train}&{normalize(r['text']) for r in validation}:raise ValueError('Text leakage across splits')
    for group in ('occupation_code','context_id'):
        if {r[group] for r in train if group in r}&{r[group] for r in validation if group in r}:raise ValueError('Source-group leakage')
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification,set_seed
    set_seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.base_model)
    config=AutoConfig.from_pretrained(args.base_model)
    if config.model_type!='bert':raise ValueError('This baseline requires BERT')
    config.num_labels=2;config.id2label=dict(enumerate(LABELS));config.label2id={k:i for i,k in enumerate(LABELS)}
    config.problem_type='multi_label_classification'
    model=AutoModelForSequenceClassification.from_pretrained(args.base_model,config=config)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    encoded=tokenizer([r['text'] for r in train],truncation=True,max_length=args.max_length)
    lengths=[len(value) for value in encoded['input_ids']]
    targets=torch.tensor([[r['labels'][k] for k in LABELS] for r in train],dtype=torch.float32)
    positive=targets.sum(0);negative=len(train)-positive
    if bool((positive==0).any() or (negative==0).any()):raise ValueError('Both targets need positive and negative training examples')
    weights=(negative/positive).to(device)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    print('Training examples: '+str(len(train))+'; positive weights: '+str(weights.cpu().tolist()),flush=True)
    clock=time.monotonic();losses=[]
    for epoch in range(args.epochs):
        # Every training example is visited once. Shuffle within similar-length
        # buckets and shuffle batch order to reduce padding without source sorting.
        indices=sorted(range(len(train)),key=lambda i:lengths[i])
        batches=[]
        for start in range(0,len(indices),args.batch_size*20):
            bucket=indices[start:start+args.batch_size*20];random.shuffle(bucket)
            batches.extend(bucket[i:i+args.batch_size] for i in range(0,len(bucket),args.batch_size))
        random.shuffle(batches);model.train();total=0
        for step,indices in enumerate(batches,1):
            batch=tokenizer.pad([{k:values[i] for k,values in encoded.items()} for i in indices],padding=True,return_tensors='pt').to(device)
            optimizer.zero_grad();logits=model(**batch).logits
            loss=criterion(logits,targets[indices].to(device));loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();total+=loss.item()*len(indices)
            if step==1 or step%50==0:print(f'Epoch {epoch+1}/{args.epochs}: batch {step}/{len(batches)}; elapsed {time.monotonic()-clock:.0f}s',flush=True)
        losses.append(total/len(train));print(f'Epoch {epoch+1} mean weighted BCE: {losses[-1]:.5f}',flush=True)
    model.config.task_context_classifier={'trained':True,'max_length':args.max_length,
        'thresholds':[args.task_threshold,args.context_threshold],'source_labels':True,'inputs':['text']}
    args.output.mkdir(parents=True);model.save_pretrained(args.output);tokenizer.save_pretrained(args.output)
    manifest={'base_model':args.base_model,'seed':args.seed,'epochs':args.epochs,'batch_size':args.batch_size,
        'max_length':args.max_length,'learning_rate':2e-5,'loss':'BCEWithLogitsLoss',
        'pos_weight_training_only':weights.cpu().tolist(),'training_loss_by_epoch':losses,
        'input_fields':['text'],'device':str(device),'torch_version':torch.__version__,
        'data_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('train',args.train),('validation',args.validation)]},
        'training_seconds':time.monotonic()-clock}
    (args.output/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    del optimizer,model
    evaluate(args.output,validation,train,args.task_threshold,args.context_threshold)

if __name__=='__main__':main()
