"""Reproducible occupation retrieval preparation, contrastive training, indexing and evaluation.

Run each stage explicitly: python -m training.occupation_semantic <stage> --config <json>.
Formal source ownership is a retrieval proxy, never a career preference gold label.
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time

from recommender.occupation_semantic import (ROOT, EXPERIMENT, PRETRAINED, MODEL_ID, REVISION,
    BUILDER, digest, fingerprint, load_encoder)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normal(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def provenance(config):
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], text=True).splitlines()
    sources = {n: digest(ROOT/n) for n in names if Path(n).suffix in {'.py', '.json', '.yaml', '.txt'}
               and not n.startswith(('reports/', 'data/')) and (ROOT/n).is_file()}
    return {'config': config, 'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'git_status': subprocess.check_output(['git','status','--short'],text=True), 'code_hashes': sources,
            'dirty_code_fingerprint': hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
            'packages': {p:version(p) for p in ['torch','sentence-transformers','transformers','numpy','scikit-learn','huggingface-hub']},
            'model_id': MODEL_ID, 'model_revision': REVISION, 'pretrained_sha256': fingerprint(PRETRAINED)}


def grouped_components(texts, threshold=.85):
    """Transitive char-TFIDF near-duplicate groups, independent of labels/splits."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), dtype='float32')
    matrix = vec.fit_transform([normal(t) for t in texts])
    near = NearestNeighbors(metric='cosine', radius=1-threshold, n_jobs=1).fit(matrix)
    parents = list(range(len(texts)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for start in range(0,len(texts),128):
        neighbors = near.radius_neighbors(matrix[start:start+128], return_distance=False)
        for offset, ids in enumerate(neighbors):
            for j in ids:
                a,b=root(start+offset),root(int(j))
                parents[max(a,b)] = min(a,b)
    return [root(i) for i in range(len(texts))], vec, matrix


def prepare(cfg):
    out = ROOT/cfg['artifacts']
    if (out/'data.json').exists():
        raise ValueError('Preparation already exists; use a new version/output instead of overwriting')
    from recommender.data_loader import load_eligible_occupations
    pool = {o.code for o in load_eligible_occupations(ROOT/'data/occupations.csv')}
    paths = ['data/occupations.csv','data/tasks.csv', 'data/work_context/onet_31_0/prepared/occupation_work_context.jsonl',
             'data/work_context/onet_31_0/source_manifest.json']
    occupations = {r['code']:r for r in csv.DictReader((ROOT/paths[0]).open(encoding='utf-8-sig',newline='')) if r['code'] in pool}
    tasks = list(csv.DictReader((ROOT/paths[1]).open(encoding='utf-8-sig',newline='')))
    # Context text states the observed modal response and its percentage. Numeric means are never verbalized.
    context = [json.loads(l) for l in (ROOT/paths[2]).open(encoding='utf-8')]
    units = []
    for r in tasks:
        if r['code'] in pool and r['Task'].strip():
            units.append({'code':r['code'],'text':r['Task'].strip(),'kind':'task','source_id':r['Task ID']})
    suppressed = 0
    for r in context:
        if r['occupation_code'] not in pool: continue
        for c in r['context']:
            choices = [x for x in c['ratings'] if x['Category'] and x['Recommend Suppress'] != 'Y'
                       and x.get('category_description') and x.get('Data Value') not in ('',None)
                       and x.get('Not Relevant') != 'Y']
            suppressed += sum(x['Recommend Suppress']=='Y' for x in c['ratings'])
            if not choices: continue
            best = max(choices,key=lambda x:(float(x['Data Value']),x['Category']))
            units.append({'code':r['occupation_code'], 'text':c['element_name']+': '+best['category_description']+'.',
                          'kind':'context','source_id':c['element_id']+':'+best['Category'],
                          'reported_percentage':float(best['Data Value'])})
    unique = sorted({u['text'] for u in units})
    print('Grouping',len(unique),'unique source statements',flush=True)
    groups, vectorizer, matrix = grouped_components(unique, cfg['near_duplicate_cosine'])
    text_group = dict(zip(unique,groups))
    grouped = defaultdict(list)
    for u in units:
        u['group'] = str(text_group[u['text']]); grouped[u['group']].append(u)
    rng = random.Random(cfg['seed'])
    totals = Counter(u['code'] for g in grouped.values() for u in g)
    counts = {s:Counter() for s in ['train','validation','test']}
    keys = sorted(grouped); rng.shuffle(keys)
    examples=[]
    # Greedy occupation-stratified group assignment; shared/near duplicates stay indivisible.
    for key in keys:
        rows = grouped[key]; owners=sorted({u['code'] for u in rows})
        split = min(counts, key=lambda s:sum(counts[s][c]/(totals[c]*cfg['split'][s]) for c in owners))
        for u in rows: counts[split][u['code']] += 1
        examples.append({'group':key,'text':rows[0]['text'],'texts':sorted({u['text'] for u in rows}),
                         'positives':owners,'kind':rows[0]['kind'],'split':split})
    # Diagnostic occupations have no fine-tuning positives. Their task groups stay together.
    heldout = set(rng.sample(sorted(pool),cfg['heldout_occupations']))
    for e in examples:
        if heldout.intersection(e['positives']): e['split']='occupation_heldout'
    # Audit only development openings, never labels; exclude overlapping groups from training.
    dev = [json.loads(l)['text'] for l in (ROOT/'data/dev_profiles.jsonl').open(encoding='utf-8')]
    dev_norm = [normal(t) for t in dev]
    overlap=[]
    for e in examples:
        if any(normal(t) in d or d in normal(t) for t in e['texts'] for d in dev_norm):
            overlap.append(e['group']); e['split']='external_overlap_excluded'
    exclusions = {e['group'] for e in examples if e['split'] != 'train'}
    excluded_texts = [t for e in examples if e['group'] in exclusions for t in e['texts']]
    # Remove description sentences matching held-out sources, including near copies/substrings.
    from sklearn.neighbors import NearestNeighbors
    # Remove description near-copies of training queries too, so dynamic positive documents cannot copy them.
    excluded_matrix = matrix
    nearest = NearestNeighbors(metric='cosine',n_neighbors=1).fit(excluded_matrix)
    docs={}; description_exclusions=0
    for code,r in sorted(occupations.items()):
        description=[]
        for sentence in re.split(r'(?<=[.!?])\s+',r['Description'].strip()):
            n=normal(sentence)
            if not n: continue
            dist = nearest.kneighbors(vectorizer.transform([n]),return_distance=True)[0][0,0]
            if dist <= 1-cfg['near_duplicate_cosine'] or any(normal(t) in n or n in normal(t) for t in excluded_texts):
                description_exclusions+=1;continue
            description.append(sentence)
        docs[code]={'title':r['Title'], 'description':description,
                    'units':[u for u in units if u['code']==code and u['group'] not in exclusions]}
    data={'examples':sorted(examples,key=lambda e:int(e['group'])),'documents':docs,'heldout_codes':sorted(heldout),
          'sources':{p:digest(ROOT/p) for p in paths},'exclusions':sorted(exclusions)}
    write(out/'data.json',data)
    # Explicit source units preserve ratings metadata without putting it into query labels.
    write(out/'source_units.json',units)
    audit={'provenance':provenance(cfg),'source_counts':{'tasks':len(tasks),'eligible_source_units':len(units),'unique_texts':len(unique)},
           'source_schema':{'task':list(tasks[0]),'context':list(context[0])},
           'splits':dict(Counter(e['split'] for e in examples)),
           'units_by_kind':dict(Counter(u['kind'] for u in units)),
           'pool':sorted(pool),'candidate_count':len(pool),'context_missing_codes':sorted(pool-{r['occupation_code'] for r in context if r['has_context_data']}),
           'suppressed_rating_rows_not_verbalized':suppressed,'description_sentences_excluded':description_exclusions,
           'development_text_overlap_groups':overlap,'dev_openings_sha256':digest(ROOT/'data/dev_profiles.jsonl'),
           'dev_usage':'text overlap audit only; no profile labels/openings used as training examples',
           'label_definition':'multi-owner task/observed-modal-context association; incomplete semantic relevance, binary proxy',
           'grouping':'transitive char_wb TFIDF 3-5 cosine >= threshold; all group derivatives indivisible',
           'index_policy':'exclude all validation/test/occupation-heldout sources and near copies globally; remove query group dynamically for training',
           'limitations':['No human conversational relevance labels','No guarantee undetected semantic paraphrases are absent',
                         'Context categories are modal reported responses, not universal job properties or preferences',
                         'Primary split approximately occupation-stratified; heldout diagnostic removal changes proportions']}
    write(out/'split_manifest.json',audit)
    print(json.dumps(audit['splits']),flush=True)


def document(model, doc, exclude_groups=(), exclude_texts=()):
    """Bound each field separately to retain title/description/tasks/context within 256 tokens."""
    def safe(text):
        n=normal(text)
        return not any(normal(t) in n or n in normal(t) for t in exclude_texts)
    fields=[doc['title'], ' '.join(t for t in doc['description'] if safe(t))]
    for kind in ['task','context']:
        fields.append(' '.join(u['text'] for u in doc['units'] if u['kind']==kind and u['group'] not in exclude_groups and safe(u['text'])))
    budgets=[24,64,112,48]
    parts=[];coverage={}
    for name,text,budget in zip(['title','description','tasks','context'],fields,budgets):
        ids=model.tokenizer.encode(text,add_special_tokens=False)
        parts.append(model.tokenizer.decode(ids[:budget],skip_special_tokens=True))
        coverage[name]={'source_tokens':len(ids),'retained_tokens':min(len(ids),budget),'truncated':len(ids)>budget}
    return '\n'.join(p for p in parts if p),coverage


def encode(model,texts,batch_size):
    model.eval()
    return model.encode(texts,batch_size=batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)


def metrics(order, positives):
    ranks=[i+1 for i,c in enumerate(order) if c in positives]
    result={'mrr':1/min(ranks) if ranks else 0.}
    for k in [1,5,10]: result[f'hit@{k}']=float(any(r<=k for r in ranks))
    for k in [5,10,20,50]: result[f'recall@{k}']=sum(r<=k for r in ranks)/len(positives)
    for k in [5,10]:
        result[f'ndcg@{k}']=sum(1/math.log2(r+1) for r in ranks if r<=k)/sum(1/math.log2(r+1) for r in range(1,min(k,len(positives))+1))
    return result


def evaluate_model(model,data,examples,cfg):
    import numpy as np
    codes=sorted(data['documents'])
    texts=[document(model,data['documents'][c])[0] for c in codes]
    vectors=encode(model,texts,cfg['batch_size'])
    queries=encode(model,[e['text'] for e in examples],cfg['batch_size'])
    rows=[]
    for e,scores in zip(examples,queries@vectors.T):
        order=sorted(range(len(codes)),key=lambda i:(-float(scores[i]),codes[i]))
        ordered=[codes[i] for i in order]
        rows.append({'group':e['group'],'kind':e['kind'],'metrics':metrics(ordered,set(e['positives'])),
                     'order':ordered,'scores':[float(scores[i]) for i in order]})
    return {k:float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']},rows


def positive_mask(examples, codes):
    return [[c in e['positives'] for c in codes] for e in examples]


def train(cfg,smoke=False,resume=False):
    import numpy as np
    import torch
    import torch.nn.functional as F
    out=ROOT/cfg['artifacts'];data=read(out/'data.json')
    target=out/('smoke' if smoke else 'finetuned')
    state_path=out/('smoke_resume.pt' if smoke else 'resume.pt')
    if target.exists() and not resume: raise ValueError('Checkpoint exists; select --resume or a new experiment version')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_encoder(PRETRAINED,device)
    rng=random.Random(cfg['seed'])
    examples=[e for e in data['examples'] if e['split']=='train'];rng.shuffle(examples)
    examples=examples[:cfg['max_train_groups']]
    validation=[e for e in data['examples'] if e['split']=='validation']
    if smoke: examples=examples[:16];validation=validation[:16]
    # Baseline validation is recorded before any optimizer update. Test is untouched.
    baseline,_=evaluate_model(model,data,validation,cfg)
    write(out/('smoke_pretrained_validation.json' if smoke else 'pretrained_validation.json'),baseline)
    print('Pretrained validation',baseline,flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[];best=-1.;bad=0;start=0
    if resume and state_path.exists():
        state=torch.load(state_path,map_location=device,weights_only=False)
        if state['config'] != cfg or state['data_sha256'] != digest(out/'data.json'):
            raise ValueError('Resume config/data mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];bad=state['bad'];start=state['epoch']+1
        rng.setstate(state['random']);torch.set_rng_state(state['torch_rng'])
        if device=='cuda' and state.get('cuda_rng') is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])
    codes=sorted(data['documents'])
    for epoch in range(start,1 if smoke else cfg['epochs']):
        rng.shuffle(examples);losses=[];began=time.perf_counter()
        for offset in range(0,len(examples),cfg['batch_size']):
            batch=examples[offset:offset+cfg['batch_size']]
            # One uniformly sampled positive per query; known multi-positives remain positive across the batch.
            owners=[rng.choice(e['positives']) for e in batch]
            # Explicit random negatives; never silently treat shared-source owners as negatives.
            owners += rng.sample(codes,min(cfg['random_negatives'],len(codes)))
            owners=list(dict.fromkeys(owners))
            excluded={e['group'] for e in batch};texts=[t for e in batch for t in e['texts']]
            documents=[document(model,data['documents'][c],excluded,texts)[0] for c in owners]
            model.train();optimizer.zero_grad(set_to_none=True)
            def embed(text):
                features={k:v.to(device) if torch.is_tensor(v) else v for k,v in model.tokenize(text).items()}
                return F.normalize(model(features)['sentence_embedding'],dim=1)
            queries=embed([e['text'] for e in batch]);vectors=embed(documents)
            logits=queries@vectors.T/cfg['temperature']
            mask=torch.tensor(positive_mask(batch,owners),device=device)
            loss=(torch.logsumexp(logits,dim=1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),dim=1)).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
            if offset % (cfg['batch_size']*24)==0: print('epoch',epoch+1,'batch',offset//cfg['batch_size'],'loss',losses[-1],flush=True)
        val,_=evaluate_model(model,data,validation,cfg)
        train_metrics,_=evaluate_model(model,data,examples[:128],cfg)
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':val,'train_catalog_lookup':train_metrics,
                        'seconds':time.perf_counter()-began})
        # Train catalog includes training source text: explicitly a lookup diagnostic, not generalisation.
        if val['mrr']>best:
            best=val['mrr'];bad=0;model.save_pretrained(str(target))
        else: bad+=1
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'best':best,'bad':bad,
                    'epoch':epoch,'random':rng.getstate(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,
                    'config':cfg,'data_sha256':digest(out/'data.json')},state_path)
        write(out/('smoke_history.json' if smoke else 'history.json'),history)
        print('epoch',epoch+1,'validation',val,flush=True)
        if bad>=cfg['patience']:break
    write(target/'training_manifest.json',{'provenance':provenance(cfg),'data_sha256':digest(out/'data.json'),
          'training_groups':[e['group'] for e in examples], 'validation_groups':[e['group'] for e in validation],
          'device':device,'smoke_only':smoke,'best_validation_mrr':best,'history':history,
          'selected_checkpoint_sha256':fingerprint(target),'selection':'validation MRR only; test untouched',
          'loss':'multi-positive logsumexp contrastive softmax; shared-weight dual encoder; random negatives; temperature fixed'})


def build(cfg):
    import numpy as np
    out=ROOT/cfg['artifacts'];data=read(out/'data.json')
    for name,checkpoint in [('pretrained',PRETRAINED),('finetuned',out/'finetuned')]:
        path=out/(name+'_index')
        if path.exists():raise ValueError('Index exists; use a new experiment version')
        model=load_encoder(checkpoint)
        codes=sorted(data['documents']);built=[document(model,data['documents'][c]) for c in codes]
        vectors=encode(model,[x[0] for x in built],cfg['batch_size'])
        path.mkdir(parents=True)
        np.save(path/'vectors.npy',vectors);write(path/'codes.json',codes)
        write(path/'documents.json',{c:{'text':x[0],'coverage':x[1]} for c,x in zip(codes,built)})
        write(path/'manifest.json',{'checkpoint':str(checkpoint.relative_to(ROOT)).replace('\\','/'),
              'checkpoint_sha256':fingerprint(checkpoint),'catalog_sha256':digest(ROOT/'data/occupations.csv'),
              'builder':BUILDER,'normalization':'l2','dimension':384,'model_id':MODEL_ID,'revision':REVISION,
              'query_document_format':'plain text, no prefixes; mean pooling, 256 wordpieces maximum',
              'sources':data['sources'],'data_sha256':digest(out/'data.json'),
              'implementation_hashes':{n:digest(ROOT/n) for n in ['recommender/occupation_semantic.py','training/occupation_semantic.py']},
              'files':{n:digest(path/n) for n in ['codes.json','vectors.npy','documents.json']},'provenance':provenance(cfg)})
        print('Built',name,'index',vectors.shape,flush=True)


def evaluate(cfg):
    import numpy as np
    from recommender.models.ranker import RankingPipeline
    from recommender.schemas import TYPES
    out=ROOT/cfg['artifacts'];report=ROOT/cfg['report'];data=read(out/'data.json')
    if (report/'evaluation.json').exists(): raise ValueError('Evaluation exists; do not retune against test or overwrite')
    model=load_encoder(PRETRAINED)
    from pipeline.extractor import Extractor
    extractor=Extractor(ROOT/'models/extractor',ROOT/'data/labels.json')
    modes=['control','pretrained','finetuned','fusion']
    pipelines={name:RankingPipeline(config=ROOT/f'configs/ranking/occupation_semantic_{name}_v1.json') for name in modes}
    all_rows={};summary={}
    for split in ['test','occupation_heldout']:
        examples=[e for e in data['examples'] if e['split']==split]
        if split == 'occupation_heldout':
            # Shared statements can also label known occupations. Credit only held-out owners in this diagnostic.
            examples=[{**e,'positives':sorted(set(e['positives']) & set(data['heldout_codes']))} for e in examples]
        rows=[];began=time.perf_counter()
        for j,e in enumerate(examples):
            reading=extractor.read(e['text']);v=dict.fromkeys(TYPES,0.);obs=set()
            for s in reading.slots:v[s.slot]=s.value;obs.add(s.slot)
            row={'group':e['group'],'kind':e['kind'],'text':e['text'],'positives':e['positives'],'modes':{}}
            for mode,pipe in pipelines.items():
                resp=pipe.rank(v,obs,[e['text']],378)
                order=[o.code for o in resp.occupations]
                row['modes'][mode]={'metrics':metrics(order,set(e['positives'])),'order':order,
                                    'status':resp.status,'candidate_recall':len(set(order)&set(e['positives']))/len(e['positives'])}
            rows.append(row)
            if j%200==0: print(split,j,'/',len(examples),flush=True)
        summary[split]={'groups':len(rows),'seconds':time.perf_counter()-began,'modes':{}}
        for mode in modes:
            summary[split]['modes'][mode]={k:float(np.mean([r['modes'][mode]['metrics'][k] for r in rows])) for k in rows[0]['modes'][mode]['metrics']}
        # Paired group bootstrap; no pair-row resampling.
        rng=np.random.default_rng(cfg['seed'])
        for a,b in [('finetuned','pretrained'),('fusion','control')]:
            dif=np.array([r['modes'][a]['metrics']['mrr']-r['modes'][b]['metrics']['mrr'] for r in rows])
            boot=[float(np.mean(rng.choice(dif,size=len(dif),replace=True))) for _ in range(2000)]
            summary[split][f'{a}_minus_{b}_mrr']={'difference':float(dif.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
        all_rows[split]=rows
    write(report/'evaluation.json',{'summary':summary,'provenance':provenance(cfg),'split_manifest_sha256':digest(out/'split_manifest.json'),
          'ranking_depth':378,'label_definition':'binary multi-owner task/context retrieval proxy; not graded career suitability',
          'frozen_choices':'equal-weight RRF k60, checkpoint selected on validation MRR; test first opened here'})
    write(report/'per_query.json',all_rows)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','train','index','evaluate'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/occupation_semantic_v1.json')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cfg=read(args.config)
    if args.stage=='train':train(cfg,args.smoke,args.resume)
    else:{'prepare':prepare,'index':build,'evaluate':evaluate}[args.stage](cfg)


if __name__=='__main__':main()
