import json, collections, numpy as np
T="RIASEC"
key=json.load(open('work/relabel_key.json')); b={}
for j in range(3): b.update(json.load(open(f'work/relabel_out_{j}.json')))
n=len(key); exact=0; dis=[]; per=collections.Counter(); perN=collections.Counter(); reason_ok=0
conf=collections.Counter()
for sid,k in key.items():
    g=b[sid]; ok=all(g[t]==k['slots'][t] for t in T)
    exact+=ok; reason_ok+= g['reason']==k['reason']
    for t in T:
        if k['slots'][t]!='neutral' or g[t]!='neutral':
            perN[t]+=1; per[t]+= g[t]==k['slots'][t]
        conf[(k['slots'][t],g[t])]+=1
    if not ok or g['reason']!=k['reason']: dis.append(sid)
print(f"sentences {n}; all six labels agree {exact} ({exact/n:.1%}); reason agrees {reason_ok/n:.1%}")
print("per type agreement where either side marks a stance:",{t:f"{per[t]/perN[t]:.0%}" for t in T})
print("data label -> blind label:")
for a in ['positive','negative','neutral']: print(' ',a,{c:conf[(a,c)] for c in ['positive','negative','neutral']})
# type-of-disagreement breakdown
kinds=collections.Counter()
for sid in dis:
    k=key[sid]; g=b[sid]
    for t in T:
        a,c=k['slots'][t],g[t]
        if a!=c: kinds[('flip' if {a,c}=={'positive','negative'} else 'missed stance' if c=='neutral' else 'extra stance')]+=1
    if g['reason']!=k['reason']: kinds[f"reason {k['reason']}->{g['reason']}"]+=1
print("disagreement kinds:",dict(kinds))
by=collections.Counter(key[s]['seed_id'].split('_')[0] if '+' not in key[s]['seed_id'] else 'combo' for s in dis); tot=collections.Counter(k['seed_id'].split('_')[0] if '+' not in k['seed_id'] else 'combo' for k in key.values())
print("disagree rate by source:",{s:f"{by[s]}/{tot[s]}" for s in tot})
json.dump(dis,open('work/dis_ids.json','w'))
