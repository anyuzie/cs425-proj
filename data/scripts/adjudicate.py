import json,collections
from sklearn.metrics import f1_score
T=list('RIASEC')
S=json.load(open('stress_all.json'))
lab={'x':{},'y':{}}
for rd in 'xy':
    for c in '01':
        for r in json.load(open(f'stress_lab_{rd}{c}.json')): lab[rd][r['id']]=r
SV={'dontmind':0,'mild':1,'strong':2}; INV={v:k for k,v in SV.items()}
A=collections.Counter(); out=[]; three=[]; yt,yp=[],[]; flips=0
for g in S:
    x=lab['x'][g['sid']]; y=lab['y'][g['sid']]; fs,fl={},{}
    for t in T:
        votes=[g['slots'][t],x['slots'][t],y['slots'][t]]; c=collections.Counter(votes).most_common()
        d=c[0][0] if c[0][1]>=2 else g['slots'][t]
        if c[0][1]<2: three.append((g['sid'],t,votes))
        fs[t]=d
        for l in (x,y):
            A['dir_same']+=l['slots'][t]==g['slots'][t]; yt.append(g['slots'][t]); yp.append(l['slots'][t])
            flips+={l['slots'][t],g['slots'][t]}=={'positive','negative'}
            if d!='neutral' and l['slots'][t]==d==g['slots'][t]:
                A['str_n']+=1; A['str_same']+=l['strength_label'].get(t)==g['strength_label'].get(t)
                A['str_within1']+=abs(SV[l['strength_label'].get(t,'mild')]-SV[g['strength_label'].get(t,'mild')])<=1
        if d!='neutral':
            s=sorted(SV[l['strength_label'][t]] for l in (g,x,y) if l['slots'][t]==d and t in l['strength_label'])
            fl[t]=INV[s[len(s)//2]]
    rv=collections.Counter([g['reason'],x['reason'],y['reason']]).most_common()
    reason=rv[0][0] if rv[0][1]>=2 else g['reason']
    if all(v=='neutral' for v in fs.values()): reason='neither'
    elif reason=='neither': reason='enjoyment'
    A['reason_same']+=(x['reason']==g['reason'])+(y['reason']==g['reason'])
    A['changed']+= fs!=g['slots'] or fl!=g['strength_label'] or reason!=g['reason']
    out.append({'sid':g['sid'],'text':g['text'],'variation':g['variation'],'slots':fs,'strength_label':fl,'reason':reason})
n=len(S)*6*2
print('direction agreement',round(A['dir_same']/n,3),'macro-F1',round(f1_score(yt,yp,average='macro'),3),'like/dislike flips',flips,'of',n)
print('strength exact',round(A['str_same']/A['str_n'],3),'within one',round(A['str_within1']/A['str_n'],3))
print('reason agreement',round(A['reason_same']/800,3),'| 3-way splits',len(three),'| rows changed by majority',A['changed'])
json.dump(out,open('stress_final.json','w'),indent=1)
