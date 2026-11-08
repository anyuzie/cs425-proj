"""Real local encoder, terminal, TCP HTTP, restart and multi-turn acceptance (no mocks).

Browser observation is recorded separately, never inferred from HTTP success.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recommender.components import ComponentInput
from recommender.models.ranker import RankingPipeline
from recommender.schemas import TYPES
from pipeline.extractor import Extractor
from app.ranking_terminal import rank_utterance

CASES={
    'background':'My degree is in chemistry and I have experience analysing laboratory samples.',
    'no_preferences':'I studied information systems, cloud computing and artificial intelligence.',
    'positive':'I enjoy helping students understand mathematics.',
    'negation':'I do not want to teach mathematics.',
    'contradiction':'I enjoy laboratory work but I do not want laboratory work.',
    'mixed':'I enjoy designing software and teaching children.',
    'vague':'I am not sure what I want.',
    'nonsense':'flibberty zargle wump 928374',
    'empty':''}


def request(url,body=None):
    data=json.dumps(body).encode() if body is not None else None
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8017)
    args=parser.parse_args();report=ROOT/'reports/ranking/occupation_semantic_v1'
    extractor=Extractor(ROOT/'models/extractor',ROOT/'data/labels.json')
    evidence={'kind':'actual local models and actual TCP HTTP, no mocked components','cases':{},'http':{},'browser':'pending separate observed UI check'}
    pipelines={mode:RankingPipeline(config=ROOT/f'configs/ranking/occupation_semantic_{mode}_v1.json') for mode in ['control','pretrained','finetuned','fusion']}
    for mode,pipe in pipelines.items():
        evidence['cases'][mode]={}
        for name,text in CASES.items():
            began=time.perf_counter();reading,v,obs,resp=rank_utterance(text,extractor,pipe,top_k=378)
            elapsed=time.perf_counter()-began
            if mode!='control' and text:
                assert resp.status=='ranked'
                raw=pipe._components['m4'].score(ComponentInput(user_sentences=(text,)))
                bycode={x.code:x for x in raw.occupations}
                for o in resp.occupations:
                    assert o.signals['semantic'].score==bycode[o.code].score
                    assert o.signals['semantic'].rank==bycode[o.code].rank
                direct=pipe.rank(v,obs,[text],378)
                assert [asdict(o) for o in direct.occupations]==[asdict(o) for o in resp.occupations]
            evidence['cases'][mode][name]={'seconds':elapsed,'reading':asdict(reading),'status':resp.status,
                'observed_types':sorted(obs),
                'top5':[asdict(o) for o in resp.occupations[:5]],'diagnostics':resp.query_diagnostics}
        (report/'acceptance_in_progress.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        if mode in ['pretrained','finetuned','fusion']:
            # Controlled one-positive-I verifies semantic treatment is not code-sorted RIASEC ties.
            v=dict.fromkeys(TYPES,0.);v['I']=1.
            one=pipe.rank(v,{'I'},[CASES['background']],378)
            assert len({o.signals['semantic'].score for o in one.occupations})>300
            evidence['cases'][mode]['one_positive_I']={'top5':[asdict(o) for o in one.occupations[:5]],'semantic_distinct_scores':len({o.signals['semantic'].score for o in one.occupations})}
    for mode in ['pretrained','finetuned','fusion']:
        for restart in range(2 if mode=='finetuned' else 1):
            env=os.environ.copy();env['CM_CONFIG']=str(ROOT/f'configs/system.occupation-semantic-{mode}-v1.json')
            env['CM_LOG_DIR']=str(ROOT/'logs/occupation_semantic_acceptance')
            env['CM_STUDY_MODE']='0'
            base=f'http://127.0.0.1:{args.port}'
            log=(report/f'http_{mode}_{restart}.log').open('w')
            began=time.perf_counter()
            process=subprocess.Popen([sys.executable,'-m','uvicorn','app.server:app','--host','127.0.0.1','--port',str(args.port)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                for _ in range(180):
                    if process.poll() is not None:raise RuntimeError(f'Server exited; see {log.name}')
                    try:health=request(base+'/api/health');break
                    except (OSError,ValueError):time.sleep(1)
                else:raise RuntimeError('Server startup timed out')
                cold=time.perf_counter()-began
                session=request(base+'/api/session',{'participant_code':f'semantic-{mode}-{restart}','debug':True})
                sid=session['session_id'];turns=[]
                texts=[CASES['background'],'I also studied statistics.','I have no additional preferences.']
                for turn,text in enumerate(texts,1):
                    start=time.perf_counter();reply=request(base+'/api/chat',{'session_id':sid,'text':text,'message_id':f'turn-{turn}'})
                    latency=time.perf_counter()-start
                    trace=reply['trace']['ranking_trace'][-1]
                    assert trace['user_sentences']==texts[:turn]
                    assert trace['query_diagnostics']['occupation_semantic']['components']['m4']['status']=='usable'
                    if turn==1:
                        # Invoke the actual terminal entry point and compare all policy-depth rows.
                        run=subprocess.run([sys.executable,'-m','app.ranking_terminal','--terminal-config',f'configs/ranking_terminal/occupation_semantic_{mode}_v1.json','--text',text,'--json','--top-k','378'],cwd=ROOT,text=True,capture_output=True,check=True)
                        terminal=json.loads(run.stdout)
                        assert terminal['response']['occupations'][:100]==trace['occupations']
                        evidence['http'][f'{mode}_{restart}_terminal_match']=True
                    turns.append({'text':text,'seconds':latency,'response':reply})
                assert len(turns[-1]['response']['recommendations'])==5
                evidence['http'][f'{mode}_{restart}']={'cold_start_seconds':cold,'health':health,'turns':turns}
                if mode=='finetuned' and restart==1:
                    first=evidence['http']['finetuned_0']['turns'][0]['response']['trace']['ranking_trace'][0]['occupations']
                    assert first==turns[0]['response']['trace']['ranking_trace'][0]['occupations']
                    evidence['restart_ranking_exact']=True
                print('HTTP and terminal passed',mode,restart,flush=True)
                (report/'acceptance_in_progress.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
            finally:
                process.terminate();process.wait(timeout=30);log.close()
    (report/'acceptance.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print('Saved real model, terminal, HTTP and restart acceptance. Browser remains separate.',flush=True)


if __name__=='__main__':main()
