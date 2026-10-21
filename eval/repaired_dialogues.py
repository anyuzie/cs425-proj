"""Replay actual input scripts through all production components and audit T5."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch

from pipeline.system import System
from pipeline.interfaces import State
from eval.t5_wording import AuditedT5V3, controlled, summarize
from eval.t5_compare import heldout, test_summary

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--test-data',type=Path,default=ROOT.parent/'data/t5_questions_v3b/test.jsonl')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    config=json.loads((ROOT/'configs/system.t5-v3.json').read_text())
    system=System.from_config(ROOT/'data',config)
    # Reuse the loaded checkpoint, avoiding a duplicate model instance.
    system.filler.__class__=AuditedT5V3
    filler=system.filler;filler.records=[]
    controls=controlled(filler)
    (args.output/'controlled.json').write_text(json.dumps(controls,indent=2),encoding='utf-8')
    print('Controlled',summarize(controls),flush=True)
    testrows=[json.loads(s) for s in args.test_data.read_text().splitlines()]
    generations=heldout(filler,testrows,2)
    (args.output/'heldout.json').write_text(json.dumps(generations,indent=2),encoding='utf-8')
    sessions=[]
    cases=[]
    for path in sorted((ROOT/'reports/t5/live_inputs_20261021T073446Z').glob('*.json')):
        case=json.loads(path.read_text(encoding='utf-8'))
        if 'inputs' not in case:continue
        cases.append(case)
    cases.extend([
        dict(case='challenge_repair',inputs=['I enjoy repairing broken machines.',
            'I hate repairing broken machines.', 'hmm',
            'I enjoy analysing scientific data and investigating research questions.', 'The second one.']),
        dict(case='probe_repair',inputs=['I studied accounting because it was a safe option.',
            'Could you give me a concrete example?', 'I dislike keeping financial records.']),
    ])
    for case in cases:
        state=State();system.start(state);turns=[]
        for text in case['inputs']:
            result=system.step(state,text)
            turns.append(dict(user=text,response=asdict(result),profile=system.profile(state),
                              flags=[asdict(f) for f in state.flags],wording=filler.records[-1]))
        sessions.append(dict(case=case['case'],turns=turns))
        (args.output/(case['case']+'.json')).write_text(json.dumps(sessions[-1],indent=2),encoding='utf-8')
        print('Conversation',case['case'],flush=True)
    checks=[]
    bycase={s['case']:s['turns'] for s in sessions}
    for name,case,turn,slot,sign in [('shop_negative','family_shop',1,'E',-1),
        ('records_negative','education',1,'C',-1),('creative_preference','education',1,'A',1),
        ('choice_resolution','contradiction',2,'R',-1),('helping_recall','multiple_interests',0,'S',1)]:
        value=bycase[case][turn]['profile'][slot]['value']
        checks.append(dict(name=name,passed=value is not None and value*sign>0,value=value))
    for case,turn in [('uncertain',1),('off_topic',1)]:
        profile=bycase[case][turn]['profile']
        checks.append(dict(name=case+'_no_invented_preference',passed=all(s['value'] is None for s in profile.values())))
    student=bycase['cs_student'][0]
    checks.append(dict(name='enjoyment_over_education',passed=student['profile']['I']['reason']=='enjoyment'))
    challenge=bycase['challenge_repair']
    checks.append(dict(name='challenge_survives_hesitation_and_unrelated_answer',
        passed=bool(challenge[2]['flags']) and bool(challenge[3]['flags'])))
    checks.append(dict(name='challenge_resolves_explicit_choice',
        passed=not challenge[4]['flags'] and challenge[4]['profile']['R']['value']<0))
    probe=bycase['probe_repair']
    checks.append(dict(name='probe_repair_preserves_profile',
        passed=probe[0]['profile']==probe[1]['profile'] and probe[1]['response']['action']=='ABSTAIN'))
    checks.append(dict(name='probe_negative_after_repair',passed=probe[2]['profile']['C']['value']<0))
    summary=dict(config=config,components=system.components,controlled=summarize(controls),
        heldout=test_summary(generations),live=summarize([t['wording'] for s in sessions for t in s['turns']]),
        checks=checks,checkpoint_sha256=hashlib.sha256((ROOT/config['paths']['t5_checkpoint']/'model.safetensors').read_bytes()).hexdigest(),
        caveat='Development checks, not population accuracy. Fixed human-readable inputs; no simulated answers to policy targets.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=['# Actual-input replay: revised dialogue', '',
            'All production components are loaded. Inputs are fixed scripts, not answers generated from policy targets.', '',
            '## Controlled generation', '', '```json',json.dumps(summary['controlled'],indent=2),'```','',
            '## End-to-end checks', '']
    report.extend(f'- {c["name"]}: {"PASS" if c["passed"] else "FAIL"}' for c in checks)
    for session in sessions:
        report.extend(['',f'## {session["case"]}',''])
        for turn in session['turns']:
            profile=', '.join(f'{s}={v["value"]:+.2f}' for s,v in turn['profile'].items() if v['value'] is not None) or 'No preferences'
            report.extend([f'**User:** {turn["user"]}', '',f'**Assistant ({turn["response"]["action"]}, {turn["wording"]["source"]}):** {turn["response"]["reply"]}', '',f'Tracked: {profile}', ''])
    (args.output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
