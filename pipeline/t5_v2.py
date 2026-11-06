import json
import logging
import re

from .t5 import T5Filler
from .templates import QuestionBankFiller, _pick

FOCUS = {
    'R': ('working with equipment and materials', 'repairing a broken machine', 'working outdoors or in a workshop'),
    'I': ('investigating why something happens', 'analysing data to find a pattern', 'solving problems with no obvious answer'),
    'A': ('creating your own approach to a task', 'designing how something looks', 'coming up with original ideas'),
    'S': ('helping or teaching people', 'helping a colleague who is struggling', 'talking with people throughout the day'),
    'E': ('pitching ideas and persuading people', 'leading a team and making decisions', 'negotiating deals or selling'),
    'C': ('following clear procedures', 'organising records and schedules', 'working to a predictable routine'),
}
STYLES = ('direct', 'scenario', 'workday')
KNOWN = {'R':'practical work','I':'research','A':'creative work','S':'helping people','E':'persuading people','C':'organising information'}
SUBJECT = {'R':'doing practical work','I':'doing research','A':'doing creative work','S':'helping people','E':'persuading people','C':'organising information'}


def prompt(fields):
    return 'career question v2: ' + json.dumps(fields, ensure_ascii=False)


class T5V2Filler(T5Filler):
    name = 't5-v2'
    uses_context = True

    @staticmethod
    def valid(text, kind, previous='', activity=''):
        if not T5Filler.valid(text,kind,previous,activity):
            return False
        return not any(marker in text.lower() for marker in (
            'last_user','known_interest','career question','reference:','action:','style:','context:','focus:')) and not re.match(r'^(direct|scenario|workday)\b', text, re.I)

    def fields(self, kind, payload):
        retry = kind == 'ABSTAIN'
        effective = payload.get('retry','CLARIFY') if retry else kind
        slot = payload['slot']
        variant = payload.get('clarify_index',payload.get('index',0)) % 3
        context = [] if effective == 'CLARIFY' else [payload['new_src']] if effective == 'PROBE' else [payload['old_src'],payload['new_src']]
        reference = QuestionBankFiller.fill(self,'CLARIFY',payload) if effective == 'CLARIFY' else (
            'Is this something you would enjoy, or mainly something you have experience with?' if effective == 'PROBE' else
            'Which statement better describes the work you would actually want to do?')
        focus = FOCUS[slot][variant] if effective == 'CLARIFY' else SUBJECT[slot]
        if payload.get('task') and effective in ('CLARIFY', 'PROBE'):
            focus = payload['task']['activity']
            context = context + [payload['task']['text']]
            if effective == 'PROBE':
                reference = f'Is {focus} something you enjoy, or mainly something you already know?'
        return dict(action=effective,type=self.type_names[slot],focus=focus,reference=reference,
                    context=context,retry=retry,previous_question=payload.get('previous_question','') if retry else '',
                    style=STYLES[payload.get('index',0)%3],last_user=payload.get('last_user',''),
                    known_interest=payload.get('known_interest','') if effective == 'CLARIFY' else '')

    def fill(self, action, payload):
        kind = getattr(action,'kind',action)
        fallback = QuestionBankFiller.fill(self,action,payload)
        if kind == 'ABSTAIN' and payload.get('repair'):
            return fallback
        if kind in ('OPENING','RECOMMEND'):
            return fallback
        fields = self.fields(kind,payload)
        payload['focus'] = fields['focus']
        payload['question_style'] = fields['style']
        try:
            question = self.generate(prompt(fields))
        except (RuntimeError,ValueError,OSError):
            logging.getLogger(__name__).exception('T5 v2 failed; using template')
            return fallback
        if not self.valid(question,fields['action'],fields['previous_question'],payload.get('task', {}).get('activity', '')):
            return fallback
        if fields['action'] == 'PROBE' and not re.search(r'experience|familiar|already know|from scratch|new to you', question, re.I):
            return fallback
        prefix = _pick(self.qb['ABSTAIN'],payload.get('index',0))+' ' if fields['retry'] else ''
        if fields['action']=='PROBE':
            prefix += f'You said "{fields["context"][0]}". '
        elif fields['action']=='CHALLENGE':
            prefix += f'Earlier you said "{fields["context"][0]}" and then "{fields["context"][1]}". '
        return prefix+question
