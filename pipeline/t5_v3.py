"""Focused question generation; background is retained in state, outside wording."""
from .t5_v2 import T5V2Filler, STYLES


class T5V3Filler(T5V2Filler):
    name = 't5-v3'

    def fields(self, kind, payload):
        fields = super().fields(kind, payload)
        # Background and the target competed for the subject in v2 generations.
        fields.update(known_interest='', last_user='')
        if kind == 'ABSTAIN':
            previous = payload.get('question_style', fields['style'])
            fields['style'] = STYLES[(STYLES.index(previous) + 1) % len(STYLES)]
        return fields
