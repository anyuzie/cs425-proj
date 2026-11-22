import csv
import json
import re

from .interfaces import TYPES


def task_stance(text):
    """Explicit short answers concern the named task, never an inferred global interest."""
    # Only unambiguous short forms are recognised.
    normalized = re.sub(r"[^\w\s']", '', text.lower().replace('’', "'")).strip()
    if normalized in {'yes', 'yes i would', 'yes definitely', 'i would enjoy that', 'i enjoy it'}:
        return 1
    if normalized in {'no', 'no i would not', "no i wouldn't", 'no definitely not', "i don't enjoy it", 'i dislike it'}:
        return -1
    return None


def load_tasks(tasks_path, catalog_path):
    with tasks_path.open(encoding='utf-8-sig', newline='') as handle:
        source = {(r['code'], r['Task ID']): r for r in csv.DictReader(handle)}
    result, seen = [], set()
    for row in json.loads(catalog_path.read_text(encoding='utf-8'))['tasks']:
        key = (row['occupation_code'], row['task_id'])
        if row['slot'] not in TYPES or key in seen or key not in source:
            raise ValueError(f'Invalid or duplicate question task: {row}')
        if not isinstance(row['activity'], str) or not row['activity'].strip():
            raise ValueError('Question task requires an activity')
        original = source[key]
        result.append(dict(row, text=original['Task'], occupation_title=original['Title']))
        seen.add(key)
    return tuple(result)


def select_task(tasks, state, slot):
    # Tasks come from the small hand-picked catalog in configs/question_tasks.json.
    if state.turn < 3:
        return None
    for occupation in state.ranking[:100]:
        for task in tasks:
            key = (task['occupation_code'], task['task_id'])
            if task['slot'] == slot and task['occupation_code'] == occupation.code and key not in state.asked_tasks:
                state.asked_tasks.add(key)
                return dict(task)
    return None


def task_question(kind, payload):
    activity = payload['task']['activity']
    if kind == 'PROBE':
        if payload.get('previous_question'):
            return (f'You said "{payload["new_src"]}". Imagine {activity} were new to you. '
                    'Would you still choose it, even without previous experience?')
        return (f'You said "{payload["new_src"]}". Would you choose {activity} because you enjoy it, '
                'or mainly because it is familiar?')
    if payload.get('repair'):
        return f'Imagine spending part of a working day {activity}. Would you enjoy that activity, dislike it, or feel unsure?'
    options = (f'Would you enjoy {activity} as part of your job?',
               f'How would you feel about {activity} during a normal working day?',
               f'Would a job involving {activity} appeal to you?')
    return options[payload.get('index', 0) % len(options)]
