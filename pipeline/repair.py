"""Wording for explicit repair requests; never filters extractor readings."""
import re

PREFERENCE = re.compile(r"\b(?:enjoy|like|love|hate|dislike|prefer|rather|satisfying|interested|appeal)\b", re.I)
# shortcut: explicit phrase handling covers common repairs; expand with independently reviewed dialogues.
REPAIR = re.compile(r"\b(?:give (?:me |us )?(?:a |an )?(?:concrete )?example|what (?:do you|does that) mean|(?:do not|don't|don’t) (?:know|understand)|not sure|unsure|undecided|hard to answer|just testing|not told you|weather|cannot decide|no preference)\b|^\s*(?:hmm+|um+|uh+)[.!?\s]*$", re.I)


def needs_repair(text):
    # Preserve mixed replies that also contain an explicit preference.
    return bool(REPAIR.search(text) and not PREFERENCE.search(text))


EXAMPLES = {
    'R': ('assembling equipment and checking that it works', 'finding and replacing a faulty machine part', 'maintaining plants and equipment outdoors'),
    'I': ('testing possible explanations for a recurring problem', 'comparing measurements to understand a trend', 'trying several explanations for an unsolved problem'),
    'A': ('inventing a different way to present a project', 'designing a poster layout', 'brainstorming original ideas for a new product'),
    'S': ('explaining a difficult task to someone who needs help', 'showing a colleague how to complete a difficult task', 'answering questions from visitors throughout the day'),
    'E': ('presenting an idea and persuading a team to support it', 'allocating tasks and making a final team decision', 'negotiating a price with a customer'),
    'C': ('checking forms against a written checklist', 'updating records and arranging appointments', 'completing the same scheduled checks each day'),
}


def repair_question(payload):
    kind = payload.get('retry', 'CLARIFY')
    if kind == 'CHALLENGE':
        return (f'Earlier you said "{payload["old_src"]}" and then "{payload["new_src"]}". '
                'Which statement would you choose to describe the work you want now?')
    if kind == 'PROBE':
        return (f'You said "{payload["new_src"]}". Imagine {EXAMPLES[payload["slot"]][0]} at work. Setting your experience aside, '
                'would you enjoy that activity, or prefer to avoid it?')
    if payload.get('repair_count', 0) > 1:
        return ('It is okay to be unsure. For the activity we discussed, would you lean toward '
                'enjoying it, avoiding it, or leaving it undecided for now?')
    # Reuse the exact pending activity instead of rotating to another facet.
    focus = payload.get('focus')
    example = EXAMPLES[payload['slot']][payload.get('clarify_index', 0) % 3]
    if focus:
        return f'By {focus}, I mean a task such as {example}. Would you enjoy that activity, dislike it, or feel unsure?'
    return (f'For example, imagine {example} at work. '
            'Would you enjoy that activity, dislike it, or feel unsure?')
