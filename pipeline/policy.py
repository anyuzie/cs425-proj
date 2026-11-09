from statistics import pvariance

from .interfaces import MAX_TURNS, TYPES, Action, SlotState, State
from .repair import needs_repair
from .task_questions import select_task

PROBE_REASON_CONF = 0.60   # default for settings.probe_reason_conf (manual section 7)


class PolicyBase:
    """Shared machinery. Subclasses implement _decide(state) and return an Action; decide() completes it."""

    name = "policy-base"

    def __init__(self, max_turns: int = MAX_TURNS, no_probing: bool = False) -> None:
        self.max_turns = max_turns
        self.no_probing = no_probing   # True = never CHALLENGE or PROBE (the evaluation's control arm)

    def _decide(self, state: State) -> Action:
        raise NotImplementedError

    def _finish(self, state: State) -> None:
        """Called once before the forced RECOMMEND at max_turns."""

    def decide(self, state: State) -> Action:
        # Guaranteed here, not left to _decide: the conversation always ends with recommendations,
        # and the control arm can never probe (otherwise the full vs no_probing comparison is invalid).
        if state.turn >= self.max_turns:
            self._finish(state)   # e.g. settle contradictions still held, so both arms recommend from what was said
            action = Action("RECOMMEND", None, f"turn {state.turn} reached max {self.max_turns}")
        else:
            action = self._decide(state)
            if self.no_probing and action.kind in ("CHALLENGE", "PROBE"):
                raise ValueError(f"policy returned {action.kind} while no_probing is on; skip rungs 1-2 in _decide")
        payload = dict(action.payload)
        if action.kind == 'RECOMMEND':
            payload['uncertain_slots'] = sorted((set(TYPES) - set(state.slots)) | state.settled | set(state.unresolved_conflicts))
        payload["index"] = state.kind_count.get(action.kind, 0)
        if action.slot:
            payload["slot"] = action.slot
            payload["clarify_index"] = (action.payload.get('clarify_index', 0) if action.kind == 'ABSTAIN'
                                        else state.clarify_count.get(action.slot, 0))
        action.payload = payload
        if action.kind in ("CLARIFY", "ABSTAIN") and action.slot:
            state.clarify_count[action.slot] = state.clarify_count.get(action.slot, 0) + 1
        state.kind_count[action.kind] = state.kind_count.get(action.kind, 0) + 1
        return action


class ManualPolicy(PolicyBase):
    name = "manual"
    RETRIES = 1   # unsure answers: rephrase once (with a concrete example), then move on

    def __init__(self, max_turns: int = MAX_TURNS, no_probing: bool = False,
                 probe_reason_conf: float = PROBE_REASON_CONF, task_questions=()) -> None:
        super().__init__(max_turns, no_probing)
        if isinstance(probe_reason_conf, bool) or not isinstance(probe_reason_conf, (int, float)) \
                or not 0 <= probe_reason_conf <= 1:
            raise ValueError("probe_reason_conf must be a number in [0, 1]")
        self.probe_reason_conf = float(probe_reason_conf)   # manual section 7: PROBE needs reason confidence above this
        self.task_questions = task_questions

    def _finish(self, state: State) -> None:
        # A contradiction confirmed on the last turn was never asked about: settle it as a stuck one
        # (newer statement at half confidence), the way the control arm merges what it heard.
        for flag in list(state.flags):
            self._give_up(state, "CHALLENGE", flag.slot)

    def _give_up(self, state: State, retry: str, slot: str) -> str:
        """After one failed retry: settle the question and move on. Returns a note for the action log."""
        if retry == "CHALLENGE":
            flag = next((f for f in state.flags if f.slot == slot), None)
            state.flags = [f for f in state.flags if f.slot != slot]
            if flag:   # manual section 7: keep the newer statement at half confidence, log it as unresolved
                new = flag.new
                old = state.slots.get(slot)
                history = old.history if old else [new]   # the tracker already recorded the newer reading
                state.slots[slot] = SlotState(new.value, new.confidence / 2, new.reason, new.source, new.turn,
                                              history, new.reason_confidence, new.confidence / 2)
                state.unresolved_conflicts.append(slot)
                state.profile_changed = True   # System re-ranks; no RECOMMEND on a turn the profile changed
            state.resolved.add(slot)   # closed (not answered): the rule must not raise it again
            return f"challenge on {slot} unresolved after a retry; kept the newer statement at half confidence"
        if retry == "PROBE":
            return f"probe on {slot} unanswered after a retry; kept the reading as it was"
        state.settled.add(slot)   # no clear preference: counts as covered for recommending
        return f"{slot} still unsure after a retry; marked as no clear preference"

    def _decide(self, state: State) -> Action:
        last = state.last_action
        pending = last and last.kind in ("CLARIFY", "CHALLENGE", "PROBE", "ABSTAIN")
        latest_task = state.task_history[-1] if state.task_history else None
        task_only = bool(pending and last.payload.get('task') and latest_task
                         and latest_task['turn'] == state.turn and latest_task.get('preference') is not None
                         and not any(r['slot'] == last.slot for r in latest_task['readings']))
        neutral_retry = bool(pending and state.last_reading_empty and not task_only)
        note, just_settled = "", None
        if pending and last.slot and not neutral_retry:
            state.strikes.pop(last.slot, None)   # a usable answer resets the retry count
        if neutral_retry:
            retry = last.payload.get("retry", "CLARIFY") if last.kind == "ABSTAIN" else last.kind
            slot = last.slot
            state.strikes[slot] = state.strikes.get(slot, 0) + 1
            revisit = retry == "CLARIFY" and slot in state.revisited
            if state.strikes[slot] <= self.RETRIES and not revisit and not (
                    self.no_probing and retry in ("CHALLENGE", "PROBE")):
                payload = {**last.payload, "retry": retry, "previous_question": state.last_reply}
                asked_meaning = bool(state.user_texts) and needs_repair(state.user_texts[-1])
                if retry == "CLARIFY" and not asked_meaning and not last.payload.get('task'):
                    # an unreadable answer: next phrasing (manual section 7). "What do you mean?" / "not sure"
                    # keeps the same question and gets the concrete-example wording instead.
                    payload["clarify_index"] = last.payload.get("clarify_index", 0) + 1
                why = ("asked what it means: same question with an example" if asked_meaning
                       else "no stance; ask once more with the next phrasing")
                return Action("ABSTAIN", slot, why, payload)
            state.strikes.pop(slot, None)
            note = self._give_up(state, retry, slot) if slot else ""
            just_settled = slot   # come back later, not straight away
            neutral_retry = False   # handled: carry on to the next question
        probe = next((t for t in TYPES if t in state.slots and state.slots[t].reason == "familiarity"
                      and state.slots[t].reason_confidence > self.probe_reason_conf
                      and t not in state.probed), None)
        snapshots = state.profile_history[-3:]
        covered = set(state.slots) | state.settled   # filled, or marked as no clear preference
        converged = len(snapshots) == 3 and all(
            t in a and t in b and abs(a[t] - b[t]) <= .1 + 1e-12
            for a, b in zip(snapshots, snapshots[1:]) for t in state.slots)
        if state.turn >= 5 and len(covered) == 6 and state.slots and converged and not state.profile_changed and (
                self.no_probing or (not state.flags and probe is None)):
            return Action("RECOMMEND", why=_join(note, "all six interests covered and stable over two updates"))
        if not self.no_probing:
            if state.flags:
                flag = state.flags[0]
                return Action("CHALLENGE", flag.slot, _join(note, "opposite signs confirmed by NLI"),
                              {"old_src": flag.old.source, "old_turn": flag.old.turn, "new_src": flag.new.source})
            if probe:
                state.probed.add(probe)
                payload = {"new_src": state.slots[probe].source}
                source = state.slots[probe].source
                # a short reply's source is the restated statement (pipeline/answers.py), kept in the row's readings
                previous = next((row for row in reversed(state.task_history)
                                 if row['slot'] == probe and (source in row['reply'] or any(
                                     r.get('source') == source for r in row.get('readings', [])))), None)
                if previous:
                    payload['task'] = dict(previous['task'])
                return Action("PROBE", probe, _join(note, "familiarity-based reason, not yet probed"),
                              payload)
        unfilled = [t for t in TYPES if t not in covered]
        later = [t for t in TYPES if t in state.settled and t not in state.slots and t not in state.revisited
                 and t != just_settled]
        if unfilled:
            pool = state.ranking[:100]
            slot = max(unfilled, key=lambda t: pvariance([o.occupation_z[t] for o in pool]) if pool else 0)
            why = "unfilled interest with largest top-100 occupation variance"
        elif later and state.turn < self.max_turns:
            slot = later[0]   # turns to spare: ask an unsure type once more, with its next phrasing
            state.revisited.add(slot)
            why = f"turns to spare: ask {slot} once more (it was unsure earlier)"
        elif state.slots:
            slot = min(state.slots, key=lambda t: state.slots[t].confidence)
            why = "refine least-confident interest while waiting for convergence or minimum turns"
        else:
            slot = min(TYPES, key=lambda t: state.clarify_count.get(t, 0))
            why = "nothing stated yet: ask the least-asked interest again"
        task = select_task(self.task_questions, state, slot)
        payload = {'task': task} if task else {}
        if task:
            why += f'; O*NET task {task["task_id"]} from {task["occupation_title"]}'
        return Action("CLARIFY", slot, _join(note, why), payload)


def _join(note: str, why: str) -> str:
    return f"{note}; {why}" if note else why
