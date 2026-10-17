"""Replay chronological tracker snapshots without supplying evaluation labels to rankers."""

from .schemas import RankingRequest


def runtime_requests(profile, input_mode):
    if profile.get("input_mode") != input_mode:
        raise ValueError("profile input_mode must match evaluation input_mode")
    provenance = profile.get("upstream")
    if not isinstance(provenance, dict) or any(
            not isinstance(provenance.get(k), str) or not provenance[k].strip()
            for k in ("extractor", "tracker", "revision")):
        raise ValueError("tracker profiles require upstream extractor, tracker and revision provenance")
    if input_mode == "actual_tracker" and any(
            provenance[k].startswith("test-") for k in ("extractor", "tracker")):
        raise ValueError("test components cannot be labelled actual_tracker")
    turns = profile.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError("tracker profiles require nonempty chronological turns")
    history, requests = [], []
    for turn in turns:
        if not isinstance(turn, dict) or not {"text", "interests", "observed_types"} <= turn.keys():
            raise ValueError("each turn requires text, interests and observed_types")
        if not isinstance(turn["text"], str) or not turn["text"].strip():
            raise ValueError("turn text must be a nonempty user utterance")
        observed = turn["observed_types"]
        if not isinstance(observed, list) or any(not isinstance(t, str) for t in observed) or len(set(observed)) != len(observed):
            raise ValueError("observed_types must be a unique list of RIASEC keys")
        history.append(turn["text"])
        requests.append(RankingRequest(dict(turn["interests"]), set(observed), list(history)))
    return requests
