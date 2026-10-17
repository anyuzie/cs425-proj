"""Explicit mock upstream components; real configurable ranking and dialogue orchestration."""
import json
from pathlib import Path

import fakes  # registers test components only
from pipeline import registry
from pipeline.interfaces import State
from pipeline.system import System


SCRIPTS = [
    ["I love working with machines and tools.", "I hate repairing machines, it is boring.",
     "second", "I love analysing data."],
    ["I am unsure.", "My dad runs a sales company so that's what I know.",
     "Honestly I grew up around it, I don't enjoy selling.", "I love analysing data."],
]


def make_system(ranking_path=Path("configs/ranking.yaml")):
    config = json.loads(Path("tests/test_config.json").read_text())
    config["components"]["ranker"] = "configured"
    config["paths"]["ranking_config"] = str(ranking_path.resolve())
    parts = registry.build(config, Path("data"))
    names = json.loads(Path("data/labels.json").read_text())["type_names"]
    return System(**parts, type_names=names, input_mode="mock_tracker")


def capture(system):
    profiles, traces = [], []
    for i, texts in enumerate(SCRIPTS, 1):
        state = State()
        system.start(state)
        for text in texts:
            system.step(state, text)
        record = system.tracker_record(state, profile_id=f"script_{i}", revision="m5_scripted_fixture_v1")
        # Independent illustrative labels, not measured real-user relevance.
        record.update(target="15-2051.00", related={"short": [], "long": [], "supplemental": []})
        profiles.append(record)
        traces.append(system.trace(state))
    return profiles, traces
