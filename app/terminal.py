"""Terminal presentation; conversation logic stays in pipeline.System."""

import argparse
import os
from pathlib import Path
from pipeline.interfaces import State
from pipeline.system import System

ROOT = Path(__file__).resolve().parents[1]


def run(system, *, read=input, write=print):
    """Run any System-compatible conversation with injectable terminal I/O."""
    state = State()
    write("Research prototype - not career advice. Ctrl-D to quit.\n")
    write(f"bot> {system.start(state).reply}")
    while not state.done:
        try:
            text = read("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            write()
            return
        if not text:
            continue
        result = system.step(state, text[:2000])
        write(f"bot> [{result.action}] {result.reply}")
        for rec in result.recommendations or []:
            write(f"  {rec['rank']}. {rec['title']} ({rec['code']}) - {rec['why']}")
            for task in rec["tasks"]:
                write(f"       - {task}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-config", type=Path, default=ROOT / "configs/system.json")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("CM_DATA_DIR", ROOT / "data")))
    args = parser.parse_args()
    run(System.load(args.data_dir, config_path=args.system_config))


if __name__ == "__main__":
    main()
