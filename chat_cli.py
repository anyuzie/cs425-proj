"""Terminal loop over the same System the server uses."""

import os
from pathlib import Path

from pipeline.interfaces import State
from pipeline.system import System


def main() -> None:
    system = System.load(os.environ.get("CM_DATA_DIR", Path(__file__).parent / "data"))
    state = State()
    print("Research prototype - not career advice. Ctrl-D to quit.\n")
    print(f"bot> {system.start(state).reply}")
    while not state.done:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            return
        if not text:
            continue
        result = system.step(state, text[:2000])
        print(f"bot> [{result.action}] {result.reply}")
        for rec in result.recommendations or []:
            print(f"  {rec['rank']}. {rec['title']} ({rec['code']}) - {rec['why']}")
            for task in rec["tasks"]:
                print(f"       - {task}")


if __name__ == "__main__":
    main()
