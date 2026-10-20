"""One central workflow configuration for preparation, evaluation and terminal use."""

import argparse
import json
import os
from pathlib import Path

from recommender.configuration import CONFIG_ROOT, load_configuration

ROOT = CONFIG_ROOT.parent


def load_workflow(path):
    config, _ = load_configuration(path)
    if set(config) != {"ranking", "evaluation", "preparation", "system"}:
        raise ValueError("workflow requires ranking, evaluation, preparation and system paths")
    if any(not isinstance(value, str) or not value for value in config.values()):
        raise ValueError("workflow paths must be nonempty strings")
    return {key: ROOT / value for key, value in config.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "chat"))
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "ranking/workflow.json")
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()
    if args.download_model and args.command != "prepare":
        parser.error("--download-model is only valid for prepare")
    if (args.output or args.history) and args.command != "evaluate":
        parser.error("--output and --history are only valid for evaluate")
    paths = load_workflow(args.config.resolve())
    # Existing dataset/model paths use repository-relative identities.
    previous = Path.cwd()
    output = args.output.resolve() if args.output else None
    history = args.history.resolve() if args.history else None
    try:
        os.chdir(ROOT)
        if args.command == "prepare":
            from recommender.preparation.build_dense import prepare
            result = prepare(paths["preparation"], download=args.download_model)
        elif args.command == "evaluate":
            from recommender.experiments.evaluate import run
            result = run(paths["evaluation"], output, ranking_config=paths["ranking"],
                         **({"history": history} if history else {}))
        else:
            from app.terminal import run
            from pipeline.system import System
            # Share the workflow's ranking selection with the conversation too.
            config, _ = load_configuration(paths["system"])
            config.setdefault("paths", {})["ranking_config"] = str(paths["ranking"])
            system = System.from_config(ROOT / "data", config)
            run(system)
            return
        print(json.dumps(result, indent=2))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
