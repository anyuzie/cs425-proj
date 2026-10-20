"""Compatibility launcher for the independent-turn ranking terminal."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ranking_terminal import main, rank_utterance, show_result

if __name__ == "__main__":
    raise SystemExit(main())
