"""Download pinned-package English Stanza resources for offline terminal use."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recommender.utterance.stanza_pipeline import PACKAGES

if __name__ == "__main__":
    import stanza
    stanza.download("en", model_dir=str(ROOT / "models/stanza"), package=None, processors=PACKAGES)
