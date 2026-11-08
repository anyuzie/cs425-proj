"""Independent entry point; does not import or change the existing pipeline."""
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from clause_rewrite_lab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
