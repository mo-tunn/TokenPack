"""Compatibility entry point for the versioned offline canonical analysis."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from submission.evaluation.reanalyze import main

if __name__ == "__main__":
    main()
