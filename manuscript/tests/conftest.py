"""Tests of the article code, run in its own environment (``manuscript/pyproject.toml``).

Run them from ``manuscript/`` with ``uv run --extra sars pytest``.
"""

import sys
from pathlib import Path

# test_substitution.py and test_sars_pipeline.py import the JC / HB models. The two
# sampling.py modules of manuscript/ are loaded by path (manuscript_modules.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "substitution_models"))
