"""Tests of the article code in ``manuscript/``, kept apart from the package's test suite.

They check that the article code runs against the current package. Run them with
``uv run pytest manuscript/tests``.
"""

import sys
from pathlib import Path

# test_substitution.py and test_sars_pipeline.py import the JC / HB models. The two
# sampling.py modules of manuscript/ are loaded by path (manuscript_modules.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "substitution_models"))
