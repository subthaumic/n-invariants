"""Import modules of ``manuscript/`` by path.

Both applications in ``manuscript/`` have a ``sampling.py``, so the tests cannot import
them by bare name; each is loaded from its file under a distinct module name instead.
"""

import importlib.util
import sys
from pathlib import Path

MANUSCRIPT = Path(__file__).resolve().parents[1]


def load(relative_path: str):
    """The module at ``manuscript/<relative_path>``, e.g. ``"sars_cov2/sampling.py"``."""
    name = "manuscript_" + relative_path.removesuffix(".py").replace("/", "_")
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MANUSCRIPT / relative_path)
        module = importlib.util.module_from_spec(spec)
        # Registered before execution so that worker processes can unpickle its functions.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
