"""Make the checkout under test the one that is imported.

The scripts are checked by tests too, so they have to be importable.  So does
``ecs`` itself: an editable install points at the src of the checkout it was
installed from, which is the wrong one in a git worktree -- the tests would run
against another branch's modules and fail to collect on anything new.  Putting
this checkout's ``src`` at the front of the path is what makes the suite
describe the tree it sits in.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
