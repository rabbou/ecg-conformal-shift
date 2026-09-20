"""Where the corpora live, and the label vocabulary shared across them."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# PTB-XL is not copied: the 4.5 GB already on disk from an earlier project is
# read in place.  Override with ECS_PTBXL_DIR when it moves.
PTBXL_DIR = Path(os.environ.get("ECS_PTBXL_DIR", Path.home() / "Developer/ptbxl5d/data"))
SPH_DIR = Path(os.environ.get("ECS_SPH_DIR", REPO_ROOT / "data/sph"))
ACS_DIR = Path(os.environ.get("ECS_ACS_DIR", REPO_ROOT / "data/acs"))
RESULTS_DIR = Path(os.environ.get("ECS_RESULTS_DIR", REPO_ROOT / "results"))

# The PhysioNet/CinC Challenge-2021 bundle, read only to count how much
# infarction its non-PTB-XL partitions carry.  It lives on the box, not in
# this repository, and nothing else here depends on it being present.
CHALLENGE2021_DIR = Path(
    os.environ.get("ECS_CHALLENGE2021_DIR", Path.home() / "data/challenge2021")
)

# The published code tables the label mapping joins on, copied into the
# repository so the table can be rebuilt without a network.  Provenance,
# licence and digest of each are in mappings/NOTICE.md.
MAPPINGS_DIR = Path(os.environ.get("ECS_MAPPINGS_DIR", REPO_ROOT / "mappings"))

# The cached encoder representations.  They are large, they are not committed,
# and one machine's copy serves every checkout of this repository, so the
# default sits inside the results directory and ECS_EMBEDDINGS_DIR moves it
# somewhere shared.  A cache inside a checkout is also a cache a checkout of
# another branch will find, which is how an arm that branch does not know about
# ends up in its test run.
EMBEDDINGS_DIR = Path(os.environ.get("ECS_EMBEDDINGS_DIR", RESULTS_DIR / "embeddings"))

# PTB-XL ships the MI superclass with five subendocardial-injury statements
# folded in.  They are a different clinical entity from an infarct pattern, so
# the mapping keeps them separable rather than deciding for the caller.
PTBXL_INJURY_STATEMENTS = frozenset({"INJAS", "INJAL", "INJIN", "INJLA", "INJIL"})

# Shandong codes its diagnoses in the AHA scheme.  Category M is infarction.
SPH_MI_CODES = frozenset({"160", "161", "165", "166"})
SPH_MODIFIER_ACUTE = "330"
SPH_MODIFIER_RECENT = "331"
SPH_MODIFIER_OLD = "332"

# Chongqing codes its diagnoses as one column per finding.  AMI is the corpus's
# own acute-myocardial-infarction column and is the union of its STEMI and
# NSTEMI columns on all but two of the 17,960 labelled records.
ACS_MI_COLUMN = "AMI"
ACS_LABELLED_SPLIT = "CSV/train.csv"

# Every corpus is read at 500 Hz and cropped to ten seconds.  PTB-XL and
# Chongqing records are exactly that long; Shandong records run from ten to
# sixty seconds.
SAMPLING_RATE_HZ = 500
WINDOW_SAMPLES = 5000
N_LEADS = 12
