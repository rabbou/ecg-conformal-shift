#!/usr/bin/env bash
# A separate environment for ECG-FM's fairseq_signals, which is not in this
# project's environment: it is a fairseq fork built from source (C++/Cython)
# with its own hydra/omegaconf pins.  transformers is held below 5 because
# fairseq_signals imports it at package import and 5.x disables itself on
# torch < 2.5, which then fails with "NameError: name 'torch' is not defined".
# The project's torch pin (2.2.2) is kept.
#
# Usage: ./scripts/setup_ecgfm_env.sh [env dir]   (default ~/.venvs/ecgfm)
# Then:  <env dir>/bin/python scripts/timing_probe.py --arm ecgfm
set -euo pipefail

ENV="${1:-$HOME/.venvs/ecgfm}"
SRC="$(dirname "$ENV")/fairseq-signals"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

[ -d "$SRC" ] || git clone --depth 1 https://github.com/Jwoo5/fairseq-signals.git "$SRC"
uv venv "$ENV" --python 3.11
uv pip install --python "$ENV/bin/python" "torch==2.2.2" "numpy<2" scipy pandas wfdb h5py \
    cython "setuptools<70" wheel
uv pip install --python "$ENV/bin/python" --no-build-isolation -e "$SRC"
uv pip install --python "$ENV/bin/python" "transformers<5"
uv pip install --python "$ENV/bin/python" --no-deps -e "$REPO"
"$ENV/bin/python" -c "import torch, fairseq_signals; print('fairseq_signals ok, torch', torch.__version__)"
