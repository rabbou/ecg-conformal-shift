#!/usr/bin/env bash
# Fetch the PhysioNet corpora that need no account and no agreement.
# Verified open (HTTP 200, unauthenticated) on 2026-08-23.
#
# Usage: ./scripts/fetch_open_corpora.sh [destination]   (default /home/ruben/data)
set -uo pipefail

DEST="${1:-/home/ruben/data}"
mkdir -p "$DEST"

# slug|version|directory|why it is worth the disk
CORPORA=(
  "challenge-2021|1.0.3|challenge2021|the deliberately SEEN target: ECG-FM was pre-trained on these, which is what makes the contamination contrast measurable"
  "ptbdb|1.0.0|ptbdb|148 MI subjects of 290, 15-lead, and ECG-FM explicitly excluded PTB from pre-training, so it is MI-labelled and uncontaminated"
  "incartdb|1.0.0|incart|acute MI and ischemia annotations, Russia, a third annotation culture"
  "ludb|1.0.1|ludb|tiny but carries explicit STEMI / non-STEMI labels with delineated wave boundaries"
)

for entry in "${CORPORA[@]}"; do
  IFS='|' read -r slug version dir why <<< "$entry"
  echo "=============================================================="
  echo "$slug v$version -> $DEST/$dir"
  echo "  reason: $why"
  wget -q -r -N -c -np -nH --cut-dirs=3 -P "$DEST/$dir" \
       --tries=5 --waitretry=10 "https://physionet.org/files/${slug}/${version}/" \
    && echo "  ok: $(du -sh "$DEST/$dir" | cut -f1)" \
    || echo "  FAILED: $slug (exit $?)"
done

echo "=============================================================="
df -BG --output=avail /home | tail -1 | tr -d ' ' | sed 's/^/free on \/home now: /'
du -sh "$DEST"/* 2>/dev/null | sort -h
