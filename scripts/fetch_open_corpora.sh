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

# ACS-ECG / Chongqing (figshare, CC0, no account).  The three files of figshare
# article 29925314 (Sci Data 2026, 10.1038/s41597-026-07278-0): the label tables,
# the median beats, and the raw 10 s tracings.  Sizes and MD5 sums are the ones
# figshare publishes per file (api.figshare.com/v2/articles/29925314, 2026-08-23).
checksum() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi; }
ACS_DEST="$DEST/acs"
mkdir -p "$ACS_DEST"
# figshare file id|name|size|md5
ACS_FILES=(
  "62951134|CSV.zip|0.3 MB|1cb46279c0e68e6512bd39214fc56528"
  "62951137|ECG_median_data.zip|140 MB|4da1650cdbfb9817aa66df58c66564e1"
  "62951395|ECG_row_data.zip|1.28 GB|acea6ca86a2d0b937ecfe7d1df6d30a2"
)
for entry in "${ACS_FILES[@]}"; do
  IFS='|' read -r file_id name size md5 <<< "$entry"
  echo "=============================================================="
  echo "acs: $name ($size) -> $ACS_DEST/$name"
  if curl -L -C - --retry 5 --retry-delay 10 -o "$ACS_DEST/$name" \
          "https://ndownloader.figshare.com/files/${file_id}"; then
    got="$(checksum "$ACS_DEST/$name")"
    [ "$got" = "$md5" ] && echo "  ok, md5 matches" || echo "  MD5 MISMATCH: expected $md5, got $got"
  else
    echo "  FAILED: $name (exit $?)"
  fi
done

echo "=============================================================="
df -BG --output=avail /home | tail -1 | tr -d ' ' | sed 's/^/free on \/home now: /'
du -sh "$DEST"/* 2>/dev/null | sort -h
