#!/usr/bin/env bash
# Re-download the three published code tables in mappings/ and check their
# digests against the ones mappings/NOTICE.md records.
#
# Usage: scripts/fetch_mappings.sh [destination]   (default: a temporary dir)
set -euo pipefail

DEST="${1:-$(mktemp -d)}"
mkdir -p "$DEST"

fetch() {
  local url="$1" name="$2" want="$3"
  curl -fsSL --retry 3 -o "$DEST/$name" "$url"
  local got
  got=$(sha256sum "$DEST/$name" | cut -d' ' -f1)
  if [ "$got" != "$want" ]; then
    echo "$name: sha256 $got, expected $want" >&2
    exit 1
  fi
  echo "$name ok"
}

fetch \
  "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/main/dx_mapping_scored.csv" \
  dx_mapping_scored.csv \
  fad13ad9f7ca230e7e6392ac8a264cb7cd157879525129f964c5f708eabb41d0

fetch \
  "https://raw.githubusercontent.com/UTU-Health-Research/dl-ecg-classifier/main/data/AHA_SNOMED_mapping.csv" \
  AHA_SNOMED_mapping.csv \
  23e0641aac859fd89ef8969402d844bb47bfb1caa6503bee451da359c5daeaae

fetch \
  "https://physionet.org/files/ptb-xl-plus/1.0.1/labels/mapping/ptbxlToSNOMED.csv" \
  ptbxlToSNOMED.csv \
  63ae57a3a51387da39a6225a86ee2248b9cffc8d54215cbf177060ad59764d9f

echo "three tables verified in $DEST"
