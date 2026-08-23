#!/usr/bin/env bash
# Fetch EchoNext v1.1.1 from PhysioNet onto the Linux box.
#
# EchoNext is restricted: you must have a PhysioNet account AND have signed the
# Data Use Agreement at https://physionet.org/content/echonext/1.1.1/ before this
# will return anything but 401.  That signature is yours to give; this script
# only moves bytes afterwards.
#
# Credentials are read from ~/.netrc on the box, so no password is ever passed
# on a command line, stored here, or printed.  Create it once, on the box:
#
#     printf 'machine physionet.org\n  login YOUR_USERNAME\n  password YOUR_PASSWORD\n' >> ~/.netrc
#     chmod 600 ~/.netrc
#
# Usage:  ./scripts/fetch_echonext.sh [destination]     (default /home/ruben/data/echonext)

set -euo pipefail

VERSION="1.1.1"
BASE="https://physionet.org/files/echonext/${VERSION}/"
DEST="${1:-/home/ruben/data/echonext}"

if [[ ! -f "$HOME/.netrc" ]] || ! grep -q 'physionet\.org' "$HOME/.netrc"; then
  echo "error: no physionet.org entry in ~/.netrc — see the header of this script." >&2
  exit 2
fi

# Cheap auth probe before committing to a multi-gigabyte pull: a signed-in user
# gets 200, an unsigned or unregistered one gets 401/403.
echo "probing access to ${BASE} ..."
code=$(curl -s -o /dev/null -w '%{http_code}' --netrc --max-time 60 -L "$BASE")
case "$code" in
  200) echo "access confirmed (HTTP 200)" ;;
  401|403) echo "error: HTTP $code — the account is not signed onto the DUA for this project yet." >&2; exit 3 ;;
  *)   echo "error: unexpected HTTP $code from PhysioNet." >&2; exit 4 ;;
esac

avail_gb=$(df -BG --output=avail "$(dirname "$DEST")" | tail -1 | tr -dc '0-9')
echo "destination ${DEST} — ${avail_gb} GB free"
[[ "$avail_gb" -lt 40 ]] && { echo "error: want at least 40 GB free." >&2; exit 5; }

mkdir -p "$DEST"
# -c resumes a partial file, -N skips what is already current, -np stays inside
# the version directory.  Safe to re-run after an interruption.
wget --netrc -r -N -c -np -nH --cut-dirs=3 -P "$DEST" \
     --progress=dot:giga --tries=5 --waitretry=10 "$BASE"

echo
echo "downloaded $(du -sh "$DEST" | cut -f1) to ${DEST}"
if [[ -f "$DEST/SHA256SUMS.txt" ]]; then
  echo "verifying checksums ..."
  ( cd "$DEST" && sha256sum -c SHA256SUMS.txt --quiet ) && echo "all checksums match"
else
  echo "note: no SHA256SUMS.txt published with this version; skipping verification"
fi
find "$DEST" -maxdepth 1 -type f -printf '%10s  %p\n' | sort -rn | head -20
