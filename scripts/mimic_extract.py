"""Extract named MIMIC-IV-ECG records from the release zip, digest-checked.

The rotation and the shadow run read about 27,000 of the release's 800,035
records.  This pulls their ``.hea`` and ``.dat`` members out of the 36 GB zip
with Python's ``zipfile`` (no ``unzip`` needed), checks every file against the
release's own ``SHA256SUMS.txt``, and either writes them under ``--dest`` in the
release layout or streams them as a tar to ``--tar -``, so the box that holds
the zip and the machine that reads the records can be two machines:

    scp scripts/mimic_extract.py list.txt box:/tmp/
    ssh box python3 /tmp/mimic_extract.py --zip Z --sums S --list /tmp/list.txt --tar - \\
        | tar -x -C ~/data/mimic-iv-ecg

Standard library only, so it runs on a machine without the project's
environment.  ``--list`` holds one record path per line, as ``record_list.csv``
spells it (``files/p1000/p10000032/s40689238/40689238``).

Usage: python3 scripts/mimic_extract.py --zip Z --sums S --list L (--dest D | --tar T)
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import sys
import tarfile
import zipfile
from pathlib import Path

PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
EXTENSIONS = (".hea", ".dat")


def published_digests(sums: Path) -> dict[str, str]:
    out = {}
    for line in sums.read_text().splitlines():
        digest, _, name = line.partition(" ")
        out[name.strip()] = digest
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--sums", required=True, type=Path)
    parser.add_argument("--list", required=True, help="file of record paths, or - for stdin")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dest", type=Path)
    target.add_argument("--tar", help="tar file to write, or - for stdout")
    args = parser.parse_args()

    if args.list == "-":
        records = [line.strip() for line in sys.stdin if line.strip()]
    else:
        records = [
            line.strip() for line in Path(args.list).read_text().splitlines() if line.strip()
        ]
    digests = published_digests(args.sums)
    mismatched = 0
    with contextlib.ExitStack() as stack:
        tar = None
        if args.tar is not None:
            stream = (
                sys.stdout.buffer if args.tar == "-" else stack.enter_context(open(args.tar, "wb"))
            )
            tar = stack.enter_context(tarfile.open(fileobj=stream, mode="w|"))
        archive = stack.enter_context(zipfile.ZipFile(args.zip))
        for count, record in enumerate(records, 1):
            for extension in EXTENSIONS:
                name = record + extension
                data = archive.read(PREFIX + name)
                if hashlib.sha256(data).hexdigest() != digests.get(name):
                    mismatched += 1
                    print(f"digest mismatch: {name}", file=sys.stderr)
                    continue
                if tar is not None:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                else:
                    path = args.dest / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            if count % 2000 == 0:
                print(f"{count} / {len(records)}", file=sys.stderr, flush=True)
    print(f"{len(records)} records, {mismatched} digest mismatches", file=sys.stderr)
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
