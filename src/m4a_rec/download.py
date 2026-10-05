"""Download the needed Music4All-Onion files from Zenodo, with resume and md5 check.

Usage: python -m m4a_rec.download
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

from .config import load_config

CHUNK = 1 << 20  # 1 MiB


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Stream url to dest, resuming a partial .part file if one exists."""
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:  # server ignored the range; start over
            have = 0
        total = have + int(resp.headers.get("Content-Length", 0))
        with part.open("ab" if have else "wb") as out:
            done = have
            while block := resp.read(CHUNK):
                out.write(block)
                done += len(block)
                if total:
                    print(f"\r  {dest.name}: {done / total:6.1%} of {total / 1e9:.2f} GB", end="")
    print()
    part.rename(dest)


def main() -> int:
    cfg = load_config()
    raw: Path = cfg["paths"]["raw"]
    base = f"https://zenodo.org/records/{cfg['zenodo_record']}/files"
    failed = []
    for name, md5 in cfg["files"].items():
        dest = raw / name
        if dest.exists() and md5sum(dest) == md5:
            print(f"ok       {name}")
            continue
        if dest.exists():
            print(f"bad md5  {name}; downloading again")
            dest.unlink()
        fetch(f"{base}/{name}?download=1", dest)
        if md5sum(dest) != md5:
            failed.append(name)
            print(f"FAILED   {name}: md5 mismatch after download")
        else:
            print(f"ok       {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
