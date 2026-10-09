#!/usr/bin/env python3
"""Fetch only the RT-SAFE chunk from the pinned, public SimWorld ZIP.

Uses HTTP ranges so users with the matching base need not download the entire
26 GB archive. Downloads from SimWorld's own hosting, without credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = "SimWorld/Content/Paks/pakchunk3002-Linux.pak"


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only HTTP file with one bounded block of cache."""

    def __init__(self, url, size, *, block_size=8 * 1024 * 1024, opener=urllib.request.urlopen):
        self.url, self.size, self.block_size, self.opener = url, size, block_size, opener
        self.position = 0
        self.cache_start = -1
        self.cache = b""
        self.downloaded_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek origin")
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = min(remaining, size if size >= 0 else remaining)
        result = bytearray()
        while size:
            if not self.cache_start <= self.position < self.cache_start + len(self.cache):
                start = self.position // self.block_size * self.block_size
                end = min(start + self.block_size, self.size) - 1
                # Distinct URLs prevent intermediaries from serving a cached,
                # differently ranged response. Content-Range is still checked.
                separator = "&" if "?" in self.url else "?"
                url = f"{self.url}{separator}download=true&range={start}-{end}"
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with self.opener(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                        raise ValueError("Server did not honor the requested HTTP range; no full download attempted")
                    self.cache = response.read(end - start + 2)
                if len(self.cache) != end - start + 1:
                    raise ValueError("Truncated or oversized HTTP range response")
                self.cache_start = start
                self.downloaded_bytes += len(self.cache)
            offset = self.position - self.cache_start
            count = min(size, len(self.cache) - offset)
            result += self.cache[offset:offset + count]
            self.position += count
            size -= count
        return bytes(result)


def fetch(output: Path, metadata: dict, expected: dict) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    remote = HTTPRangeFile(metadata["url"], metadata["size"])
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".rt-safe-download-", delete=False) as target:
        temporary = Path(target.name)
        try:
            digest, size = hashlib.sha256(), 0
            with zipfile.ZipFile(remote) as archive:
                name = "Linux/" + PAYLOAD
                if archive.namelist().count(name) != 1:
                    raise ValueError("Missing or duplicate real-time chunk in upstream archive")
                if archive.getinfo(name).file_size != expected["size"]:
                    raise ValueError("Unexpected uncompressed payload size")
                with archive.open(name) as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
                        if size > expected["size"]:
                            raise ValueError("Oversized payload")
                        target.write(block)
            target.flush()
            if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError("Upstream add-on SHA-256 mismatch")
            temporary.chmod(0o644)
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"size": size, "sha256": digest.hexdigest(), "downloaded_bytes": remote.downloaded_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="New output file; download first, then verify your base before installation")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "validation/realtime-addon-upstream.json").read_text())
    manifest = json.loads((ROOT / "validation/realtime-addon-manifest.json").read_text())
    print(json.dumps(fetch(args.output, provenance["upstream"], manifest["payload"][PAYLOAD]), indent=2))


if __name__ == "__main__":
    main()
