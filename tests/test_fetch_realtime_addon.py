import hashlib
import io
import re
import zipfile

import pytest

from tools import fetch_realtime_addon as fetcher


class Response(io.BytesIO):
    def __init__(self, data, content_range, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


def opener_for(data, *, status=206, wrong_range=False, truncate=False):
    def open_request(request, timeout):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["Range"]).groups())
        body = data[start:end + 1]
        return Response(body[:-1] if truncate else body,
                        "invalid" if wrong_range else f"bytes {start}-{end}/{len(data)}", status)
    return open_request


def test_range_seek_across_blocks_and_eof():
    data = bytes(range(100))
    remote = fetcher.HTTPRangeFile("https://example.com/test", len(data), block_size=8, opener=opener_for(data))
    assert remote.read(11) == data[:11]
    assert remote.seek(-4, 1) == 7
    assert remote.read(40) == data[7:47]
    assert remote.seek(-3, 2) == 97
    assert remote.read() == data[-3:]
    assert remote.read(1) == b""
    remote.seek(200)
    assert remote.read() == b""
    with pytest.raises(ValueError):
        remote.seek(-1)


@pytest.mark.parametrize("kwargs", [{"status": 200}, {"wrong_range": True}, {"truncate": True}])
def test_bad_range_response_fails_closed(kwargs):
    remote = fetcher.HTTPRangeFile("https://example.com/test", 10, opener=opener_for(b"0123456789", **kwargs))
    with pytest.raises(ValueError):
        remote.read(1)


@pytest.mark.parametrize("corrupt_hash", [False, True])
def test_fetch_extracts_only_expected_member_and_verifies_hash(tmp_path, monkeypatch, corrupt_hash):
    payload = b"real-time-content" * 100
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Linux/" + fetcher.PAYLOAD, payload)
        archive.writestr("../../unwanted", b"must never be extracted")
    data = buffer.getvalue()
    cls = fetcher.HTTPRangeFile
    monkeypatch.setattr(fetcher, "HTTPRangeFile", lambda url, size: cls(url, size, block_size=64, opener=opener_for(data)))
    target = tmp_path / "addon.pak"
    expected = {"size": len(payload), "sha256": "0" * 64 if corrupt_hash else hashlib.sha256(payload).hexdigest()}
    if corrupt_hash:
        with pytest.raises(ValueError, match="SHA-256"):
            fetcher.fetch(target, {"url": "https://example.com/test", "size": len(data)}, expected)
        assert not target.exists()
    else:
        fetcher.fetch(target, {"url": "https://example.com/test", "size": len(data)}, expected)
        assert target.read_bytes() == payload
        with pytest.raises(FileExistsError):
            fetcher.fetch(target, {"url": "https://example.com/test", "size": len(data)}, expected)
    assert not list(tmp_path.glob(".rt-safe-download-*"))
    assert not (tmp_path.parent / "unwanted").exists()
