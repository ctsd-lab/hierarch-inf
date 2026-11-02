"""Downloader tests use synthetic bytes, never the observed GAMMA samples."""
import hashlib
import io
import zipfile

import pytest

from examples.queueing import gamma_study as study


def manifest(data=b"synthetic\n"):
    return {"files": [{"path": "run/sample.csv", "uncompressed_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "source_url": "https://www.kaggle.com/api/v1/datasets/download/gagansomashekar/microservices-bottleneck-detection-dataset/sample.csv?datasetVersionNumber=1"}]}


def test_reject_path_escape(tmp_path):
    for path in ("../escape", "/etc/passwd", "."):
        with pytest.raises(ValueError):
            study.safe_path(tmp_path, path)


@pytest.mark.parametrize("zipped", [False, True])
def test_download_and_reuse(tmp_path, monkeypatch, zipped):
    data = b"synthetic\n"
    payload = data
    if zipped:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("sample.csv", data)
        payload = buffer.getvalue()
    calls = []
    def request(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(payload)
    monkeypatch.setattr(study, "urlopen", request)
    study.download_files(manifest(), tmp_path, max_total_bytes=100)
    study.download_files(manifest(), tmp_path, max_total_bytes=100)
    assert len(calls) == 1
    assert (tmp_path / "run/sample.csv").read_bytes() == data


def test_download_rejects_budget_and_bad_source(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        study.download_files(manifest(), tmp_path, max_total_bytes=1)
    bad = manifest()
    bad["files"][0]["source_url"] = "https://other.example/sample.csv"
    with pytest.raises(ValueError, match="official"):
        study.download_files(bad, tmp_path, max_total_bytes=100)


def test_download_bad_content_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="checksum"):
        study.download_files(manifest(), tmp_path, max_total_bytes=100)
    assert not (tmp_path / "run/sample.csv").exists()


def test_prepare_checks_provenance_before_reading(tmp_path):
    source = tmp_path / "run"
    source.mkdir()
    (source / "sample.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        study.prepare(manifest(), tmp_path, tmp_path / "out", guard_seconds=5, window_seconds=10)


def test_prepare_rejects_unlisted_consumed_input(tmp_path):
    m = {"files": [], "runs": [{"directory": "run", "request_type": "compose", "graph_id": 1}]}
    with pytest.raises(ValueError, match="absent from manifest"):
        study.prepare(m, tmp_path, tmp_path / "out", guard_seconds=5, window_seconds=10)
