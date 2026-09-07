from pathlib import Path

from juniper_encoder.artifacts import make_sha256sums, package_descriptor, verify_sha256sums


def test_content_addressed_descriptor_and_checksums(tmp_path: Path):
    payload_dir = tmp_path / "candidate"
    payload_dir.mkdir()
    (payload_dir / "payload.txt").write_text("immutable\n", encoding="utf-8")
    descriptor = package_descriptor(payload_dir / "candidate.json", {"source_sha": "abc", "status": "NOT_RELEASED"})
    assert len(descriptor["candidate_id"]) == 64
    make_sha256sums(payload_dir, payload_dir / "SHA256SUMS")
    assert verify_sha256sums(payload_dir, payload_dir / "SHA256SUMS")["status"] == "PASS"
