import hashlib
import json
from pathlib import Path
import pytest
from traceloom_prepared_dispatch.mount import CONTRACT, mount, unmount, STATE, HOOK


@pytest.fixture
def checkout(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    for key, name in [("backend_path", "original-backend.py.source"),
                      ("utils_path", "original-utils.py.source")]:
        target = tmp_path / CONTRACT[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((fixtures / name).read_bytes())
    return tmp_path


def snapshot(root):
    return {key: (root / CONTRACT[key]).read_bytes() for key in ("backend_path", "utils_path")}


def test_mount_baseline_and_exact_restore(checkout):
    before = snapshot(checkout)
    receipt = mount(checkout, with_batch_copy=True)
    assert snapshot(checkout)["backend_path"] == before["backend_path"] + HOOK
    assert hashlib.sha256(snapshot(checkout)["utils_path"]).hexdigest() == CONTRACT["batch_copy_utils_sha256"]
    assert len(receipt["files"]) == 2
    with pytest.raises(RuntimeError):
        mount(checkout, with_batch_copy=True)
    unmount(checkout)
    assert snapshot(checkout) == before
    assert not (checkout / STATE).exists()


def test_prerequisite_is_explicit(checkout):
    before = snapshot(checkout)
    with pytest.raises(RuntimeError, match="baseline required"):
        mount(checkout)
    assert snapshot(checkout) == before
    assert not (checkout / STATE).exists()


def test_modified_host_rejected_before_writes(checkout):
    backend = checkout / CONTRACT["backend_path"]
    backend.write_bytes(backend.read_bytes() + b"\n# local edit\n")
    before = snapshot(checkout)
    with pytest.raises(RuntimeError, match="modified"):
        mount(checkout, with_batch_copy=True)
    assert snapshot(checkout) == before


def test_unmount_preserves_later_edits(checkout):
    mount(checkout, with_batch_copy=True)
    backend = checkout / CONTRACT["backend_path"]
    backend.write_bytes(backend.read_bytes() + b"\n# later edit\n")
    before = snapshot(checkout)
    with pytest.raises(RuntimeError, match="changed since mount"):
        unmount(checkout)
    assert snapshot(checkout) == before


def test_mount_is_inert_without_enable(checkout, monkeypatch):
    from traceloom_prepared_dispatch.policy import install
    monkeypatch.delenv("TRACELOOM_PREPARED_DISPATCH_ENABLE", raising=False)
    class Backend:
        def capture_one(self):
            return 3
    original = Backend.capture_one
    install(Backend)
    assert Backend.capture_one is original
    monkeypatch.setenv("TRACELOOM_PREPARED_DISPATCH_ENABLE", "1")
    monkeypatch.setenv("TRACELOOM_PREPARED_DISPATCH_KILL_SWITCH", "1")
    install(Backend)
    assert Backend.capture_one is original
