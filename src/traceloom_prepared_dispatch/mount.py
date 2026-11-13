"""Explicit, byte-guarded mounting into a dedicated SGLang source checkout."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "contract.json").read_text())
STATE = ".traceloom-prepared-dispatch"
HOOK = ("\nfrom traceloom_prepared_dispatch.policy import install as _install_prepared_dispatch\n"
        "_install_prepared_dispatch(NPUCudaGraphBackend)\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _target(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise RuntimeError("Source file must stay inside the dedicated checkout")
    return path


def _replace(path, data):
    # Preserve mode and atomically replace the complete file.
    temp = path.with_name(path.name + ".traceloom-tmp")
    with temp.open("xb") as output:
        output.write(data)
    os.chmod(temp, path.stat().st_mode)
    os.replace(temp, path)


def mount(source, *, with_batch_copy=False):
    root = Path(source).resolve(strict=True)
    state = root / STATE
    if state.exists():
        raise RuntimeError("Already mounted or an interrupted mount exists; unmount first")
    backend = _target(root, CONTRACT["backend_path"])
    utils = _target(root, CONTRACT["utils_path"])
    old_backend, old_utils = backend.read_bytes(), utils.read_bytes()
    if digest(old_backend) != CONTRACT["backend_sha256"]:
        raise RuntimeError("Unsupported or modified NPU backend source")
    new_utils = old_utils
    if digest(old_utils) != CONTRACT["batch_copy_utils_sha256"]:
        if not with_batch_copy or digest(old_utils) != CONTRACT["original_utils_sha256"]:
            raise RuntimeError("Accepted batch-copy baseline required; use --with-batch-copy on pristine pinned source")
        new_utils = (ROOT / "batch-copy-utils.py.source").read_bytes()
        if digest(new_utils) != CONTRACT["batch_copy_utils_sha256"]:
            raise RuntimeError("Packaged baseline identity mismatch")
    new_backend = old_backend + HOOK
    paths = {CONTRACT["backend_path"]: (old_backend, new_backend),
             CONTRACT["utils_path"]: (old_utils, new_utils)}
    state.mkdir()
    receipt = {"format": 1, "files": {}}
    for index, (relative, (old, new)) in enumerate(paths.items()):
        name = str(index) + ".original"
        (state / name).write_bytes(old)
        receipt["files"][relative] = {"backup": name, "before": digest(old), "after": digest(new)}
    (state / "state.json").write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        for relative, (_, new) in paths.items():
            _replace(_target(root, relative), new)
    except Exception:
        # Keep state/backup evidence even when rollback is needed.
        for relative, (old, _) in paths.items():
            _replace(_target(root, relative), old)
        raise
    return receipt


def unmount(source):
    root = Path(source).resolve(strict=True)
    state = root / STATE
    receipt = json.loads((state / "state.json").read_text())
    expected = {CONTRACT["backend_path"], CONTRACT["utils_path"]}
    if set(receipt["files"]) != expected:
        raise RuntimeError("Unexpected mount receipt")
    restoration = []
    for relative, info in receipt["files"].items():
        backup = _target(root, STATE + "/" + info["backup"])
        data = backup.read_bytes()
        target = _target(root, relative)
        if digest(data) != info["before"] or digest(target.read_bytes()) not in {info["after"], info["before"]}:
            raise RuntimeError("Source changed since mount; preserve local edits before unmounting")
        restoration.append((target, data, backup))
    for target, data, _ in restoration:
        _replace(target, data)
    for _, _, backup in restoration:
        backup.unlink()
    (state / "state.json").unlink()
    state.rmdir()
    return {"restored": sorted(expected)}
