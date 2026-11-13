"""Build and validate the frozen dispatcher without initializing an NPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "contract.json").read_text())
NAME = "traceloom_prepared_dispatch_native"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_directory():
    return Path(os.environ.get("TRACELOOM_PREPARED_DISPATCH_BUILD",
        str(Path.home() / ".cache" / "traceloom-prepared-dispatch" / "0.1.0"))).resolve()


def check_runtime():
    import torch
    import torch_npu
    actual = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "npu_git_version": torch_npu.version.git_version,
              "schema": str(torch.ops.npu.npu_fused_infer_attention_score.out._schema)}
    for key, value in actual.items():
        if value != CONTRACT[key]:
            raise RuntimeError(f"Unsupported {key}: {value}")
    if sha256(ROOT / "typed_plan.cpp") != CONTRACT["source_sha256"]:
        raise RuntimeError("Native source identity mismatch")
    return actual


def build():
    actual = check_runtime()
    import torch_npu
    from torch.utils.cpp_extension import load
    if torch_npu.npu.is_initialized():
        raise RuntimeError("Build in a fresh CPU process before starting the engine")
    directory = build_directory()
    directory.mkdir(parents=True, exist_ok=True)
    npu = Path(torch_npu.__file__).parent
    os.environ.setdefault("MAX_JOBS", "1")
    module = load(name=NAME, sources=[str(ROOT / "typed_plan.cpp")],
        extra_include_paths=[str(npu / "include")],
        extra_ldflags=["-L" + str(npu / "lib"), "-ltorch_npu", "-Wl,-rpath," + str(npu / "lib")],
        extra_cflags=["-O2", "-g0"], build_directory=str(directory), verbose=True)
    if not module.validate_schema() or torch_npu.npu.is_initialized():
        raise RuntimeError("CPU typed-schema validation failed")
    binary = Path(module.__file__).resolve()
    identity = {**actual, "source_sha256": CONTRACT["source_sha256"],
                "binary_name": binary.name, "binary_sha256": sha256(binary),
                "npu_initialized": False}
    (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


@lru_cache(maxsize=1)
def load_native():
    actual = check_runtime()
    directory = build_directory()
    identity = json.loads((directory / "identity.json").read_text())
    for key, value in actual.items():
        if identity.get(key) != value:
            raise RuntimeError("Native build/runtime mismatch; rebuild in this environment")
    binary_name = identity["binary_name"]
    if Path(binary_name).name != binary_name:
        raise RuntimeError("Invalid native binary filename")
    binary = directory / binary_name
    if (sha256(binary) != identity["binary_sha256"] or
            identity.get("source_sha256") != CONTRACT["source_sha256"]):
        raise RuntimeError("Native build identity mismatch")
    spec = importlib.util.spec_from_file_location(NAME, binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.validate_schema():
        raise RuntimeError("Typed schema validation failed")
    return module
