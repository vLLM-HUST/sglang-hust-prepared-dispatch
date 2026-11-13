import hashlib
import json
from pathlib import Path
from traceloom_prepared_dispatch.native import ROOT, CONTRACT


def test_measured_native_core_identity():
    assert hashlib.sha256((ROOT / "typed_plan.cpp").read_bytes()).hexdigest() == CONTRACT["source_sha256"]
    assert CONTRACT["source_sha256"] == "c1baa3db99667ddc1aac83bdb228843ff00ed58617681c3b5f074778bc476b8b"


def test_frozen_source_and_baseline_identities():
    fixtures = Path(__file__).parent / "fixtures"
    assert hashlib.sha256((fixtures / "original-backend.py.source").read_bytes()).hexdigest() == CONTRACT["backend_sha256"]
    assert hashlib.sha256((fixtures / "original-utils.py.source").read_bytes()).hexdigest() == CONTRACT["original_utils_sha256"]
    assert hashlib.sha256((ROOT / "batch-copy-utils.py.source").read_bytes()).hexdigest() == CONTRACT["batch_copy_utils_sha256"]
