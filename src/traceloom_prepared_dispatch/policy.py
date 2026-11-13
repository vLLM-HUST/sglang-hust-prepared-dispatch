"""Retain SGLang's executor, stream, task grouping, replay and events."""
import functools
import json
import os
import sys


def enabled():
    return (os.environ.get("TRACELOOM_PREPARED_DISPATCH_ENABLE") == "1" and
            os.environ.get("TRACELOOM_PREPARED_DISPATCH_KILL_SWITCH") != "1")


def _emit(event, **fields):
    if os.environ.get("TRACELOOM_PREPARED_DISPATCH_EVIDENCE") == "1":
        print(json.dumps({"mod": "sglang-prepared-dispatch", "event": event,
                          "pid": os.getpid(), **fields}), file=sys.stderr, flush=True)


def _values(record):
    if len(record.args) != 3:
        raise RuntimeError("Expected three captured FIA positional tensors")
    values = {}
    for index, arg in enumerate(record.op_cache_entry._schema.arguments):
        if index < len(record.args):
            values[arg.name] = record.args[index]
        elif arg.name in record.kwargs:
            values[arg.name] = record.kwargs[arg.name]
        elif arg.has_default_value():
            values[arg.name] = arg.default_value
        else:
            raise RuntimeError(f"Missing captured argument: {arg.name}")
    return values


def prepare_graph(graph, operator, native, *, guard=False, expected_count=28):
    """Prepare every record before mutating any record's dispatch entry."""
    records = graph.graph_dispatch_mode.graph_dispatch_records
    if len(records) != expected_count or any(r.op_cache_entry is not operator for r in records):
        raise RuntimeError("Unsupported graph operation population")
    plans = [native.Plan(_values(record), guard) for record in records]
    original_update = graph.update
    effective_emitted = False

    def update(*args, **kwargs):
        nonlocal effective_emitted
        payload = kwargs.get("cpu_update_input", args[0] if args else None)
        if (not isinstance(payload, (list, tuple)) or len(payload) != 1 or
                not isinstance(payload[0], dict) or set(payload[0]) != {"actual_seq_lengths_kv"}):
            raise RuntimeError("Only one live actual_seq_lengths_kv update is qualified")
        result = original_update(*args, **kwargs)
        if guard:
            expected = list(payload[0]["actual_seq_lengths_kv"])
            for record, plan in zip(records, plans):
                if list(record.kwargs["actual_seq_lengths_kv"]) != expected or plan.last_kv != expected:
                    raise RuntimeError("Live KV metadata was not consumed")
        if not effective_emitted:
            _emit("runtime_effective", records=len(plans))
            effective_emitted = True
        return result

    for record, plan in zip(records, plans):
        record.op_cache_entry = plan
    graph.update = update
    return plans


def wrap_capture(original):
    @functools.wraps(original)
    def capture(self, shape_key, *args, **kwargs):
        result = original(self, shape_key, *args, **kwargs)
        import torch
        from .native import load_native
        native = load_native()
        plans = prepare_graph(self._graphs[shape_key],
            torch.ops.npu.npu_fused_infer_attention_score.out, native,
            guard=os.environ.get("TRACELOOM_PREPARED_DISPATCH_METADATA_GUARD") == "1")
        _emit("graph_prepared", shape_key=str(shape_key), records=len(plans))
        return result
    return capture


def install(backend_class):
    if not enabled():
        return
    if getattr(backend_class.capture_one, "_traceloom_prepared_dispatch", False):
        return
    from importlib.metadata import version
    from .native import CONTRACT, check_runtime
    if version("sglang") != CONTRACT["sglang_version"]:
        raise RuntimeError("Unsupported SGLang distribution")
    check_runtime()
    replacement = wrap_capture(backend_class.capture_one)
    replacement._traceloom_prepared_dispatch = True
    backend_class.capture_one = replacement
    _emit("installed")
