"""CPU differential check: original record dispatch vs prepared record dispatch."""
from types import SimpleNamespace
import pytest
from traceloom_prepared_dispatch.policy import prepare_graph


class Arg:
    def __init__(self, name):
        self.name = name
    def has_default_value(self):
        return False


class Operator:
    _schema = SimpleNamespace(arguments=[Arg(n) for n in ("query", "key", "value", "scale", "actual_seq_lengths_kv")])
    def __init__(self):
        self.calls = []
    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs["scale"], list(kwargs["actual_seq_lengths_kv"])))


class Plan:
    def __init__(self, values, guard):
        self.values, self.guard, self.last_kv = values, guard, []
    def __call__(self, *args, **kwargs):
        live = list(kwargs["actual_seq_lengths_kv"])
        if self.guard:
            self.last_kv = live
        # The real native source is separately hash-checked against the measured core.
        self.values["operator"](*args, scale=self.values["scale"], actual_seq_lengths_kv=live)


def graph(operator):
    records = [SimpleNamespace(op_cache_entry=operator, args=(object(), object(), object()),
                 kwargs={"scale": 0.125, "actual_seq_lengths_kv": [128]}) for _ in range(28)]
    calls = []
    def update(*, cpu_update_input):
        calls.append(cpu_update_input)
        for record in records:
            record.kwargs.update(cpu_update_input[0])
            record.op_cache_entry(*record.args, **record.kwargs)
        return "original-result"
    return SimpleNamespace(graph_dispatch_mode=SimpleNamespace(graph_dispatch_records=records), update=update), calls


def test_live_lengths_and_original_update_are_preserved():
    operator = Operator()
    g, calls = graph(operator)
    original_args = [r.args for r in g.graph_dispatch_mode.graph_dispatch_records]
    native = SimpleNamespace(Plan=lambda values, guard: Plan({**values, "operator": operator}, guard))
    plans = prepare_graph(g, operator, native, guard=True)
    for lengths in ([129, 129, 129], [149, 155], [160]):
        assert g.update(cpu_update_input=[{"actual_seq_lengths_kv": lengths}]) == "original-result"
        assert all(p.last_kv == lengths for p in plans)
        observed = operator.calls[-28:]
        assert [(a, scale, kv) for a, scale, kv in observed] == [(a, 0.125, lengths) for a in original_args]
    assert len(calls) == 3 and len(operator.calls) == 84


@pytest.mark.parametrize("payload", [None, [], [{"scale": 2}], [{"actual_seq_lengths_kv": [1]}, {"actual_seq_lengths_kv": [2]}]])
def test_unsupported_update_refused_before_original_dispatch(payload):
    operator = Operator(); g, calls = graph(operator)
    native = SimpleNamespace(Plan=lambda v, guard: Plan({**v, "operator": operator}, guard))
    prepare_graph(g, operator, native)
    with pytest.raises(RuntimeError):
        g.update(cpu_update_input=payload)
    assert not calls and not operator.calls


def test_no_partial_record_replacement():
    operator = Operator(); g, _ = graph(operator)
    records = g.graph_dispatch_mode.graph_dispatch_records
    records[-1].op_cache_entry = object()
    with pytest.raises(RuntimeError):
        prepare_graph(g, operator, SimpleNamespace(Plan=Plan))
    assert records[0].op_cache_entry is operator


def test_failed_plan_preparation_leaves_original_records():
    operator = Operator(); g, _ = graph(operator)
    count = 0
    def fail_second(values, guard):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("allocation failed")
        return Plan(values, guard)
    with pytest.raises(RuntimeError):
        prepare_graph(g, operator, SimpleNamespace(Plan=fail_second))
    assert all(r.op_cache_entry is operator for r in g.graph_dispatch_mode.graph_dispatch_records)
