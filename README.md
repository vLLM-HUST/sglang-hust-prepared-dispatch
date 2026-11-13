# SGLang Prepared Dispatch

An opt-in SGLang MOD for Ascend graph-update host dispatch. It prepares fixed
FIA Tensor/scalar arguments once per graph record, then calls the registered
typed C++ dispatcher with live KV lengths. SGLang's worker, update stream,
task grouping, replay and event dependencies remain in charge.

The measured core increased median throughput by 9.35% over the accepted
batch-copy baseline on Qwen3-0.6B BF16, TP1, Ascend910B2/CANN9.1. Four fresh
process pairs were all faster;240 requests/15360 generated tokens had exact
token/top1-ID/logprob agreement. This is a workload-qualified historical result;
the packaged adapter has CPU contract tests and retains the exact native source.

## Install on the pinned stack

SGLang0.5.23 commit `297e54fd105e974cdf401e5b4d20253a22c16f90`,
Torch2.10.0+cpu, TorchNPU2.10.0.post6 git
`38e483ee79ec8a3286517808740a811c0f0ae66c`.
Use a dedicated source checkout and task environment:

```bash
git clone https://github.com/vLLM-HUST/sglang-hust-prepared-dispatch.git
cd sglang-hust-prepared-dispatch
python -m pip install .

git clone https://github.com/sgl-project/sglang.git /path/to/sglang-mod
git -C /path/to/sglang-mod checkout 297e54fd105e974cdf401e5b4d20253a22c16f90
# In the already prepared pinned SGLang/TorchNPU environment:
python -m pip install --no-deps -e /path/to/sglang-mod/python
traceloom-prepared-dispatch mount /path/to/sglang-mod --with-batch-copy
export TRACELOOM_PREPARED_DISPATCH_BUILD=/path/to/task-build
traceloom-prepared-dispatch build
traceloom-prepared-dispatch check
```

Build/check run in a fresh CPU process without loading a model or initializing
an NPU. A C++ compiler, Ninja and the pinned Torch/TorchNPU libraries are needed.
The compiler uses the actual TensorList-out typed signature. Rebuild after
relocation or changing the runtime; there is no portable prebuilt binary.

`--with-batch-copy` explicitly installs the exact accepted copy baseline into
the dedicated source copy. Both performance arms use it. A modified backend
or an unknown utils source is rejected before writing. Mounting alone leaves
prepared dispatch off; the hook is loaded by every SGLang worker, including
spawned workers.

After obtaining a device allocation, launch the normal SGLang command with:

```bash
export TRACELOOM_PREPARED_DISPATCH_ENABLE=1
export TRACELOOM_PREPARED_DISPATCH_EVIDENCE=1
# Run the normal SGLang application from this environment.
# e.g. python -m sglang.launch_server --model-path PATH --device npu --attention-backend ascend --tp 1
```

Installed/graph-prepared/first-runtime-effective JSON receipts include the PID.
For dynamic-metadata qualification also set
`TRACELOOM_PREPARED_DISPATCH_METADATA_GUARD=1`; it checks every native Plan's
consumed KV vector. Disable that diagnostic guard for independent performance
measurements. Only the pinned 28-FIA graph population and a single
`actual_seq_lengths_kv` update are accepted. Other operators, changing fixed
arguments, tensor subclasses, speculative multi-step updates, training, and
other stacks need separate qualification.

## Disable and restore

Stop the engine, unset `TRACELOOM_PREPARED_DISPATCH_ENABLE`, and restart for the
accepted batch-copy baseline. `TRACELOOM_PREPARED_DISPATCH_KILL_SWITCH=1` wins
at startup. To restore the complete source copy, after stopping its workers:

```bash
traceloom-prepared-dispatch unmount /path/to/sglang-mod
python -m pip uninstall sglang-hust-prepared-dispatch
```

Unmount restores both original files byte-for-byte and refuses to overwrite
later local edits. Installation adds no global import hook or vLLM entry point.
This SGLang-specific MOD uses an explicit source mount; it is not activated by
the vLLM Extension Manager.

## Verification and evidence

```bash
python -m pip install '.[test]'
python -m pytest -q
python -m build
```

See [evidence](docs/EVIDENCE.md), [provenance](docs/PROVENANCE.md),
and the packaged [source/runtime contract](src/traceloom_prepared_dispatch/contract.json).
Maintainer: [Shuhao Zhang](https://github.com/ShuhaoZhangTony).
