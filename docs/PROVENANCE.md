# Provenance

Extracted 2026-11-13 from the completed TraceLoom graph-update investigation.
The native source SHA256 is
`c1baa3db99667ddc1aac83bdb228843ff00ed58617681c3b5f074778bc476b8b`.
This repository builds that exact source with a different extension module name;
the native Plan code and typed call are unchanged.

The Python capture wrapper is adapted from the successful experiment: construct
a Plan per recorded FIA call, replace only op_cache_entry, check the live update
contract, and keep the original graph.update implementation. Experiment output
paths and observer imports have been removed; optional guards check the same
native last_kv field. All records are prepared before any replacement, so a
failed preparation leaves the original population untouched.

The packaged batch-copy utils snapshot is the common baseline, not an additional
gain attributed to dispatch. Its pristine source fixture and backend fixture
come from SGLang commit 297e54fd105e974cdf401e5b4d20253a22c16f90, Apache2.0.
The mount records before/after hashes and preserves original bytes for rollback.
No research capsule, profiler database, model weight, anonymous review alias,
or manuscript material is redistributed.
