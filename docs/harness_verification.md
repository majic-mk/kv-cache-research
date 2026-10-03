# Harness verification

Checked on 2026-10-03 in the cloud CPU workspace, Python 3.12.14, Git 2.52.0.

## Passed

- `python -m unittest discover -s tests -t . -v`: 46 CPU tests
- `python -m compileall -q kv_harness tests`: Python compilation
- End-to-end `cpu-fixture` → `validate` → `analyze` CLI flow
- 8 generated fixture units × 3 repeats × 2 identical CPU methods, plus 4 warmup
  rows: all 52 JSONL rows validated; identical output checksums were verified by
  the CLI unit test. The fixture byte-mean metric matched between methods
- Fixed-seed cluster-bootstrap output is stable under input-record reordering
- Missing/duplicate pairs, failed trials, mismatched inputs, mixed evidence,
  malformed JSON/Unicode, invalid numerics and unobserved amortization are rejected
- Context-cluster tests: five questions sharing one cache count as one independent
  cluster; repeats and unequal question counts do not inflate that count
- Source checks reject submodules and assume-unchanged/skip-worktree flags
- Manifest copy isolation, checksum tampering and no-overwrite publication tests
- Read-only audit of the existing cost ledger: one zero-CNY no-purchase entry

## Not performed

No real model inference, GPU execution, baseline reproduction, KV-policy benchmark,
paid API, rental, deployment, or remote Git push was performed by these checks.
Numeric test vectors and CPU-primitive timings are nonexperimental fixtures.
No test validates a scientific performance or physical-memory improvement.

The original records and research/data-preparation artifacts are not modified by
the harness. The package has no model adapter, paid-compute launcher, or remote
publication command. Topic-specific tests are separate from the 46 tests above.
