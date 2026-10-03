# Deferred runtime installation and reproducibility

Nothing in this guide has been installed or GPU-tested as part of the CPU stage. No weights/inference/download/rental is triggered by this document.

## Minimal CPU checks now

Python 3.11+ standard library suffices for `run` without `--execute`, config generation, and the unit/mock suite. Run from repository root. No root `pyproject.toml` edits are required: `python -m echo_pilot ...` resolves the package directly.

## Proposed GPU profile

Use Linux x86-64, a single visible NVIDIA CUDA GPU and Python 3.12.14. `requirements-gpu.in` fixes the principal package versions and author Git tree; it is **not a complete transitive dependency/wheel lock**. Versions exist on official package indexes, but installation compatibility is unverified:

- https://pypi.org/project/torch/2.9.1/
- https://pypi.org/project/transformers/5.2.0/
- https://pypi.org/project/numpy/2.2.6/
- https://pypi.org/project/peft/0.20.0/
- https://github.com/ljwljwljwljw/kvpress/commit/39748e9c8301944128eb2e681bfb074e9a2de860

After compute/provisioning authorization, in a fresh virtual environment:

```bash
python -m pip install -r echo_pilot/requirements-gpu.in
python -m pip check
python -m echo_pilot freeze-environment --output packages.lock.json
```

Archive the resolved distribution wheels/sdists or a hash-locked resolver output, the Python distribution/container digest, CUDA-driver version, and the package lock before experiments. Initial dependency resolution can change the transitive package choices, so the proposed `.in` file alone cannot reproduce an already-run experiment. If resolution fails, stop and deliberately revise the profile; do not silently float version pins. A generated `packages.lock.json` records **every installed package exact version**, with no raw credential-bearing install URLs. The runtime checks exact normalized equality with this complete inventory and the installed KVPress PEP-610 commit, and hashes audited algorithm files. A public version string alone is insufficient.

For exact recreation, reinstall the archived pinned distributions with hashes and re-check the environment lock. KVPress's VCS origin must retain its PEP-610 commit metadata; do not substitute an unrecorded source copy. Save the install transcript and platform artifact hashes with the experiment. No credentials or full environment-variable dump belongs in those files.

## Assets and preflight

Stage checkpoint, configuration and tokenizer separately at exact revision `b968826d9c46dd6066d109eabc6255188de91218`. Stage the prepared data using `data_prep/README.md`; no unpinned `datasets.load_dataset` scripts are needed. Any model-gating terms/access requirements must be handled before staging. The runtime uses `local_files_only=True` and `trust_remote_code=False` everywhere.

`CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m echo_pilot preflight --config ...` checks installed package/source pins, CUDA availability, local model config, immutable config revision, tokenizer template/prefix/suffix, known architecture, unscaled RoPE, vocabulary bounds, exact protected budgets and positional capacity. It loads **no model weights** and performs **no inference**. Checkpoint completeness, cache hooks and CUDA numerical behavior still require the subsequent small all-arms smoke.

Do not start the full pilot because a smoke is merely scheduled. First establish that all four arms execute, retain exact requested pair counts, preserve protected positions, answer from independent cache clones and emit complete artifacts. No GPU batch has been run yet.

## Deterministic CUDA startup

Before launching any preflight/inference process with deterministic mode enabled:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

`:16:8` is also accepted. The setting must precede CUDA initialization; setting it late inside an already-running process is insufficient. The runner checks it before importing runtime dependencies and records this specific non-secret setting, then disables TF32 and enables deterministic algorithms. No complete environment dump is saved. This follows the [pinned PyTorch 2.9 deterministic-algorithms documentation](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html). Deterministic mode can still reject an unsupported kernel rather than silently producing nondeterministic results.
