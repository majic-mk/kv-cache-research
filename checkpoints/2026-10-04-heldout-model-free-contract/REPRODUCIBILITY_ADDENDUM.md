# Clean-workspace reproducibility addendum, 2026-10-04

Applies to source checkpoint commit `fbb3fc0097a2418f6a0f975b4c983eae4d6496e0`, `checkpoints/2026-10-04-cpu-attempt06-source/`. This is an additive restoration clarification; frozen source, protocols, outcomes, and authorization gates are unchanged.

## What was verified

The source ZIP (1,947,480 bytes, SHA-256 `9646bd399b959e4dfe92f69979eb73cca15924e99de289900bba6fcaaa447de4`) was safely extracted in a fresh directory. All 346 selected byte hashes match selection-manifest SHA-256 `6a8adb27d8ab8c0fea491a8be4a485cbd85738cb66f0652d06fcab9c290f283c`. All 78 Python files compile without importing them.

Model-free checks passed on Python 3.12.14:

- Seven original stdlib contract groups, using the preserved **locked pre-release snapshot**, restored to its original repository-relative location in a separate clean tree
- Six limited attempt06 helper groups: protocol/source hashes, all sixteen parent input-ID hashes and diagnostic boundary proof, historical acquisition anchor, four EOS/exhaustion fixtures, and process-tree accounting/refusal

The two test commands together took 0.5375 seconds; the sampled peak including the supervisor was 44,769,280 bytes, with at least 8,161,263,616 disk bytes free. They were bounded by a 60-second outer timeout, an earlier 55-second supervisor stop, a sampled 1-GiB process-tree RSS ceiling, 1-GiB per-process address-space limit, and a 5-GB disk floor. No retries, installs, downloads, Torch imports, model construction, or forwards occurred. This is not a full test-suite, runtime-installation, inference, or cross-hardware numerical-reproduction pass.

## Correct source and test layout

The ZIP has an outer `KV_CPU_Engineering_Through_Attempt06/` directory, containing `prior_cpu_v9/` and `attempt06/`. Keep these historical roots intact. To inspect attempt06 helpers, copy each root's `research/` and `outputs/` subtree into a **third, fresh repository-shaped tree**, requiring equal bytes for duplicates. Package-level manifests/status stay with their original roots. No own-source import or input/config dependency was missing from this assembled model-free path.

The seven-group test belongs to the archived locked version: copy the eight files named by `prior_cpu_v9/research/reconstructed/revision_cpu_file_pager_20261004/reviewed_locked_source/source_manifest.json` to `research/reconstructed/revision_cpu_file_pager_20261004/` in a **separate** clean test tree. Restore the original protocol-v2 JSON, `constructed_variants.json`, and acquisition-start receipt at their unchanged relative paths. The locked source-manifest SHA-256 is `c6029de34fed6ba8a278cc7e59bdc2b0feaa1d5087cc7c9acb9fdc0af366ae1e`. This preserves `CPU_EXECUTION_RELEASED = False`; it does not edit or relock the released snapshot.

Do not run the test from inside `reviewed_locked_source/`: its `HERE.parents[2]` root calculation expects the original layout. The released copy's same test is not a fresh generic suite: its negative fixtures expect the historical false release flag, and its result receipt already exists. Likewise, `check_preparation.py` is an asset/reference audit that ultimately preserves an existing receipt, not a standalone unit test. `run_guard.py`/`verify_tiny.py` construct random tiny models and are outside a no-model audit.

## Declared external dependencies and exact restoration locations

These exclusions are documented source-only boundaries, not missing-file packaging defects:

1. **Attempt05 references.** `outputs/revision_cpu_author_budget_attempt_05/native_full/result.json`, `native_full/calls/call_000.json` through `call_079.json`, and `native_full/call_000_logits.bf16.gz` through `call_079_logits.bf16.gz` are 161 required files absent from the source subset. All 161 paths exist in the hash-verified `KV_CPU_AuthorBudget_Attempt5_20261004.zip`, Library `libfile_7e6d9a3442b0819185da082a80ec79e6`, version 0, 60,156,346 bytes, SHA-256 `3e91c5f49a10c90675cfdee400018b9bce33db00a31c14031b92acdab8b412da`. The source-only prefix check correctly stops at the absent `native_full/result.json`. This audit checked the reference archive's hash and path inventory, not its entire raw-vector payload.

2. **Runtime recipe.** CPU v9 includes only `research/reconstructed/runtime_recovery_20261004/cpu_python.sh`, not its `.venv/bin/python`. The complete reconstruction instructions, 33 hashed wheel pins, provenance, and verifiers are in `../2026-10-04-source/KV_Main_V7_Source_Only_20261004.zip`: 2,980,165 bytes, SHA-256 `12df98838856cfe7e81772e6ee8c63d2c43ed2c53cfec0acd80c064153878825`, under `kv-cache-research/research/reconstructed/runtime_recovery_20261004/`. Read its README and `requirements.cpython312-linux_x86_64.lock`; the latter's SHA-256 is `326b82057537fb2a7c51979fb9ff07f5eb6c687e0eb465328c8a1886506a7422`.

3. **Detailed runtime-verifier inputs.** Its `verify_recovered_runtime.py` also hashes two old logs omitted from that source subset: `research/diagnostics/hybrid_recurrence_cpu/logs/install_torch.log` (3,967 bytes, SHA-256 `dbe908d19ce84dd199e037e2f8fe6049e9eb433b74e7a35407736df76f594714`) and `logs/install_transformers.log` (9,040 bytes, SHA-256 `9ec1be827d8d044bfc42c2a4acaf01be4b2ff3900133ca3683b701e30db6323d`). Both are in the hash-verified main checkpoint, Library `libfile_720dc25a6108819181ce77e9d2fadf63`, version 7, 28,671,376 bytes, SHA-256 `5a25079b1917b6283c9f884551dfd0d329ff05e146972f12304fdcea252c1e1a`.

4. **Runtime download.** The official Transformers source tarball is intentionally external: commit `02d8fb9784e8f14a1251e4c992cd82a5762417c6`, 21,424,438 bytes, SHA-256 `e61a3b05ba4fc95ad588147bcc04723f8ee25687b33141fcd89b255ef0e2c1ad`; exact URL is in runtime `provenance.json`. Restore it to that runtime directory's `downloads/`. `requirements.actual.freeze.txt` contains the original absolute `file:///workspace/...` source URI and is historical evidence, not a portable install command. `installation_commands.sh` deliberately exits 2; it is an audited recipe, not executable setup.

Official model/gate pins and complete attempt06/cumulative/pre-run raw archive identities remain in `RESTORE.md` and `OFFICIAL_ASSET_PINS.json`. Attempt06 still loads and verifies the gate even though its sole Full arm does not use it. Full K/V tensor payloads were never retained; hashes cannot restore them.

## Why unchanged inference is not portable

Even after restoring declared runtime, weight, and raw dependencies, unchanged attempt06 admission requires more than matching content hashes. `actual_common.py:verify_asset_identities()` enforces the original resolved paths, symlink status, inode, device, mtime, and size from `outputs/revision_cpu_model_assets_20261004/browser_materialization_manifest.json` (SHA-256 `2f4cdadda8920b00bdbc55babf296fd1c6229e7b61d3f3a4cbeaa2199916ed4b`). That receipt binds one regular file to the original worktree and nine symlinks to `/opt/codex/downloads/de725a8f-5471-4a85-b76d-b10b5209dc7b/` targets. Copying identical bytes elsewhere cannot satisfy those identities. The frozen `ACTUAL_ATTEMPT_CLAIM.json` and occupied output directory also intentionally prevent relaunch.

The runtime additionally assumes Linux `/proc`, POSIX process groups/resource APIs and `pread`, and libc `malloc_trim`. Its pinned binary recipe is CPython 3.12 on Linux x86-64; the Torch wheel is manylinux_2_28. The archived observation used BF16 CPU math and MKLDNN with four intraop/one interop threads. No specific CPU-ISA portability or cross-machine bitwise match has been established. The 6-GiB run RSS ceiling, 6-GB start/1.5-GB running host-memory reserves, and 5-GB disk reserve remain unchanged.

Therefore this checkpoint supports verified source restoration and bounded model-free contracts. It is not an unchanged, clean-machine inference launcher. Any later portable rerun needs a separately reviewed restoration/launch design that preserves content and protocol checks, records new asset identities and runtime provenance, and receives its own execution authorization. Do not overwrite receipts, remove attempt claims, relax guards, or treat this addendum as authorization to rerun the closed study.
