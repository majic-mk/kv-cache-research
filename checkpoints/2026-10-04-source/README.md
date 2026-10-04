# Source checkpoint — 2026-10-04

This is an additive archival delivery to the private `majic-mk/kv-cache-research` repository. The archives preserve later work without rewriting the earlier repository files or Git history. They are **not** a fully expanded source-tree sync or a complete raw-science archive.

## Contents

- `KV_Main_V7_Source_Only_20261004.zip`: 515 source, documentation, configuration, test, dependency and reference-metadata files selected from the verified main scientific checkpoint v7 (source commit `a60fac3abe9cba48105092d3e7dd792a241b3313`). The selection manifest records every included file's SHA-256 and all exclusions. Historical status documents inside this archive describe their original checkpoints.
- `KV_Revision_V2_Preparation_20261004.zip`: the later additive revision-v2 preparation checkpoint, Library version 2. It preserves constructed input-feasibility probes, baseline contracts/source records, static-gate verification metadata, and the completed tiny native reference-correctness packet. Its internal manifests retain the prior versions and failed attempts.
- `CHECKPOINT_MANIFEST.json`: archive byte counts, SHA-256 hashes, provenance and scope.
- `MAIN_V7_SOURCE_SELECTION_MANIFEST.json`: the exact main-v7 source subset and exclusions.

## Scope and scientific limits

The newer native packet records eight passing finite correctness groups on a **24,832-parameter locally random Qwen3 model with synthetic gates**. It does not establish pretrained 4B execution, a result on the actual revision-v2 inputs, a measured revision/compression interaction, GPU performance, or a validated new method. The unfinished revision pilot review is excluded. Research may continue after this frozen checkpoint.

Model weights, environments, the main Git bundle, rendered artifacts and raw experiment tensors/logits are not included in this GitHub delivery. The large raw-science archives remain in the user's separate Library collection. Some small, bounded verification evidence remains in the revision preparation archive; this does not make the delivery a complete experiment restoration package.

## Verify and inspect

Verify each downloaded ZIP's SHA-256 against `CHECKPOINT_MANIFEST.json`, then extract each archive into a **separate new directory**. Do not blindly overlay their historical snapshots on an active worktree. Read the revision archive's `README_ARCHIVE.md` and `PACKAGE_MANIFEST_VERSION_2.json` for its dependency and restoration boundaries. Missing external dependencies and raw data must be restored from the referenced original artifacts before attempting any relevant rerun.

This delivery only packages and preserves existing work. No tests, models, GPU jobs, rentals, or paid API calls were run for this GitHub upload.
