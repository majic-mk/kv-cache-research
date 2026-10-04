# Model-free source restoration

## Verify the delivered archives

- `KV_Heldout_Compact_Minimal_Closure_20261004.zip`: 650,931 bytes; SHA-256 `49962291184e7f2ff512df75944bc4e3e459a524029de865527da3c36c094321`
- `MODEL_FREE_AUDIT_EVIDENCE_20261004.zip`: 34,295 bytes; SHA-256 `9f63bcb2b616c4d2a0734a1a5ac8fc6f98d69ef2906de2c3e27072432ac4961d`
- `MINIMAL_CLOSURE_MANIFEST.json`: SHA-256 `aac71f752bfd322c799b46b9da37d7d34826c98e200237a90254d108f4d03653`

Extract the source ZIP into a fresh directory. Its entries begin directly with `research/reconstructed/`; there is no outer package directory to strip. Reject absolute paths, parent traversal, duplicate names, symlinks and conflicting pre-existing files. Verify all 82 files, 3,407,075 uncompressed bytes, against `MINIMAL_CLOSURE_MANIFEST.json`. Keep the separate historical audit ZIP in its own directory and verify its 26 entries against `AUDIT_EVIDENCE_MANIFEST.json`.

## Bounded standard-library check

From the restored `research/reconstructed/revision_heldout_compact_contract_20261004/` directory, the documented model-free command is:

    python -B check_model_free.py --output new_check_directory

Choose a new, exclusive output directory for every check. The checker enforces the frozen wall/CPU/address-space/disk constraints; do not relax them. It requires at least 5 GB free disk, but uses tiny fixtures rather than a large stress write. Model weights, a tokenizer, Torch, the full oracle corpus, prior raw vectors and a provider account are not required for this contract suite. Reference files in a directory named `revision_gpu_pilot_20261004` are frozen hash inputs; they are not permission or an instruction to run that pilot.

The earlier successful audit ran all 73 tests on Python 3.12.14/Linux after restoring the complete historical v1 preparation payload plus four references. A later separately authorized one-shot check now also verified this exact minimal-only ZIP: all 73 tests passed, zero files were borrowed or added, and all 82 source hashes matched before and after. The supplemental `MINIMAL_ONLY_CLEAN_TEST_RECEIPT.json` is bound to source ZIP SHA-256 `49962291184e7f2ff512df75944bc4e3e459a524029de865527da3c36c094321`; its log is `minimal_only/check_01/tests.log` in the evidence ZIP. The 1.751-second supervised run sampled a 94,429,184-byte process-tree peak, with 1-GiB memory and 5-GB disk guards and no retry. This supersedes only the earlier static-minimal coverage statement; the old addenda and original selection manifest stay unchanged. It is not full-project CI. See `COMPACT_REPRODUCIBILITY_ADDENDUM.md` and the original receipt/logs for exact tested scope and resource accounting. Historical audit scripts contain their original local paths and should not be run blindly in another layout.

The compact model/backend path remains unimplemented, unqualified and locked. `plan.execute()` fails closed. Passing these tests does not establish device behavior, inference readiness, model quality, compression benefit, runtime installation, a new-cohort outcome or a GPU result.

## Complete preparation history

The full additive preparation packet remains in the user's private Library:

- Library identity `libfile_f58eb350a8e88191bc9f044e60b720d5`, version 2
- Filename `KV_Revision_Heldout_Preparation_20261004.zip`
- 1,352,147 bytes; SHA-256 `e5082fff952bebabdfe7e1eda763bcc1e157e82c6ff479ee9e60a6872027febe`
- Full manifest SHA-256 `88fc16e3be1f77ab0ffeee0ad37af6f38407404e9d255c55243d6391994a27c0`

Retrieve the pinned version and verify its identity rather than accepting a different current version. It preserves all 124 prior v1 entries unchanged, the four exact references, complete preparation history and the audit/addenda. No signed download URL or credential is included here.

## Older CPU inference: unchanged replay is not portable

The [attempt06 source checkpoint](../2026-10-04-cpu-attempt06-source/) and [earlier main-v7 source checkpoint](../2026-10-04-source/) remain byte-identical. `REPRODUCIBILITY_ADDENDUM.md` is an additive correction to their restoration interpretation, not a source patch.

Even with matching external weight hashes, old CPU admission checks bind original resolved paths, inode/device/mtime/size and symlink identities. Historical outputs and the exclusive attempt claim prevent relaunch. The old runtime also assumes the archived Linux/CPython/native-library behavior; cross-machine numerical equivalence was not established. Do not delete claims, overwrite receipts, alter source manifests or weaken identity/resource guards to force a rerun. A future portable execution design would need separate review, fresh provenance/admission and separate execution authority.

The original Full/compressed/GPU pilot stays failed and closed. The separate single-query reformulation result remains a bounded diagnostic success, not a new method or an isolated causal-mechanism result. This package authorizes neither inference nor paid resources.
