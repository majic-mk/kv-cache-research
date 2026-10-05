# Restore and scope

1. Verify all upload payload hashes in CHECKPOINT_MANIFEST.json.
2. Extract KV_CPU_Topic_Screens_Source_2026-10-05.zip into a new empty directory. Verify every member against SOURCE_SELECTION_MANIFEST.json. The six original directory names are preserved.
3. Treat saved reports and results as historical evidence. The original scripts/protocols are unchanged and their saved result files bind their SHA-256 hashes. Packaging performed syntax/JSON/integrity checks only; it did not repeat the scientific runs.

## Dependencies and safe replay boundaries

- The RoPE script requires NumPy (the original result records 2.3.5), Linux resource limits and its adjacent frozen protocol. It writes `results/` beside itself. Replay only in a fresh copy so original results cannot be replaced.
- Both HIGGS tensor scripts require NumPy and `sources/EDEN2-256.pt`. The 3,238-byte author codebook is deliberately absent from this GitHub source ZIP. Restore only the hash-matched member identified in EVIDENCE_ARCHIVE_REFERENCE.json from the user-owned Library archive, or obtain the exact official commit-pinned file in `sources/grid_receipt.json`. Do not replace it with a guessed grid or unpickle an unverified object. The scripts inspect its ZIP/raw numeric member and assert its file and Git-blob hashes.
- `recover_diagnostic_arrays.py` additionally needs the two original `tensor_results*/arrays.npz` archives. Their exact Library member names, lengths and hashes are recorded. It writes separate diagnostic files and is not necessary to inspect the compact results.
- `audit_groups.py` uses only the Python standard library and its two included small JSON inputs (released head IDs and transcribed geometry). It writes the audit beside itself, so use a fresh copy. The head IDs are an attributed source artifact, not learned model weights or a Dustin configuration.
- The initial HIGGS 6Q/2KV shape was an algebra-only diagnostic; the separately frozen 32Q/4KV extension matches an author-supported source geometry. Neither executed CUDA, native BF16 arithmetic, DMA, offload, or model inference. CPU butterfly order and simulated BF16 rounding are not native-kernel validation.

## Omitted material

Full research arrays and the codebook remain in the existing private Library evidence archive, v1, SHA-256 82eddb675e9c7da8937eff1220a12bf65b0025e8c757656f248ece3fd04c1bd8. This checkpoint does not change its sharing permissions. Third-party source snapshots are generally replaced by report citations and pins; only two tiny source-data JSON inputs needed for the standard-library audit and the codebook receipt are included. Full third-party papers and the user's P&F PDF/extracted full text are excluded. Their analysis and source citation are retained.

The public primary-source references in reports are citations, not downloaded dependency bundles. Availability of a URL does not imply successful future restoration. This is not a complete offline environment, full Git-history sync, self-contained native implementation, new-cohort result or GPU admission. Cost claims, if present in a historical screen, apply to that screen only and do not describe total project spending.
