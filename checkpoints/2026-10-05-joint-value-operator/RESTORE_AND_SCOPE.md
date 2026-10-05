# Restoration and execution boundaries

1. Run `python verify_checkpoint.py` in this checkpoint directory. It verifies checkpoint and ZIP-member bytes using the standard library, without extracting files or executing research.
2. Inspect the compact reports directly. To inspect source, extract the verified ZIP into a new empty directory. Its `repository/` tree retains original repo-relative paths; `historical_restoration/` is explicitly separate.
3. For omitted raw evidence, obtain the user's existing private full archive identified in EVIDENCE_ARCHIVE_REFERENCE.json and verify its full SHA-256. RESTORATION_MANIFEST.json maps each of its 750 exact member names, sizes and hashes. No new sharing access is conferred.
4. External dependencies and weights are not included in either compact source package or full archive. EXTERNAL_RESTORATION_MAP.json records the pinned Qwen/Qwen3-4B-Instruct-2507 revision, full-shard hashes, selected W-segment provenance and runtime package/source pins. These are restoration locators, not proof of a future installation or environment match.

## Frozen execution is not portable unchanged

The recovered model-runtime restoration pins record Python 3.12.14, NumPy 2.2.6 and PyTorch 2.9.1+cpu with the pinned Transformers source. Each archived launcher retains its own interpreter/source bindings; do not infer that every helper used one interchangeable environment. The source ZIP is authored research code, not a self-contained installed runtime. The C kernels require a compatible compiler; model collection additionally needs the exact model and helper/runtime dependencies. No installation or inference was performed for this checkpoint.

Original admission and launchers bind source identities, absolute paths, resource acceptance, original asset identities and one-use consumed claims. They are archival evidence, not authorization to replay. A separately approved replay needs an explicit new path/environment amendment and independent output directory. Do not delete claims, loosen guards, silently rewrite freezes or overwrite results. Do not retune the observed holdout or discard failed controls/layers/queries.

The full archive's SOURCE_PIN_CLOSURE.json and ORIGINAL_SOURCE_INVENTORY.json describe the full archive, not this compact ZIP. RESTORATION_MANIFEST.json distinguishes the two. Child-time manifests retain the original prefixes of logs finalized later; MANIFEST_HISTORY_AND_LIMITATIONS.json explains those differences rather than rewriting them. The earlier Stage A protocol review and attempt_01 preparation runner were separately reconstructed and accepted only after matching their frozen SHA-256.

Only syntax, JSON parsing, saved result/audit agreement, ZIP integrity, exact source hashes and common credential/signed-URL patterns were checked during checkpoint assembly. Previously recorded tests and the real CPU run were not repeated. No CI pass, GPU result, native serving performance or general inference readiness is claimed.
