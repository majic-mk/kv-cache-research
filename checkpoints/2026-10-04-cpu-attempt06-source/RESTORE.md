# Restore and inspect this source checkpoint

This checkpoint preserves source and provenance. It is not a self-contained runtime or a full raw-output restoration. The original pilot and the successful single-query diagnostic are both closed. Do not treat historical release flags, deadlines or launch commands as new authorization.

## Source-only inspection

1. Verify `KV_CPU_Engineering_Through_Attempt06_Source_20261004.zip`: 1,947,480 bytes; SHA-256 `9646bd399b959e4dfe92f69979eb73cca15924e99de289900bba6fcaaa447de4`.
2. Extract into a new directory, rejecting absolute paths, parent traversal, duplicate names and symlinks. Verify every selected file against `SOURCE_SELECTION_MANIFEST.json`.
3. The archive root contains `prior_cpu_v9/` and `attempt06/`. Preserve both roots. They hold source/config/tests, immutable research records and selected bounded terminal summaries. Original package manifests remain historical and include files deliberately excluded from this source subset; use the selection manifest to validate this ZIP.
4. The latest authoritative result is `attempt06/READ_ME_FIRST_TERMINAL.md` and `attempt06/CURRENT_TERMINAL_STATUS.json`. Prior v9 status explains the failed original Full answer. Earlier pending/pre-run records retain their original meaning.
5. For an authorized future checkout, carry repository-relative `research/` and selected `outputs/` paths into a fresh tree. Require byte equality for any duplicate; retain differing versions separately instead of overwriting them. Keep package-level status/manifests beside their own source roots. Earlier general project source is archived at `../2026-10-04-source/`.

## Full retained raw evidence in private Library

These identities are private restoration references, not access tokens or signed download URLs. Retrieve the exact pinned version and verify size/SHA-256 before extraction. A later current Library version is not an interchangeable substitute.

- complete_attempt06_terminal_raw: `KV_CPU_TemporalScope_Attempt6_20261004.zip`
  - Library ID `libfile_454225933afc8191b2c5ee4369b72c84`, version 0
  - 32,057,975 bytes; SHA-256 `6f4d45cc977d757603a9ac6a676aa569700d86d55abb57a9500dd9cc711e5086`

- cumulative_checkpoint: `KV_Revision_CPU_PreRun_20261004.zip`
  - Library ID `libfile_df6d73c95c28819181e8c3418427c3ec`, version 9
  - 49,203,670 bytes; SHA-256 `5f51377d4f48af69e0113f0c34b1a67baed9bdd2e339841aa45a2189b66f01e8`

- complete_per_run_raw: `KV_CPU_AuthorBudget_Attempt5_20261004.zip`
  - Library ID `libfile_7e6d9a3442b0819185da082a80ec79e6`, version 0
  - 60,156,346 bytes; SHA-256 `3e91c5f49a10c90675cfdee400018b9bce33db00a31c14031b92acdab8b412da`

- diagnostic_pre_run: `KV_CPU_TemporalScope_Diagnostic_PreRun_20261004.zip`
  - Library ID `libfile_312e7b1cb2648191901af09ed972ccbc`, version 0
  - 106,440 bytes; SHA-256 `cffaad45772a021993959992bd0563f80b37fe3849255f107d83c3fb5ee7d15e`

Attempt06 retains all 586 original runner/guard files and 131 original BF16 logit vectors. Attempt05 raw is an explicit reference dependency; cumulative CPU v9 retains the older source, failures and engineering evidence. Full K/V tensor payloads were never retained; stored K/V shapes/dtypes/byte counts/hashes cannot recreate those missing tensors. The source ZIP is not a substitute for these raw archives.

## External official assets

See `OFFICIAL_ASSET_PINS.json` for all ten base-model files and the trained static-gate file, with exact byte sizes, SHA-256 hashes and immutable official URLs. The source of these hashes is the preserved acquisition/verification receipt; no model payload was downloaded or rehashed by this packaging task.

Base model: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`; 8,060,896,487 total bytes.
- `config.json`: 727 bytes; `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`
- `generation_config.json`: 238 bytes; `835fffe355c9438e7a25be099b3fccaa98350b83451f9fd2d99512e74f1ade48`
- `merges.txt`: 1,671,839 bytes; `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3`
- `model-00001-of-00003.safetensors`: 3,957,900,840 bytes; `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6`
- `model-00002-of-00003.safetensors`: 3,987,450,520 bytes; `0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1`
- `model-00003-of-00003.safetensors`: 99,630,640 bytes; `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`
- `model.safetensors.index.json`: 32,819 bytes; `d6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c`
- `tokenizer.json`: 11,422,654 bytes; `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- `tokenizer_config.json`: 9,377 bytes; `a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3`
- `vocab.json`: 2,776,833 bytes; `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910`

Trained gate: `ngocbh/TrimKV-Qwen3-4B-Instruct-2507` at `89112809f83d05c3058eadcab4104abde5721b81`
- `trimkv_weights.pth`: 94,725,981 bytes; `a43e83cbd2612924a722206f15777f36d1457fa8dadb7dbe5761d6f54c01037a`

Weights, virtual environments and platform credentials are excluded. Reconstruct any approved runtime from the pinned original package/runtime provenance rather than treating current package versions as equivalent. Source extraction and evidence inspection need no model execution. Any new inference study requires its own reviewed endpoints, resource schedule and authorization.
