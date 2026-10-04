# Compact-contract clean-tree verification, 2026-10-04

**All 73 standard-library tests pass after restoring four declared reference files.** The single clean-tree run used archived bytes only. It did not import Torch, install software, create a model, call a provider, or change frozen source or guards. The compact source contract is runnable as a model-free test suite; its model/backend execution remains deliberately locked.

## Verified artifact and result

Input: `KV_Revision_Heldout_Preparation_20261004.zip`, Library `libfile_f58eb350a8e88191bc9f044e60b720d5`, version 1; 1,273,736 bytes; SHA-256 `75dc11b605c2f324fe44abdcad254b1a8d111bcc778a3558aeff5db8f72421f5`.

All 124 ZIP entries were safely extracted, and every one of the 123 payload rows in `PACKAGE_MANIFEST_VERSION_1.json` was verified. All 65 compact-protocol dependency files were hash-checked after restoration. The 11 current Python modules' import graph is stdlib plus local contract modules. The complete v1 payload was rechecked unchanged after the tests.

The documented command was run once from `research/reconstructed/revision_heldout_compact_contract_20261004/`:

    python -B check_model_free.py --output <new exclusive directory>

Result: `Ran 73 tests in 1.709s`, `OK`. The preserved runner reports 1.8694 seconds and 68,366,336-byte child peak RSS; the independent supervisor sampled 88,092,672 bytes for its entire process tree, with at least 8,155,750,400 free disk bytes. The command had a 60-second outer timeout, an earlier 53-second supervisor stop, 1-GiB address-space and sampled process-tree RSS limits, and a 5-GB disk floor. No retry occurred. Together with the old-checkpoint stdlib checks, supervised test-command wall time was 2.445 seconds.

## Exact restoration mapping

Strip only the package's outer `KV_Revision_Heldout_Preparation_20261004/` directory. Preserve the `research/reconstructed/` relative layout in a fresh checkout. Keep historical payloads byte-identical and reject conflicting duplicates.

From preparation v1, restore:

- `revision_heldout_compact_contract_20261004/`: all 39 frozen files for complete history. The current test implementation consists of the 16 files named by `source_manifest.json` plus that manifest. Source-manifest SHA-256: `47ac8ddc24062f85ad09924a51a242b74f920b7bab63f250560a3af3c40cc197`; complete artifact-manifest SHA-256: `04505037d7a0e7ba01456ed096260ebccadb510034ae0246ed9ce037de70982b`
- `revision_serving_holdout_v1_20261004/`: all 55 files. Its artifact-manifest SHA-256 is `885d09b68d729dc2931d0f62e8359a84b6c868da1d3fd6d01a6d5f6546ba6266`; its eight-input `constructed_inputs.json` is 1,489,179 bytes, SHA-256 `b7c95c4d2cf6429e5306702e571a593d5ab6e6e82708bbd743208d368996a753`
- Six sibling documents: `revision_heldout_common_budget_clarification_20261004.{md,freeze.json}`, `revision_heldout_execution_feasibility_20261004.{md,json,freeze.json}`, and `revision_heldout_success_branch_recommendation_20261004.md`

The remaining four files are declared archived dependencies, totaling only **103,807 bytes**. They must be present under the same `research/reconstructed/` root:

1. `revision_gpu_pilot_20261004/protocol.py`: 26,210 bytes; SHA-256 `5e769672e24069abcf80924e16c274a5d296e5b73cb34d721f502241bab24e87`
2. `revision_gpu_pilot_20261004/pilot_config.json`: 13,137 bytes; SHA-256 `1b7c177bea9a11ef288f7f278fc060de695fc3cfc4cf74e6b474acf095ede55a`
3. `revision_baseline_contracts_20261004/phase2_static_gate_verification/tensor_manifest.json`: 63,510 bytes; SHA-256 `0738cdb8fe07d5676c2310000ab05b08be459b3e6768364c5e7af6a831b18ac7`
4. `revision_temporal_scope_diagnostic_20261004/decode_contract.py`: 950 bytes; SHA-256 `6e8cdf3864c939a87597763daa0d82b16c3aaf553cc2feeaf3137517acfd662c`

Files 1–3 come from `KV_Revision_V2_Preparation_20261004.zip`, Library `libfile_eb47ac747be081918af5c0edb54c6808`, version 3; 2,266,655 bytes; SHA-256 `772b551c4e0cfa44c4e255ae9c91e2362d2331cbf9360793994ffa7adc94fe07`. Their archive prefix is `KV_Revision_V2_Preparation_20261004/research/reconstructed/`.

File 4 comes from `KV_CPU_TemporalScope_Attempt6_20261004.zip`, Library `libfile_454225933afc8191b2c5ee4369b72c84`, version 0; 32,057,975 bytes; SHA-256 `6f4d45cc977d757603a9ac6a676aa569700d86d55abb57a9500dd9cc711e5086`. Its archive prefix is `KV_CPU_TemporalScope_Attempt6_20261004/research/reconstructed/`. An already verified copy of these same 950 bytes also exists in the CPU GitHub source checkpoint.

`COMPACT_MINIMAL_RESTORATION_MAP.json` supplies the exact per-file sizes, hashes and original artifact/member locators for the entire mapping. The statically determined minimal current test closure is 82 files: 17 current contract files, 55 input-directory files, six sibling documents and four references. The actual successful audit retained the complete historical package plus the four restored references.

## Documentation and packaging clarification

Preparation v1 alone does not contain every file checked by `test_source_dependencies_unchanged`. Its eight frozen inputs and compact implementation are included, while four small hash-bound references remain external. This is a declared dependency boundary, not a source defect or reason to alter the tests. A later additive packaging version can include those four unchanged bytes and this explicit mapping to remove the large-archive download requirement for model-free tests. No packaging publication was performed by this audit.

No full oracle, tokenizer, environment, model weights, or prior raw-logit payload is required to execute these 73 tests. Source-selection/rerendering and a real backend have separate dependencies and authorization requirements. Passing the suite does not establish inference readiness, device behavior, model quality, scientific benefit, provider readiness or cost settlement.
