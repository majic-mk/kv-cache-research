# Joint-value local-operator CPU checkpoint, 2026-10-05

The fixed-population CPU experiment completed validly. **Its frozen usefulness gate FAILED.** This additive private checkpoint preserves the exact proposal, authored implementation, protocols, tests, compact evidence and final audit; it does not admit a successful research method.

## Registered result

Metric: unweighted mean of six layerwise holdout SSE / same-layer nearest SSE; lower is better. These are correlated isolated-layer observations from one teacher-forced passage, not independent task-quality samples.

| Arm | Mean nearest-normalized holdout SSE |
|---|---:|
| nearest | 1.000000000 |
| headwise CD | 0.828324749 |
| group-joint CD | 0.821914459 |
| full-joint CD | 0.808808818 |
| guarded token GPTQ | 0.966594180 |
| projected-block GPTQ | 0.804669366 |

Full-joint improves 19.12% versus nearest, but only 2.36% versus headwise and is 0.51% worse than projected-block. It wins against headwise in 2/6 layers (required: 4/6). Its worst query is 1.541438398 times nearest SSE, exceeding the 1.25 cap. The required 10% advantage over every strong control failed. All 36 encodings completed; no arm, layer or query was dropped. Encoder row caps passed; the failed cap is held-out operator quality.

See [actual report](ACTUAL_RESULT_REPORT.md), [independent final audit](FINAL_INDEPENDENT_RESULT_AUDIT.json), and [compact results](COMPACT_RESULT_SUMMARY.json). The exact proposal is closed at its frozen stopping gates; no retuning or rerun followed.

## Source and restoration

- [Source ZIP](Joint_Value_Operator_Source_2026-10-05.zip): verbatim authored sources, frozen protocol/claim files, tests and small evidence, including separately labeled exact-hash historical restoration
- [Source selection manifest](SOURCE_SELECTION_MANIFEST.json): exact member bytes and SHA-256
- [Restoration manifest](RESTORATION_MANIFEST.json): all 750 full-evidence archive members, distinguishing this compact ZIP from Library-only payloads
- [Private evidence references](EVIDENCE_ARCHIVE_REFERENCE.json): full archive v0 and comparison figures v1
- [External restoration pins](EXTERNAL_RESTORATION_MAP.json), [manifest history](MANIFEST_HISTORY_AND_LIMITATIONS.json), and [restore limitations](RESTORE_AND_SCOPE.md)

Full raw captures, error vectors, code arrays, diagnostic paths and large forecasts stay in the private Library archive. Weights, installed dependencies, full third-party papers, credentials and signed URLs are excluded. Historical source-preparation-only reports remain unchanged and are superseded by the final actual report above.

This is FP64 isolated-layer operator error on captured BF16 Q/K/V and signed W_o. There is no generated-answer/QA evaluation, autoregressive propagation test, new GPU/native-kernel speed result, serving-memory measurement or paper-ready novelty claim. Hypothetical packed bytes are not measured native memory. Packaging integrity is not a CI pass or portable full inference.
