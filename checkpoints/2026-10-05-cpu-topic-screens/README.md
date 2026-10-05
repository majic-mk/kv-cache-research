# CPU topic screens, 2026-10-05

Private, additive source checkpoint. All previous repository files and checkpoints are to remain byte-identical. This is a preservation package, not a newly admitted method or a performance release.

## Current scientific status

- Fixed RoPE U8 fusion failed the frozen uniform phase gate: native maximum 0.1053186445 versus threshold 0.05; YaRN-normalized maximum 0.1053210524. The secondary quantization stage did not run.
- HIGGS/EDEN transformed-attention FP64 algebra passed on fixed artificial arrays, including the separately frozen 32Q/4KV geometry. That is source geometry, not execution of the native kernel. Its output differs by 0.054649171 from the simulated per-row BF16 staging contract (scaled difference 0.00333936).
- Existing arbitrary-pair-table and GQA kernels are strong baselines. The surviving HIGGS integration is a bounded engineering hypothesis only; algebra and removal of global staging do not establish paper-ready novelty or speedup.
- Probe-and-Fetch already covers accepted speculation plus CPU fetching and sparse block verification under an approximate target. The broad combination is closed. The later union-horizon and Dustin exact-SRH screens admitted no new topic.
- The released CompressKV top4-head surrogate touches 67 of 112 layer/KV-group pairs: 536 MiB exact keys versus a 112 MiB raw 2-bit all-head index. This is source-based payload arithmetic, not Dustin reproduction or measured system memory.

## Delivered files

- `KV_CPU_Topic_Screens_Source_2026-10-05.zip`: unchanged authored scripts, frozen protocols, compact saved results and authored reports
- `SOURCE_SELECTION_MANIFEST.json`: exact per-member bytes and SHA-256 hashes
- `EVIDENCE_ARCHIVE_REFERENCE.json`: private Library v1 identity and checksums for omitted raw arrays/codebook
- `EARLIER_RESOURCE_SCREEN_SUMMARY.md`: selective non-security resource/scheduling findings
- `RESTORE_AND_SCOPE.md`: dependency and replay limitations
- `CHECKPOINT_MANIFEST.json`: upload payload hashes and declared scope

No model weights, installed dependencies, wheels, large arrays, third-party full papers, uploaded P&F PDF/full text, credentials or signed URLs are included. No security-candidate branch is included. No GPU/native-kernel/model-quality validation, CI pass, novel-method admission, or whole-project zero-fee claim is made.

The old reports are chronological evidence. In particular, the author-limitations report's then-unresolved P&F access question is superseded by `oasis_probe_fetch_comparison_20261005/FINAL_COMPARISON.md`, followed by the union-horizon and Dustin screens. No frozen report was silently rewritten.
