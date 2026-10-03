# Pinned LongBench code-completion data preparation

This is CPU-only data preparation for a natural-data EchoPress regression pilot. It does not load a model, perform inference, rent hardware, run a GPU, or establish a positive research result. Real benchmark bytes and reversible token IDs remain under ignored `data_prep/local/`; only code, source attribution, IDs, hashes, and metadata are intended for Git.

## Reproduce

From the repository root:

```sh
python -m venv data_prep/local/venv
data_prep/local/venv/bin/python -m pip install --no-deps -r data_prep/requirements-tokenizer.txt
data_prep/local/venv/bin/python -m data_prep.prepare_longbench --fetch --prepare
data_prep/local/venv/bin/python -m unittest discover -s data_prep/tests -v
```

The dependency list deliberately omits `python-Levenshtein`; the selected metric backend is the canonical fuzzywuzzy difflib fallback. `tokenizers` uses local files only in preparation; `--fetch` downloads exactly the benchmark archive plus two tokenizer JSON files. No model package is imported and no model weight filename is requested. `--prepare` alone is offline.

- Sources, exact revisions, byte sizes, hashes, and license caveats: `configs/data/longbench_sources.lock.json`
- Pilot selection and length rules: `configs/data/longbench_pilot.json`
- Actual CPU software versions: `configs/data/preprocessing_packages.lock.json`
- Every source row's ID/hash/length eligibility and selected split: `configs/data/longbench_pilot.manifest.json`
- Detailed source and leakage audit: `data_prep/PREP_REPORT.md`
- Unmodified MIT-licensed official evaluation/template sources, license, and URL/hash provenance: `data_prep/sources/longbench/`

## Runtime interface

Each task/split has three files in `data_prep/local/prepared/`:

1. `*.contexts.jsonl`: `context_id`, `source_id`, `context_token_ids`, `regions`
2. `*.questions.jsonl`: `context_id`, `question_id`, `question_token_ids`
3. `*.labels.jsonl`: `context_id`, `question_id`, `dataset`, `reference_texts`, `metric`

Only the first file is allowed into compression/scoring/anchor selection. Load questions only after the context cache has been frozen, and evaluate labels only after generation. Files are separately hashed in the manifest. Code token windows are fixed context-only partitions; they are not extracted repository/document boundaries. Prefix and suffix IDs, plus the exact template hash, are in the manifest for the runner config.

The pilot is a Qwen single-user-chat adaptation of LongBench code prompts. Official LongBench `pred.py` does not add chat wrappers for these tasks. This distinction must appear alongside results; it is not an exact reproduction of published scores.

## Metric interface

Produce a single arm/ratio/split prediction JSONL with exact fields `question_id` and `prediction` (decoded generated text only), then run:

```sh
python -m data_prep.metrics --predictions predictions.jsonl \
  --labels data_prep/local/prepared/lcc.heldout.labels.jsonl \
  --output scores.json
```

Evaluation rejects duplicate, missing, and extra question IDs so failures cannot silently shrink the denominator. It reports LongBench code-similarity (%) per dataset, with per-example scores. Raw exact-match, model answer likelihood, or a different edit-distance implementation is not a substitute for the declared metric. Keep development and heldout scores separate. A 16-context heldout slice is a small pilot, not the full official benchmark.

## Boundaries

No truncation is supported. Contexts below the predeclared 4,096-token floor are ineligible because the mechanism needs later scoring chunks. Contexts above 32,768 tokens, or whose context + query + 64 generation tokens exceed 32,768, are ineligible. Original text round-trips exactly through the pinned tokenizer. The original `length` field counts source-dependent words/characters and includes answers; it is never used to select/split examples.

Exact-context grouping is stronger than row splitting but does not certify repository-level independence or lack of near-duplicates. Public benchmark memorization by a model is also not ruled out. Never use heldout answers or full-KV results to tune an anchor policy.


## Start small

Use the manifest's `presets.smoke8k` first: three development contexts total (two LCC and one RepoBench-P), with full context + query + 64-token generation budgets <=8,192. It has its own files under `data_prep/local/prepared/smoke8k/`. After measured-memory validation, use `presets.pilot16k` for the first development/heldout pilot; its independent files are under `prepared/pilot16k/`. Preserve the original 48-example superset for later 32K testing. Explicit length buckets, sample sizes, IDs, and all file hashes are recorded, so different subsets cannot be confused.
