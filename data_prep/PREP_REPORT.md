# LongBench code-completion preprocessing audit

Date: 2026-10-03 UTC. Status: **real public benchmark bytes downloaded, verified, and CPU-tokenized; no model inference, GPU, model weights, rental, or quality result**.

## Immutable sources and licensing

- Dataset: [`zai-org/LongBench@5e628be450b7e67fb7ae6e201bd6d8f7056f7672`](https://huggingface.co/datasets/zai-org/LongBench/commit/5e628be450b7e67fb7ae6e201bd6d8f7056f7672), the official destination of the older THUDM alias. This is LongBench v1, not LongBench v2 or LongBench-E.
- Downloaded static [`data.zip`](https://huggingface.co/datasets/zai-org/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip), 113,932,529 bytes. SHA-256: `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`. Verified against published archive metadata and local bytes. Only `data/lcc.jsonl` and `data/repobench-p.jsonl` were extracted; individual hashes are locked. The deprecated dataset Python loader was not executed.
- Official evaluator/template source: [`THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e`](https://github.com/THUDM/LongBench/commit/2e00731f8d0bff23dc4325161044d0ed8af94c1e). Seven unmodified small source files, the MIT license, and exact URL/hash provenance are saved under `sources/longbench/`.
- Tokenizer only: [`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218), Apache-2.0. Downloaded only `tokenizer.json` (11,422,654 bytes) and `tokenizer_config.json` (9,732 bytes), with immutable revision and SHA-256 pins.

**License caveat:** the LongBench HF card and builder do not declare a dataset license. [LongBench's repository license](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/LICENSE) covers benchmark software; it must not be treated as evidence that every third-party code snippet has the same license. [LCC's upstream CodeBERT repository](https://github.com/microsoft/CodeBERT/tree/master/LongCoder) is MIT, while its HF dataset cards provide no explicit data license. [RepoBench's original branch](https://github.com/Leolty/repobench/tree/archive/v0) supplies [CC-BY-4.0](https://github.com/Leolty/repobench/blob/archive/v0/LICENSE) and expressly warns that individual source-repository licenses may not have been fully verified. The check supports retaining the authors' publicly provided benchmark locally for this requested research; **no redistribution clearance or blanket legal conclusion is asserted**. Raw code, labels, and reversible token IDs are ignored by Git. Resolve individual-source rights before any later release of those materials.

## Data audit and actual split

Each source member contains 500 rows with 500 unique IDs and 500 unique exact contexts. LCC languages: Python 182, Java 160, C# 158. RepoBench-P: Java 264, Python 236. There are no exact context matches between these two task files.

The pilot selects 8 development contexts and 16 heldout contexts per task, 48 total. Both are researcher-created slices of the official **test** split, not an official training/development split. Freeze this manifest before using any model scores; development tuning would invalidate comparisons that present those 8 records as untouched test data.

Context grouping is SHA-256 over the original UTF-8 context, with no whitespace cleanup. Eligible groups are sorted by SHA-256 of the declared seed plus their context hash. The first 8 groups go to development; the next 16 to heldout. All rows sharing an exact context are kept together. Ranking uses no question text, reference, prediction, or full-KV outcome. Model-length feasibility uses token counts of context plus query, not labels.

The predeclared length window is 4,096–32,768 context tokens, with context + query + 64 generation tokens also <=32,768. The lower bound ensures later scoring chunks exist for the transfer diagnostic. No truncation is performed. LCC: 96 eligible, 404 too short, 0 oversized. RepoBench-P: 397 eligible, 97 too short, 6 oversized. All rejection IDs/reasons and 1,000 source-row fingerprints are recorded in the manifest. This deliberate long-context slice changes the benchmark population; do not report its score as the full LongBench score.

Selected LCC contexts span 4,248–22,821 tokens, 182,059 context tokens total. Selected RepoBench-P contexts span 5,765–26,602 tokens, 320,127 total; their query suffixes reach 2,916 tokens. No exact context crosses development/heldout. This does **not** establish repository-level independence, absence of near-duplicates, or freedom from model-training contamination. The original RepoBench maintainers explicitly warn about public-data memorization.

## Leakage boundaries and prompt construction

The official [templates](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json) supply a fixed instruction, code context, then a fixed next-line prompt; RepoBench-P additionally inserts `input` after context. Every LCC `input` is empty; every RepoBench-P `input` is nonempty and represents target-file information. The normalizer rejects any unexpected nonempty LCC input rather than silently dropping it.

- Frozen context: Qwen chat prefix + fixed task instruction + **entire original context**
- Deferred query: RepoBench target-file input (empty for LCC) + fixed next-line prompt + Qwen chat suffix
- Separate labels: original `answers`, keyed by question ID; never included in context/query JSONL

The original source `length` field can include answer length and uses words/characters, not this tokenizer. It is never used for selection or model limits. Fixed 2,048-token region windows use context positions only; they are not claimed to be repository or document boundaries.

All 1,000 source rows have exact text round-trip checks for context and query, with truncation/padding disabled. For every selected example, independently tokenized frozen-context/query segments equal the monolithic rendered chat token sequence. Qwen template canonical SHA-256 is `41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d`; thinking is disabled. Prefix/suffix token IDs are recorded in the manifest.

The original [LongBench `pred.py`](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/pred.py) skips chat wrapping for code tasks. This pilot uses chat wrapping to fit the audited EchoPress runner's author-style pipeline. Keep this prompt-mode difference explicit; full published-score reproduction has not been validated.

Natural reference strings already appear as exact substrings in 89 LCC contexts and 17 RepoBench contexts, plus one RepoBench input. In the selected sample the context counts are 5 and 1, with none in selected inputs. These are post-selection diagnostic counts. Repeated code and the intended cross-file evidence can contain matching text; substring presence is not by itself harness leakage. **No row is filtered or assigned to an arm based on this gold-dependent audit.**

## Canonical metric and verification

Official [`code_sim_score`](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py) selects the first generated line containing no backtick, `#`, or `//`, then applies `fuzzywuzzy.fuzz.ratio`; the [evaluator](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py) takes the maximum over references and averages over examples, reporting percent. Whitespace is not stripped from the selected line. This is not exact match.

The upstream requirements leave fuzzywuzzy unpinned, and installing optional python-Levenshtein can switch its algorithm. This preparation fixes fuzzywuzzy 0.18.0 with the **difflib fallback**, implements identical fallback semantics without external dependencies, and validates it against the installed package. Results must name this backend; equivalence to a historical run with unknown optional packages is not claimed.

Fifteen CPU tests cover deterministic/grouped split behavior, future-question/gold independence of ranking, explicit length exclusion, source-schema and hash failures, separate scorer/query/label contracts, official line filtering, metric parity, complete prediction-ID coverage, and real local artifact integrity. Fixtures are labeled synthetic and establish software behavior only. See `VALIDATION.json` for final verification and byte-identical rebuild checks. No benchmark predictions or accuracy values were produced.


## First-run length presets

The frozen 48-example selection remains the superset. Two separate derived presets filter only these existing IDs by deterministic context/total-token lengths:

- `smoke8k`: up to the first two eligible **development** contexts per task in the original hash-ranked order; both context and context + query + 64 generation tokens must fit 8,192
- `pilot16k`: all originally selected development/heldout examples whose corresponding budgets fit 16,384; split identities remain unchanged

Both include separately hashed context/question/label files and explicit counts and IDs under `manifest.presets`. Every inventory/selected row also has context-length and total-length buckets. The full 32K superset is for later use only after measured peak-memory checks; a model's advertised maximum length does not establish that a GPU can run these instrumented scoring passes. Compare all arms on the same chosen preset and report that preset's denominator; never silently replace the 48-example superset or mix preset denominators.


Observed preset sizes: smoke8k has 3 development contexts (LCC 2, RepoBench-P 1), with a maximum full budget of 6,277 tokens. Only one frozen RepoBench-P development context meets this cap; no heldout context was substituted. pilot16k has 38 contexts: LCC 7 development / 15 heldout and RepoBench-P 6 development / 10 heldout, with a maximum full budget of 16,068 tokens. The superset remains 48.
