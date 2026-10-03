"""LongBench code-similarity with an explicitly pinned difflib backend.

Official source: THUDM/LongBench@2e00731f8d0bff23dc4325161044d0ed8af94c1e,
LongBench/metrics.py code_sim_score and eval.py scorer. The official unpinned
fuzzywuzzy dependency can switch algorithms when python-Levenshtein is present.
This implementation matches fuzzywuzzy==0.18.0 with its stdlib difflib fallback;
it is NOT exact match, normalized Levenshtein distance, or a syntax-aware score.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path

METRIC = 'longbench-code-sim-fuzzywuzzy-0.18.0-difflib'


def first_code_line(prediction: str) -> str:
    if not isinstance(prediction, str):
        raise ValueError('Prediction must be text')
    for line in prediction.lstrip('\n').split('\n'):
        if '`' not in line and '#' not in line and '//' not in line:
            return line
    return ''


def fuzzy_ratio_difflib(left: str, right: str) -> int:
    # Replicates fuzzywuzzy==0.18.0 fuzz.ratio with SequenceMatcher=difflib.
    if left == right:
        return 100
    if not left or not right:
        return 0
    return int(round(100 * SequenceMatcher(None, left, right).ratio()))


def code_sim_score(prediction: str, references: list[str]) -> float:
    if not isinstance(references, list) or not references or not all(isinstance(r, str) and r for r in references):
        raise ValueError('References must be a nonempty list of nonempty text')
    line = first_code_line(prediction)
    return max(fuzzy_ratio_difflib(line, reference) for reference in references) / 100


def evaluate(predictions: list[dict], labels: list[dict]) -> dict:
    """Require full question-ID coverage; failed/missing rows cannot shrink denominator.

    Input predictions is a single arm/ratio/split: [{question_id, prediction}].
    Keep separate runs separate. This function does not expose labels to scoring.
    """
    preds, gold = {}, {}
    for row in predictions:
        if set(row) != {'question_id', 'prediction'} or not isinstance(row['prediction'], str):
            raise ValueError('Expected exactly question_id and prediction text')
        if row['question_id'] in preds:
            raise ValueError('Duplicate prediction question_id')
        preds[row['question_id']] = row['prediction']
    for row in labels:
        if row['question_id'] in gold:
            raise ValueError('Duplicate label question_id')
        if row['metric'] != METRIC:
            raise ValueError('Unexpected label metric')
        gold[row['question_id']] = row
    if not gold or set(preds) != set(gold):
        raise ValueError('Predictions must match every label exactly; missing/extra IDs are errors')
    scores = defaultdict(list)
    per_example = []
    for qid in sorted(gold):
        row = gold[qid]
        score = code_sim_score(preds[qid], row['reference_texts'])
        scores[row['dataset']].append(score)
        per_example.append({'question_id': qid, 'dataset': row['dataset'], 'score': score})
    return {'metric': METRIC, 'question_count': len(gold),
            'scores_percent': {task: round(100 * sum(values) / len(values), 2) for task, values in sorted(scores.items())},
            'per_example': per_example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
    result = evaluate(load(args.predictions), load(args.labels))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['scores_percent']))


if __name__ == '__main__':
    main()
