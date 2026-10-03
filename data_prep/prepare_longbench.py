"""Prepare pinned LongBench code tasks without datasets scripts, weights, or inference.

Raw benchmark data and all reversible tokenized data stay in ignored local/.
The committed manifest contains only IDs, hashes, counts, and provenance.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'configs/data/longbench_sources.lock.json'
SETTINGS = ROOT / 'configs/data/longbench_pilot.json'
LOCAL = ROOT / 'data_prep/local'
TASK_PREFIX = 'Please complete the code given below. \n'
TASK_SUFFIX = 'Next line of code:\n'
RAW_KEYS = {'input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id'}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value: bytes):
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value):
    return digest(canonical_bytes(value))


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check_file(path, spec):
    if path.stat().st_size != spec['bytes'] or file_hash(path) != spec['sha256']:
        raise ValueError(f'Pinned byte count or SHA-256 mismatch: {path}')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        for row in rows:
            f.write(canonical_bytes(row) + b'\n')
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': file_hash(path), 'rows': len(rows)}


def download(url, path, spec):
    """Only immutable, allowlisted benchmark/tokenizer URLs are supplied by lock."""
    if path.exists():
        check_file(path, spec)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
        count = 0
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > spec['bytes']:
                raise ValueError('Download exceeds pinned byte count')
            out.write(block)
    check_file(temporary, spec)
    temporary.replace(path)


def fetch(lock, local=LOCAL):
    ds = lock['dataset']
    archive = local / 'data.zip'
    download(ds['archive']['url'], archive, ds['archive'])
    # Read only named members, never extractall or execute the deprecated loader.
    with zipfile.ZipFile(archive) as z:
        for name, spec in ds['members'].items():
            info = z.getinfo(spec['path'])
            if info.file_size != spec['bytes']:
                raise ValueError('Unexpected uncompressed member size')
            data = z.read(info)
            if digest(data) != spec['sha256']:
                raise ValueError('Unexpected member hash')
            (local / (name + '.jsonl')).write_bytes(data)
    tok = lock['tokenizer']
    for name, spec in tok['files'].items():
        url = f"https://huggingface.co/{tok['id']}/resolve/{tok['revision']}/{name}"
        download(url, local / 'tokenizer' / name, spec)


def validate_record(row, dataset):
    if not isinstance(row, dict) or set(row) != RAW_KEYS:
        raise ValueError('Unexpected source schema')
    for key in ('context', 'input', '_id', 'language', 'dataset'):
        if not isinstance(row[key], str):
            raise ValueError(f'{key} must be text')
    if row['dataset'] != dataset or not row['context'] or not row['_id']:
        raise ValueError('Invalid dataset, empty context, or empty source id')
    if not isinstance(row['answers'], list) or not row['answers'] or not all(isinstance(a, str) and a for a in row['answers']):
        raise ValueError('Answers must be a nonempty list of nonempty strings')
    if dataset == 'lcc' and row['input']:
        raise ValueError('Official LCC template ignores input; reject unexpected nonempty input')
    if type(row['length']) is not int or row['length'] < 0:
        raise ValueError('Invalid source length')


def read_records(path, dataset):
    rows = []
    ids = set()
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            validate_record(row, dataset)
            if row['_id'] in ids:
                raise ValueError(f'Duplicate source id in {dataset}')
            ids.add(row['_id'])
            row = dict(row)
            # Internal provenance only; never passed wholesale into context scorer.
            row['_source_line'] = line_number
            rows.append(row)
    return rows


def split_groups(records, settings, eligible):
    """Gold/query-blind ranking and grouping; eligibility is length feasibility only."""
    groups = defaultdict(list)
    for row in records:
        groups[digest(row['context'].encode('utf-8'))].append(row)
    candidates = [key for key, rows in groups.items() if all(eligible[r['_id']] for r in rows)]
    candidates.sort(key=lambda key: (digest((settings['seed'] + '\0' + key).encode()), key))
    dev = settings['development_contexts_per_dataset']
    heldout = settings['heldout_contexts_per_dataset']
    if len(candidates) < dev + heldout:
        raise ValueError('Too few eligible context groups for predeclared split')
    out = []
    for rank, key in enumerate(candidates[:dev + heldout]):
        split = 'development' if rank < dev else 'heldout'
        out.extend((split, key, row) for row in sorted(groups[key], key=lambda r: r['_id']))
    return out


def render_chat(template, content):
    from jinja2 import Environment
    return Environment().from_string(template).render(messages=[{'role': 'user', 'content': content}],
                    tools=None, add_generation_prompt=True, enable_thinking=False)


def load_tokenizer(lock, local):
    from tokenizers import Tokenizer
    if importlib.metadata.version('tokenizers') != '0.21.4':
        raise ValueError('Expected pinned tokenizers==0.21.4')
    for name, spec in lock['tokenizer']['files'].items():
        check_file(local / 'tokenizer' / name, spec)
    tokenizer = Tokenizer.from_file(str(local / 'tokenizer/tokenizer.json'))
    # Explicitly turn off any tokenizer-level hidden length modifications.
    tokenizer.no_truncation()
    tokenizer.no_padding()
    template = json.loads((local / 'tokenizer/tokenizer_config.json').read_text())['chat_template']
    sentinel = 'UNIQUE_CONTEXT_SENTINEL_923bd73f'
    rendered = render_chat(template, sentinel)
    if rendered.count(sentinel) != 1:
        raise ValueError('Unsupported chat template transformation')
    prefix, suffix = rendered.split(sentinel)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
    return tokenizer, template, prefix, suffix, encode


def make_example(row, split, ctx_hash, token_data, prefix_ids, settings):
    ctx_ids, question_ids, _ = token_data
    cid = 'sha256:' + ctx_hash
    qid = row['dataset'] + ':' + row['_id']
    # Fixed context-only token windows. No claims that these are document boundaries.
    regions = [{'region_id': f'window-{i:04d}', 'label': 'fixed_code_window', 'start': start,
                'end': min(start + settings['region_tokens'], len(ctx_ids))}
               for i, start in enumerate(range(len(prefix_ids), len(ctx_ids), settings['region_tokens']))]
    context = {'context_id': cid, 'source_id': row['dataset'] + ':' + row['_id'],
               'context_token_ids': ctx_ids, 'regions': regions}
    question = {'context_id': cid, 'question_id': qid, 'question_token_ids': question_ids}
    label = {'context_id': cid, 'question_id': qid, 'dataset': row['dataset'],
             'reference_texts': row['answers'], 'metric': settings['metric']}
    return context, question, label


def token_length_bucket(count):
    if count < 4096:
        return "0-4095"
    if count <= 8192:
        return "4096-8192"
    if count <= 16384:
        return "8193-16384"
    if count <= 32768:
        return "16385-32768"
    return "32769+"


def choose_preset(selected, definition):
    """Subset frozen selection by declared lengths only; never inspect references."""
    result = []
    seen_contexts = defaultdict(set)
    for row in selected:
        if row['split'] not in definition['splits']:
            continue
        if row['context_tokens'] > definition['max_context_tokens'] or row['total_tokens_with_generation'] > definition['max_total_tokens_including_generation']:
            continue
        cap = definition['max_contexts_per_dataset_split']
        seen = seen_contexts[row['split']]
        if cap is not None and row['context_id'] not in seen and len(seen) >= cap:
            continue
        result.append(row)
        seen.add(row['context_id'])
    return result


def build_presets(manifest, settings, local):
    presets = {}
    for name, definition in settings['presets'].items():
        entry = {'selection_rule': 'filter the frozen superset by context/total token length; preserve original deterministic order',
                 'limits': definition, 'max_new_tokens': settings['max_new_tokens'], 'datasets': {}, 'artifacts': {}}
        for dataset, data in manifest['datasets'].items():
            chosen = choose_preset(data['selected'], definition)
            entry['datasets'][dataset] = {'selected': chosen, 'split_counts': dict(Counter(row['split'] for row in chosen))}
            for split in definition['splits']:
                selected_rows = [row for row in chosen if row['split'] == split]
                cids = {row['context_id'] for row in selected_rows}
                qids = {row['question_id'] for row in selected_rows}
                if not cids:
                    raise ValueError(f'No available examples for preset {name}/{dataset}/{split}')
                key = dataset + '.' + split
                files = {}
                for kind, spec in manifest['artifacts'][key].items():
                    source = ROOT / spec['path']
                    check_file(source, spec)
                    rows = [json.loads(line) for line in source.read_text().splitlines()]
                    filtered = [row for row in rows if (row['context_id'] in cids if kind == 'contexts' else row['question_id'] in qids)]
                    files[kind] = write_jsonl(local / 'prepared' / name / f'{key}.{kind}.jsonl', filtered)
                entry['artifacts'][key] = files
        entry['context_count'] = sum(spec['contexts']['rows'] for spec in entry['artifacts'].values())
        entry['question_count'] = sum(spec['questions']['rows'] for spec in entry['artifacts'].values())
        presets[name] = entry
    return presets


def prepare(lock, settings, local=LOCAL, manifest_path=None):
    if settings['oversize_policy'] != 'reject':
        raise ValueError('This normalizer never truncates code-completion contexts')
    tokenizer, template, prefix, suffix, encode = load_tokenizer(lock, local)
    prefix_ids, suffix_ids = encode(prefix), encode(suffix)
    manifest = {'schema_version': 1, 'status': 'prepared_no_model_inference', 'settings': settings,
                'source_lock_sha256': file_hash(LOCK), 'source_revision': lock['dataset']['revision'],
                'benchmark_code_revision': lock['benchmark_code']['revision'],
                'preprocessor_sha256': file_hash(Path(__file__)),
                'metric_implementation_sha256': file_hash(ROOT / 'data_prep/metrics.py'),
                'settings_sha256': file_hash(SETTINGS),
                'requirements_sha256': file_hash(ROOT / 'data_prep/requirements-tokenizer.txt'),
                'tokenizer_revision': lock['tokenizer']['revision'],
                'template': {'chat_template_sha256': canonical_hash(template), 'prefix_token_ids': prefix_ids,
                             'suffix_token_ids': suffix_ids, 'enable_thinking': False},
                'preprocessing_environment': {'python': platform.python_version(), 'tokenizers': importlib.metadata.version('tokenizers'),
                    'Jinja2': importlib.metadata.version('Jinja2')}, 'datasets': {}, 'artifacts': {}}
    all_selected_hashes = {}
    for dataset, spec in lock['dataset']['members'].items():
        path = local / (dataset + '.jsonl')
        check_file(path, spec)
        records = read_records(path, dataset)
        if len(records) != spec['rows']:
            raise ValueError('Source count mismatch')
        token_data, eligible, inventory = {}, {}, []
        for row in records:
            body = TASK_PREFIX + row['context']
            query = row['input'] + TASK_SUFFIX
            ctx_ids = prefix_ids + encode(body)
            q_ids = encode(query) + suffix_ids
            total = len(ctx_ids) + len(q_ids) + settings['max_new_tokens']
            too_short = len(ctx_ids) < settings['min_context_tokens']
            oversized = len(ctx_ids) > settings['max_context_tokens'] or total > settings['max_total_tokens_including_generation']
            fit = not too_short and not oversized
            eligible[row['_id']] = fit
            # No data is dropped or normalized at the text layer. Round-trip verification is strict.
            if tokenizer.decode(ctx_ids, skip_special_tokens=False) != prefix + body:
                raise ValueError('Context tokenizer round trip altered source text')
            if tokenizer.decode(q_ids, skip_special_tokens=False) != query + suffix:
                raise ValueError('Question tokenizer round trip altered source text')
            ctx_hash = digest(row['context'].encode('utf-8'))
            info = {'source_id': row['_id'], 'source_line': row['_source_line'], 'context_sha256': ctx_hash,
                    'input_sha256': digest(row['input'].encode()), 'references_sha256': canonical_hash(row['answers']),
                    'source_record_sha256': canonical_hash({k: row[k] for k in RAW_KEYS}),
                    'context_tokens': len(ctx_ids), 'question_tokens': len(q_ids), 'total_tokens_with_generation': total,
                    'context_length_bucket': token_length_bucket(len(ctx_ids)),
                    'total_length_bucket': token_length_bucket(total),
                    'language': row['language'], 'eligible': fit, 'rejection_reason': 'too_short_for_transfer_pilot' if too_short else ('oversize_no_truncation' if oversized else None)}
            inventory.append(info)
            token_data[row['_id']] = (ctx_ids, q_ids, info)
        selected = split_groups(records, settings, eligible)
        selected_info = []
        for split in ('development', 'heldout'):
            contexts, questions, labels = [], [], []
            seen_contexts = set()
            for chosen_split, ctx_hash, row in selected:
                if split != chosen_split:
                    continue
                # Exact identical contexts anywhere cannot cross split, including across datasets.
                previous = all_selected_hashes.setdefault(ctx_hash, split)
                if previous != split:
                    raise ValueError('Cross-dataset exact context leakage; revise grouping globally')
                ctx, question, label = make_example(row, split, ctx_hash, token_data[row['_id']], prefix_ids, settings)
                if ctx_hash not in seen_contexts:
                    contexts.append(ctx)
                    seen_contexts.add(ctx_hash)
                questions.append(question)
                labels.append(label)
                monolithic = encode(render_chat(template, TASK_PREFIX + row['context'] + row['input'] + TASK_SUFFIX))
                info = dict(token_data[row['_id']][2], split=split, context_id=ctx['context_id'], question_id=question['question_id'],
                            context_token_ids_sha256=canonical_hash(ctx['context_token_ids']),
                            question_token_ids_sha256=canonical_hash(question['question_token_ids']),
                            equals_monolithic_chat_tokenization=(ctx['context_token_ids'] + question['question_token_ids'] == monolithic),
                            natural_answer_substring_in_context=any(a in row['context'] for a in row['answers']),
                            natural_answer_substring_in_input=any(a in row['input'] for a in row['answers']))
                selected_info.append(info)
            key = dataset + '.' + split
            manifest['artifacts'][key] = {name: write_jsonl(local / 'prepared' / f'{key}.{name}.jsonl', rows)
                                        for name, rows in [('contexts', contexts), ('questions', questions), ('labels', labels)]}
        manifest['datasets'][dataset] = {
            'source_rows': len(records), 'unique_contexts': len({x['context_sha256'] for x in inventory}),
            'eligible_rows': sum(eligible.values()), 'oversize_rejected_rows': sum(x['rejection_reason'] == 'oversize_no_truncation' for x in inventory),
            'too_short_rejected_rows': sum(x['rejection_reason'] == 'too_short_for_transfer_pilot' for x in inventory),
            'input_nonempty_rows': sum(bool(row['input']) for row in records),
            'languages': dict(Counter(row['language'] for row in records)),
            'natural_answer_substring_in_context_rows': sum(any(a in row['context'] for a in row['answers']) for row in records),
            'natural_answer_substring_in_input_rows': sum(any(a in row['input'] for a in row['answers']) for row in records),
            'inventory': inventory, 'selected': selected_info,
            'split_counts': dict(Counter(row['split'] for row in selected_info)),
            'selected_context_length_buckets': dict(Counter(row['context_length_bucket'] for row in selected_info)),
            'selected_total_length_buckets': dict(Counter(row['total_length_bucket'] for row in selected_info))}
        del token_data
    manifest['presets'] = build_presets(manifest, settings, local)
    if manifest_path is None:
        manifest_path = ROOT / 'configs/data/longbench_pilot.manifest.json'
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Download exact public benchmark/tokenizer files; never models')
    parser.add_argument('--prepare', action='store_true', help='Normalize and tokenize already verified local sources, CPU only')
    args = parser.parse_args()
    if not (args.fetch or args.prepare):
        parser.error('Specify --fetch and/or --prepare')
    lock, settings = json.loads(LOCK.read_text()), json.loads(SETTINGS.read_text())
    if args.fetch:
        fetch(lock)
    if args.prepare:
        manifest = prepare(lock, settings)
        print(json.dumps({name: {k: ds[k] for k in ('source_rows', 'eligible_rows', 'too_short_rejected_rows', 'oversize_rejected_rows', 'split_counts')}
                          for name, ds in manifest['datasets'].items()}, indent=2))


if __name__ == '__main__':
    main()
