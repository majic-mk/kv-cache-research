"""CPU fixtures only. Synthetic code/answers below are not benchmark observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from data_prep import prepare_longbench as prep
from data_prep import metrics


def fixture(i, context=None):
    return {'input': '', 'context': context or f'def fixture_{i}():\n', 'answers': ['    return 1'],
            'length': 4, 'dataset': 'lcc', 'language': 'python', 'all_classes': None, '_id': f'fixture-{i}'}


class PreprocessingTests(unittest.TestCase):
    def settings(self):
        return {'seed': 'fixture-seed', 'development_contexts_per_dataset': 2,
                'heldout_contexts_per_dataset': 2, 'region_tokens': 4, 'metric': metrics.METRIC}

    def test_split_invariant_to_record_order_questions_and_labels(self):
        rows = [fixture(i) for i in range(20)]
        eligible = {r['_id']: True for r in rows}
        expected = [(s, h, r['_id']) for s,h,r in prep.split_groups(rows, self.settings(), eligible)]
        changed = list(reversed(copy.deepcopy(rows)))
        for row in changed:
            row['answers'] = ['gold must not influence selection']
            row['input'] = 'future question must not influence ranking'
        actual = [(s,h,r['_id']) for s,h,r in prep.split_groups(changed, self.settings(), eligible)]
        self.assertEqual(actual, expected)

    def test_identical_contexts_never_cross_splits(self):
        rows = [fixture(i // 2, f'ctx {i // 2}') for i in range(12)]
        for i, row in enumerate(rows): row['_id'] = str(i)
        selected = prep.split_groups(rows, self.settings(), {r['_id']: True for r in rows})
        seen = {}
        for split, key, row in selected:
            self.assertEqual(seen.setdefault(key, split), split)
        self.assertEqual(len(seen), 4)
        self.assertEqual(len(selected), 8)

    def test_oversize_exclusion_is_explicit(self):
        rows = [fixture(i) for i in range(5)]
        selected = prep.split_groups(rows, self.settings(), {r['_id']: i != 0 for i,r in enumerate(rows)})
        self.assertNotIn('fixture-0', [r['_id'] for _,_,r in selected])
        with self.assertRaises(ValueError):
            prep.split_groups(rows, self.settings(), {r['_id']: False for r in rows})

    def test_lcc_nonempty_input_rejected_not_silently_dropped(self):
        row = fixture(1)
        prep.validate_record(row, 'lcc')
        row['input'] = 'unanticipated future input'
        with self.assertRaises(ValueError): prep.validate_record(row, 'lcc')

    def test_context_contract_excludes_question_and_gold(self):
        row = fixture(1)
        row['dataset'] = 'repobench-p'
        row['input'] = 'future target-file prefix'
        ctx, question, label = prep.make_example(row, 'development', 'abc', ([7,8,9,10,11,12], [99,100], {}), [7], self.settings())
        self.assertEqual(set(ctx), {'context_id','source_id','context_token_ids','regions'})
        self.assertEqual(set(question), {'context_id','question_id','question_token_ids'})
        self.assertEqual(label['reference_texts'], row['answers'])
        self.assertEqual(ctx['regions'], [{'region_id':'window-0000','label':'fixed_code_window','start':1,'end':5},
                                         {'region_id':'window-0001','label':'fixed_code_window','start':5,'end':6}])

    def test_file_integrity_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fixture'
            p.write_bytes(b'fixture-only')
            prep.check_file(p, {'bytes':12,'sha256':prep.digest(b'fixture-only')})
            with self.assertRaises(ValueError): prep.check_file(p, {'bytes':12,'sha256':'0'*64})

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.jsonl';p.write_text((json.dumps(fixture(1))+'\n')*2)
            with self.assertRaises(ValueError): prep.read_records(p,'lcc')


class MetricTests(unittest.TestCase):
    def test_official_line_filter_is_not_strip_or_last_line(self):
        self.assertEqual(metrics.first_code_line('\n```python\n# comment\n// comment\n  answer\nother'), '  answer')
        self.assertEqual(metrics.first_code_line('```\n\nanswer'), '')
        self.assertEqual(metrics.code_sim_score('a = 1\nignore', ['a = 1']), 1)
        self.assertLess(metrics.code_sim_score('  a = 1', ['a = 1']), 1)

    def test_multiple_references_max(self):
        self.assertEqual(metrics.code_sim_score('good', ['bad', 'good']), 1)

    def test_no_dropped_prediction_denominator(self):
        labels = [{'question_id':'q','dataset':'lcc','metric':metrics.METRIC,'reference_texts':['good']}]
        with self.assertRaises(ValueError): metrics.evaluate([], labels)
        with self.assertRaises(ValueError): metrics.evaluate([{'question_id':'q','prediction':'good'}]*2, labels)
        self.assertEqual(metrics.evaluate([{'question_id':'q','prediction':'good'}],labels)['scores_percent'], {'lcc':100.0})

    def test_matches_actual_fuzzywuzzy_difflib_when_installed(self):
        try:
            from fuzzywuzzy import fuzz
        except ImportError:
            self.skipTest('Optional pinned metric package not installed')
        self.assertEqual(fuzz.SequenceMatcher.__module__, 'difflib')
        for left in ['abc','tide','x'*300+'a','', '  code()']:
            for right in ['abc','diet','x'*300+'b','', 'code()']:
                self.assertEqual(metrics.fuzzy_ratio_difflib(left,right), fuzz.ratio(left,right))


class PreparedLocalIntegrationTests(unittest.TestCase):
    def test_real_manifests_when_present(self):
        path = prep.ROOT/'configs/data/longbench_pilot.manifest.json'
        if not path.exists(): self.skipTest('Real local preparation not available')
        manifest=json.loads(path.read_text());prefix=manifest['template']['prefix_token_ids'];suffix=manifest['template']['suffix_token_ids']
        splits={}
        for key, files in manifest['artifacts'].items():
            rows={}
            for kind,spec in files.items():
                p=prep.ROOT/spec['path']
                if not p.exists(): self.skipTest('Ignored local data absent; reproduce --fetch --prepare')
                prep.check_file(p,spec)
                rows[kind]=[json.loads(x) for x in p.read_text().splitlines()]
            for row in rows['contexts']:
                self.assertEqual(set(row),{'context_id','source_id','context_token_ids','regions'})
                self.assertEqual(row['context_token_ids'][:len(prefix)],prefix)
                split=key.rsplit('.',1)[1]
                self.assertEqual(splits.setdefault(row['context_id'],split),split)
                self.assertLessEqual(len(row['context_token_ids']),32768)
            for row in rows['questions']:
                self.assertEqual(set(row),{'context_id','question_id','question_token_ids'})
                self.assertEqual(row['question_token_ids'][-len(suffix):],suffix)
            self.assertEqual({r['question_id'] for r in rows['questions']},{r['question_id'] for r in rows['labels']})

class PresetTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        self.assertEqual([prep.token_length_bucket(n) for n in (4095,4096,8192,8193,16384,16385,32768,32769)],
                         ['0-4095','4096-8192','4096-8192','8193-16384','8193-16384','16385-32768','16385-32768','32769+'])

    def test_smoke_uses_development_and_total_budget(self):
        selected=[{'split':split,'context_tokens':ctx,'total_tokens_with_generation':total,'question_id':str(i),'context_id':str(i)}
                  for i,(split,ctx,total) in enumerate([('development',7000,9000),('heldout',5000,5100),
                   ('development',5000,5200),('development',6000,7000),('development',6500,7100)])]
        definition={'max_context_tokens':8192,'max_total_tokens_including_generation':8192,'splits':['development'],'max_contexts_per_dataset_split':2}
        frozen=copy.deepcopy(selected)
        chosen=prep.choose_preset(selected,definition)
        self.assertEqual([r['question_id'] for r in chosen],['2','3'])
        self.assertEqual(selected,frozen)

    def test_smoke_cap_counts_contexts_not_questions(self):
        selected=[{'split':'development','context_tokens':5000,'total_tokens_with_generation':5200,
                   'context_id':cid,'question_id':str(i)} for i,cid in enumerate(['a','a','b','c'])]
        definition={'max_context_tokens':8192,'max_total_tokens_including_generation':8192,
                    'splits':['development'],'max_contexts_per_dataset_split':2}
        self.assertEqual([r['question_id'] for r in prep.choose_preset(selected,definition)],['0','1','2'])


if __name__ == '__main__': unittest.main()
