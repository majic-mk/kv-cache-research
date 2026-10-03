"""Synthetic-only unit tests. These tests establish no model-quality result."""
import unittest
import numpy as np
from reference import (Chunk, anchor_diagnostics, calibrate_first_chunk, echo_chunk_layout,
                       global_keep_mask, quantile_map, rank_only_oracle,
                       reconstruction_ledger, scale_map_oracle, selection_report)


class QuantileMapTests(unittest.TestCase):
    def test_interpolation_and_clipping(self):
        np.testing.assert_array_equal(quantile_map([[3, 1, 2]], [[30, 10, 20]],
                                                  [[0, 1, 1.5, 2.5, 3, 4]]),
                                      [[10, 10, 15, 25, 30, 30]])

    def test_separately_sorted_not_paired_regression(self):
        np.testing.assert_array_equal(quantile_map([[1, 2, 3]], [[30, 10, 20]], [[1, 2, 3]]),
                                      [[10, 20, 30]])

    def test_independent_heads(self):
        np.testing.assert_array_equal(quantile_map([[1, 2, 3], [1, 2, 3]],
                                                  [[10, 20, 30], [100, 200, 300]], [[1.5], [1.5]]),
                                      [[15], [150]])

    def test_left_tied_minimum(self):
        np.testing.assert_array_equal(quantile_map([[1, 1, 2]], [[10, 20, 30]], [[0, 1, 1.5]]),
                                      [[10, 10, 25]])

    def test_left_tied_maximum_is_not_largest_target(self):
        np.testing.assert_array_equal(quantile_map([[1, 2, 2]], [[10, 20, 30]], [[2, 99]]), [[20, 20]])

    def test_all_tied_source(self):
        np.testing.assert_array_equal(quantile_map([[1, 1, 1]], [[10, 20, 30]], [[0, 1, 2]]),
                                      [[10, 10, 10]])

    def test_empty_evaluation_values(self):
        self.assertEqual(quantile_map([[1, 2]], [[1, 2]], np.empty((1, 0))).shape, (1, 0))

    def test_reject_degenerate_anchor(self):
        for source in (np.empty((1, 0)), [[1]], np.empty((0, 2))):
            with self.subTest(shape=np.shape(source)), self.assertRaises(ValueError):
                quantile_map(source, source, np.empty((np.shape(source)[0], 0)))

    def test_reject_nonfinite_any_argument(self):
        for index in range(3):
            for value in (np.nan, np.inf, -np.inf):
                args = [np.array([[1., 2.]]), np.array([[3., 4.]]), np.array([[1.5, 1.7]])]
                args[index][0, 0] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    quantile_map(*args)

    def test_reject_invalid_shapes(self):
        for args in (([1, 2], [[1, 2]], [[1]]), ([[1, 2]], [[1, 2, 3]], [[1]]),
                     ([[1, 2]], [[1, 2]], [[1], [2]])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                quantile_map(*args)

    def test_monotone_and_target_bounded_random_fixture(self):
        rng = np.random.default_rng(17)
        s, t = rng.uniform(0, 1, (6, 257)), rng.uniform(0, 1, (6, 257))
        v = np.tile(np.linspace(-1, 2, 1000), (6, 1))
        out = quantile_map(s, t, v)
        self.assertTrue((np.diff(out, axis=-1) >= 0).all())
        self.assertTrue((out >= t.min(-1, keepdims=True)-1e-7).all())
        self.assertTrue((out <= t.max(-1, keepdims=True)+1e-7).all())


class CalibrationTests(unittest.TestCase):
    def test_first_exact_and_prefix_unchanged(self):
        v = np.array([[[.5, 1, 2, 3, 1.5, 2.5, 4]]])
        e = np.array([[[10, 20, 30]]])
        np.testing.assert_array_equal(calibrate_first_chunk(v, e, start=1), [[[.5, 10, 20, 30, 15, 25, 30]]])

    def test_no_later_chunk_skips_map(self):
        np.testing.assert_array_equal(calibrate_first_chunk([[[1]]], [[[10]]]), [[[10]]])

    def test_layer_head_granularity(self):
        v = np.tile([1, 2, 3, 1.5], (2, 2, 1))
        scale = np.array([[10, 100], [1000, 10000]])[..., None]
        e = v[..., :3] * scale
        np.testing.assert_array_equal(calibrate_first_chunk(v, e)[..., -1:], 1.5 * scale)

    def test_bad_scope_and_bounds(self):
        for kwargs in ({"scope": "bad"}, {"start": -1}, {"start": 2}):
            with self.assertRaises(ValueError):
                calibrate_first_chunk([[[1, 2, 3]]], [[[10, 20]]], **kwargs)


class BudgetTests(unittest.TestCase):
    def test_actual_first_input_budget(self):
        chunks = echo_chunk_layout(10003, 3, 8, 12, 5)
        self.assertEqual(chunks[0], Chunk(3, 2038, 13))
        self.assertEqual(chunks[0].repeat_input_tokens, 2048)
        self.assertEqual(chunks[1], Chunk(2038, 4086, 25))
        self.assertEqual(chunks[-1].end, 10003)
        self.assertEqual(sum(c.context_tokens for c in chunks), 10000)

    def test_short_context(self):
        chunks = echo_chunk_layout(13, 3, 8, 12, 5)
        self.assertEqual(chunks, [Chunk(3, 13, 13)])

    def test_small_budget_author_max_one_can_exceed_budget(self):
        chunks = echo_chunk_layout(20, 3, 8, 12, 5, chunk_size=4)
        self.assertEqual(chunks[0], Chunk(3, 4, 13))
        self.assertGreater(chunks[0].repeat_input_tokens, 4)
        self.assertEqual(chunks[1].prompt_tokens, 12 + 1 + 5)

    def test_reject_empty_and_bad_lengths(self):
        for args in ((3,3,8,12,5), (4,3,8,12,5,0), (4,3,-1,12,5), (4.0,3,8,12,5)):
            with self.assertRaises(ValueError):
                echo_chunk_layout(*args)

    def test_ledger_token_equality_not_compute_equality(self):
        one = reconstruction_ledger([Chunk(0, 2035, 13)], 16000)
        four = reconstruction_ledger([Chunk(i*499, (i+1)*499, 13) for i in range(4)], 16000)
        self.assertEqual(one["reconstruction_input_tokens"], four["reconstruction_input_tokens"])
        self.assertNotEqual(one["dense_causal_attention_pairs"], four["dense_causal_attention_pairs"])
        self.assertNotEqual(one["context_anchor_tokens"], four["context_anchor_tokens"])


class DiagnosticTests(unittest.TestCase):
    def test_scale_oracle_repairs_scale_not_rank(self):
        v = np.array([[[.1,.2,.3,.4,.1,.2,.3,.4]]])
        exact = v * np.array([1,1,1,1,10,10,10,10])
        regions = [0]*4 + [1]*4
        np.testing.assert_allclose(scale_map_oracle(v, exact, regions), exact)
        reverse = exact[..., ::-1]
        self.assertFalse(np.allclose(scale_map_oracle(v, reverse, regions), reverse))

    def test_rank_oracle_preserves_region_histograms(self):
        c = np.array([[[.4,.3,.2,.1,8,6,4,2]]], dtype=np.float32)
        e = np.array([[[.1,.2,.3,.4,2,4,6,8]]], dtype=np.float32)
        r = np.array([0]*4 + [1]*4)
        out = rank_only_oracle(c, e, r)
        np.testing.assert_array_equal(out, e)
        for region in [0,1]:
            np.testing.assert_array_equal(np.sort(out[...,r==region]), np.sort(c[...,r==region]))

    def test_scale_oracle_ties_do_not_fabricate_rank(self):
        out = scale_map_oracle([[[1,1,1]]], [[[10,20,30]]], [0,0,0])
        np.testing.assert_array_equal(out, [[[10,10,10]]])

    def test_global_count_and_sink(self):
        s = np.arange(32).reshape(2,2,8)
        keep = global_keep_mask(s, .75, n_sink=1)
        self.assertEqual(keep.sum(), 8)
        self.assertTrue(keep[...,0].all())

    def test_stable_ties_explicitly_differ_from_torch_contract(self):
        keep = global_keep_mask(np.ones((1,1,4)), .5)
        np.testing.assert_array_equal(keep, [[[False,False,True,True]]])

    def test_reject_impossible_sink_budget(self):
        with self.assertRaises(ValueError):
            global_keep_mask(np.ones((1,1,4)), .9, n_sink=2)

    def test_report_scale_fixture_improves_logical_selection(self):
        v = np.array([[[.1,.2,.3,.4,.1,.2,.3,.4]]])
        e = v * np.array([1,1,1,1,10,10,10,10])
        r = [0]*4 + [1]*4
        before = selection_report(v, e, r, .5)
        after = selection_report(scale_map_oracle(v,e,r), e, r, .5)
        self.assertGreater(before["retention_disagreement_fraction"], after["retention_disagreement_fraction"])
        self.assertEqual(after["exact_selected_pair_recall"], 1.)
        self.assertFalse(after["physical_memory_measured"])

    def test_anchor_support_and_ties(self):
        d = anchor_diagnostics([[1,1,2]], [[0,1,2,3]])
        self.assertEqual(d["below_support_fraction"], [.25])
        self.assertEqual(d["above_support_fraction"], [.25])
        self.assertAlmostEqual(d["duplicate_anchor_fraction"][0], 1/3)

    def test_invalid_regions(self):
        for r in ([0], [0.,1.], [[0,1]]):
            with self.assertRaises(ValueError):
                scale_map_oracle([[[1,2]]], [[[10,20]]], r)


if __name__ == "__main__":
    unittest.main()
