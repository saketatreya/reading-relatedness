"""Scientific validity checks; run before examining new empirical outcomes."""
import os
from pathlib import Path
import unittest

import numpy as np

from experiments import evaluation as e


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.levels = [tuple((i >> k) & 1 for k in (2, 1, 0)) for i in range(8)]
        self.G, self.splits, _ = e.reference(self.levels)
        self.Q, _ = e.quartet_bank(8)

    def test_known_additive_tree_and_full_recovery(self):
        s = e.score_tree(e.nj(self.G), self.splits, self.G, self.Q)
        self.assertEqual(s["available"], 5)
        self.assertEqual(s["split_recall"], 1.)
        self.assertEqual(s["tree_quartet"], 1.)
        self.assertTrue(s["full_tree_exact"])
        self.assertEqual(s["normalized_rf"], 0.)

    def test_rerooting_and_complement_deduplication(self):
        t1 = ((0, 1), (2, 3))
        t2 = (0, (1, (2, 3)))
        self.assertEqual(e.tree_splits(t1, 4), e.tree_splits(t2, 4))
        self.assertEqual(len(e.tree_splits(t1, 4)), 1)
        self.assertIsNone(e.canonical_split({0}, 4))
        self.assertIsNone(e.canonical_split({0, 1, 2}, 4))

    def test_leaf_order_additive_and_scoring(self):
        p = np.random.default_rng(4).permutation(8)
        tree = e.nj(self.G[np.ix_(p, p)], labels=p)
        self.assertEqual(e.tree_splits(tree, 8), self.splits)

    def test_star_reference_is_unresolved(self):
        G, splits, _ = e.reference([(i,) for i in range(8)])
        self.assertEqual(len(splits), 0)
        q, answers = e.resolved_reference(G, self.Q)
        self.assertEqual(len(q), 0)
        self.assertIsNone(e.quartet_credit(self.G, q, answers))
        self.assertIsNone(e.score_tree(e.nj(G), splits, G, self.Q)["split_recall"])

    def test_ties_and_group_chance(self):
        D = np.ones((8, 8)) - np.eye(8)
        Q, answers = e.resolved_reference(self.G, self.Q)
        self.assertAlmostEqual(e.quartet_credit(D, Q, answers), 1 / 3)
        groups = ["a"] * 4 + ["b"] * 3 + ["c"]
        self.assertAlmostEqual(e.nn_purity(D, groups), e.nn_chance(groups))
        p = e.permutation_tests(D, self.G, groups, self.Q, nperm=19)
        self.assertEqual(p["distance_r_p"], 1)
        self.assertEqual(p["nn_purity_p"], 1)
        self.assertEqual(p["distance_quartet_p"], 1)

    def test_sample_unique_shared_and_exhaustive(self):
        Q, rule = e.quartet_bank(31, count=100)
        self.assertEqual(len(set(map(tuple, Q))), 100)
        np.testing.assert_array_equal(Q, e.quartet_bank(31, count=100)[0])
        self.assertTrue(np.all(np.diff(Q, axis=1) > 0))
        self.assertEqual(len(self.Q), 70)
        self.assertEqual(rule, "uniform distinct sample")

    def test_radial_and_translation_invariances(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(8, 6))
        np.testing.assert_allclose(e.distances(X, "cosine"),
                                   e.distances(X * np.exp(rng.normal(size=(8, 1))), "cosine"))
        np.testing.assert_allclose(e.distances(X, "euclid"),
                                   e.distances(X + 7, "euclid"), atol=1e-12)
        self.assertGreater(np.max(np.abs(e.distances(X, "cosine")
                                         - e.distances(X + 7, "cosine"))), .1)

    def test_newick_and_malformed_labels(self):
        t = e.parse_newick("((t0000:0.1,t0001:-2e-3):1,t0002:0,t0003:2);")
        self.assertEqual(e.tree_splits(t, 4), {frozenset({0, 1})})
        with self.assertRaises(ValueError):
            e.parse_newick("(bad:1,t0001:1);")
        with self.assertRaises(ValueError):
            e.tree_splits((0, 0, 2, 3), 4)

    def test_four_point_residual_exact_tree(self):
        scores = e.direct_scores(self.G, self.G, [p[:2] for p in self.levels], self.Q)
        self.assertEqual(scores["four_point_mean"], 0)
        self.assertEqual(scores["distance_quartet"], 1)

    def test_nonlinear_monotone_transform_can_change_quartet(self):
        rng = np.random.default_rng(42)
        found = False
        for _ in range(1000):
            D = e.distances(rng.normal(size=(4, 3)), "euclid")
            Q = np.array([[0, 1, 2, 3]])
            if np.argmin(e.quartet_sums(D, Q)) != np.argmin(e.quartet_sums(D ** 2, Q)):
                found = True
                break
        self.assertTrue(found)

    def test_bounded_perturbation_inside_sufficient_radius(self):
        rng = np.random.default_rng(1)
        for _ in range(10):
            U = rng.uniform(-.49, .49, (8, 8))
            U = np.triu(U, 1)
            D = self.G + U + U.T
            self.assertEqual(e.tree_splits(e.nj(D), 8), self.splits)

    def test_fastme_known_tree(self):
        binary = Path(os.environ.get("FASTME", str(Path(__file__).resolve().parents[1] / "experiments/.cache/fastme/bin/fastme")))
        if not binary.exists():
            self.fail("Build FastME first; the independent implementation check is required.")
        tree, _ = e.fastme(self.G, binary)
        self.assertEqual(e.tree_splits(tree, 8), self.splits)
        p = np.random.default_rng(2).permutation(8)
        tree, _ = e.fastme(self.G[np.ix_(p, p)], binary)
        def remap(t):
            return int(p[t]) if isinstance(t, int) else tuple(remap(c) for c in t)
        self.assertEqual(e.tree_splits(remap(tree), 8), self.splits)

    def test_fastme_small_decimal_distances(self):
        binary = Path(os.environ.get("FASTME", str(Path(__file__).resolve().parents[1] / "experiments/.cache/fastme/bin/fastme")))
        tree, _ = e.fastme(self.G * 1e-6, binary)
        self.assertEqual(e.tree_splits(tree, 8), self.splits)


if __name__ == "__main__":
    unittest.main(verbosity=2)
