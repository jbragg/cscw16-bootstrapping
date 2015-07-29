import unittest
import collections

import exp

config_std = {
    "n_authors": 1000,
    "n_resources": 1000,
    "n_links": 10,
    "p_cite": 0.5,
    "p_use": 0.45,
    "p_response_used": 0.059,
    "p_response_used_std": 0.03,
    "p_response_used_bins": 5,
    "p_response_unused": 0.016,
    "p_response_unused_std": 0.0075,
    "p_response_unused_bins": 5,
    "p_response_uncited": 0.0
}
config_basic = {
    "n_authors": 1000,
    "n_resources": 1000,
    "n_links": 10,
    "p_cite": 0.5,
    "p_use": 0.45,
    "p_response_used": 0.059,
    "p_response_unused": 0.016,
    "p_response_uncited": 0.0
}


class TestCalculateProbabilities(unittest.TestCase):
    def setUp(self):
        self.probs_basic_true, self.probs_basic_est = \
            exp.calculate_probabilities(**config_basic)
        self.probs_std_true, self.probs_std_est = \
            exp.calculate_probabilities(**config_std)

    def test_basic(self):
        self.assertDictEqual(self.probs_basic_true, self.probs_basic_est)

    def test_std(self):
        counts = collections.defaultdict(int)
        for x in self.probs_std_true:
            counts[self.probs_std_est[x], self.probs_std_true[x]] += 1
        p_est = set(p1 for p1, p2 in counts)
        for p in p_est:
            p_true = set(p2 for p1, p2 in counts if p1 == p)
            #print p, [(p_, counts[p, p_]) for p_ in sorted(p_true)]
            self.assertLessEqual(len(p_true), 5)  # Number of bins.

class TestExpectation(unittest.TestCase):
    def setUp(self):
        self.probs = {1: 0.3, 2: 0.3, 3:0.4, 4:0.4, 5:0.4, 6:0, 7:0}
        self.f = exp.f_utility_h

    def test_equal(self):
        self.assertAlmostEqual(
            exp.expectation_over_powerset(self.probs, self.f),
            exp.expectation_bounded(self.probs, self.f))

if __name__ == '__main__':
    unittest.main()
