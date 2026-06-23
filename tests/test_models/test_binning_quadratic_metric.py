import unittest
import numpy as np

from hscredit.core.metrics import quadratic_curve_coefficient, composite_binning_quality


class TestQuadraticCurveCoefficient(unittest.TestCase):
    def test_descending_lift_curve_returns_positive_score(self):
        bins = np.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40)
        y = np.array([1] * 16 + [0] * 24 + [1] * 10 + [0] * 30 + [1] * 6 + [0] * 34 + [1] * 3 + [0] * 37 + [0] * 40)

        score = quadratic_curve_coefficient(bins, y, metric='lift', monotonic='descending')

        self.assertGreater(score, 0)

    def test_bad_rate_metric_supports_valley_trend(self):
        bins = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30 + [4] * 30)
        y = np.array([1] * 9 + [0] * 21 + [1] * 4 + [0] * 26 + [1] * 2 + [0] * 28 + [1] * 5 + [0] * 25 + [1] * 10 + [0] * 20)

        score = quadratic_curve_coefficient(bins, y, metric='bad_rate', monotonic='valley')

        self.assertGreater(score, 0)

    def test_trend_violation_returns_negative_score(self):
        bins = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30)
        y = np.array([1] * 3 + [0] * 27 + [1] * 12 + [0] * 18 + [1] * 6 + [0] * 24 + [1] * 15 + [0] * 15)

        score = quadratic_curve_coefficient(bins, y, metric='lift', monotonic='descending')

        self.assertLess(score, 0)


if __name__ == '__main__':
    unittest.main()

class TestCompositeBinningQuality(unittest.TestCase):
    def test_composite_quality_prefers_stronger_head_tail_and_margin(self):
        bins_good = np.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40)
        y_good = np.array([1] * 18 + [0] * 22 + [1] * 11 + [0] * 29 + [1] * 7 + [0] * 33 + [1] * 4 + [0] * 36 + [1] * 2 + [0] * 38)

        bins_bad = np.array([0] * 40 + [1] * 40 + [2] * 40 + [3] * 40 + [4] * 40)
        y_bad = np.array([1] * 9 + [0] * 31 + [1] * 8 + [0] * 32 + [1] * 7 + [0] * 33 + [1] * 6 + [0] * 34 + [1] * 5 + [0] * 35)

        score_good = composite_binning_quality(bins_good, y_good, metric='lift', monotonic='descending')
        score_bad = composite_binning_quality(bins_bad, y_bad, metric='lift', monotonic='descending')

        self.assertGreater(score_good, score_bad)


