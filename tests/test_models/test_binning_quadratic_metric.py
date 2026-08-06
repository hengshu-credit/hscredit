import unittest
import numpy as np

from hscredit.core.metrics import quadratic_curve_coefficient, composite_binning_quality
from hscredit.core.metrics._binning import _fit_monotone_quadratic


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


class TestFitMonotoneQuadratic(unittest.TestCase):
    """单调约束二次拟合：保证拟合曲线在 x 区间内只朝一个方向发展."""

    def test_descending_constraint_pushes_vertex_outside_interval(self):
        """倒U数据 + descending：约束拟合应把顶点推到区间左侧，区间内单调递减."""
        x = np.linspace(-1.0, 1.0, 5)
        y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])  # 倒U，无约束顶点在 x=0

        a, b, c = _fit_monotone_quadratic(x, y, 'descending')

        # 区间内导数（线性函数）两端点均 <= 0，保证整个区间单调递减
        derivs = 2 * a * x + b
        self.assertLessEqual(derivs[0], 1e-8)
        self.assertLessEqual(derivs[-1], 1e-8)

        # 无约束拟合的顶点应位于区间内（证明约束确实生效）
        a_u, b_u, _ = np.polyfit(x, y, 2)
        vertex_unconstrained = -b_u / (2 * a_u)
        self.assertGreater(vertex_unconstrained, -1.0)
        self.assertLess(vertex_unconstrained, 1.0)

    def test_ascending_constraint_pushes_vertex_outside_interval(self):
        """U形数据 + ascending：约束拟合应保证区间内单调递增."""
        x = np.linspace(-1.0, 1.0, 5)
        y = np.array([3.0, 2.0, 1.0, 2.0, 3.0])  # U形，无约束顶点在 x=0

        a, b, c = _fit_monotone_quadratic(x, y, 'ascending')

        derivs = 2 * a * x + b
        self.assertGreaterEqual(derivs[0], -1e-8)
        self.assertGreaterEqual(derivs[-1], -1e-8)

    def test_monotone_data_keeps_monotone_fit(self):
        """本身单调递减的数据：约束拟合结果应与无约束接近且区间内单调."""
        x = np.linspace(-1.0, 1.0, 5)
        y = np.array([3.0, 2.5, 2.0, 1.5, 1.0])

        a, b, c = _fit_monotone_quadratic(x, y, 'descending')

        derivs = 2 * a * x + b
        self.assertLessEqual(derivs[0], 1e-8)
        self.assertLessEqual(derivs[-1], 1e-8)
        # 拟合值应与原数据接近
        y_pred = a * x ** 2 + b * x + c
        self.assertLess(np.sum((y - y_pred) ** 2), 1e-8)

    def test_ascending_data_with_constraint(self):
        """单调递增数据 + ascending 约束."""
        x = np.linspace(-1.0, 1.0, 5)
        y = np.array([1.0, 1.5, 2.0, 2.5, 3.0])

        a, b, c = _fit_monotone_quadratic(x, y, 'ascending')

        derivs = 2 * a * x + b
        self.assertGreaterEqual(derivs[0], -1e-8)
        self.assertGreaterEqual(derivs[-1], -1e-8)

    def test_quadratic_coefficient_uses_constrained_fit_for_monotone(self):
        """整体指标：对趋势反复的数据，descending 约束拟合给出有限值且不报错."""
        bins = np.array([0] * 30 + [1] * 30 + [2] * 30 + [3] * 30 + [4] * 30)
        # 坏率先降后升（U形），descending 期望下存在违例
        y = np.array([1] * 12 + [0] * 18 + [1] * 4 + [0] * 26 + [1] * 2 + [0] * 28 + [1] * 6 + [0] * 24 + [1] * 10 + [0] * 20)

        score = quadratic_curve_coefficient(bins, y, metric='bad_rate', monotonic='descending')

        self.assertTrue(np.isfinite(score))
        # 存在违例时应为负值惩罚
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


