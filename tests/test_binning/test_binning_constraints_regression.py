"""分箱约束回归测试.

固化以下约束在受影响方法上的回归覆盖：
1. 单调性约束
2. min_bin_size 约束
3. max_bin_size 约束
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from hscredit.core.binning import OptimalBinning
from hscredit.core.metrics import quadratic_curve_coefficient
from hscredit.core.binning.or_binning import ORTOOLS_AVAILABLE
from hscredit.report.feature_analyzer import benchmark_binning_methods


class TestBinningConstraintRegression(unittest.TestCase):
    """验证 OptimalBinning 在不同方法下统一遵守约束。"""

    MONOTONIC_METHODS = ['tree', 'chi', 'best_ks', 'best_iv', 'best_lift', 'cart', 'mdlp', 'smooth']
    SIZE_CONSTRAINT_METHODS = [
        'target_bad_rate', 'best_lift', 'smooth', 'cart',
        'mdlp', 'best_iv', 'best_ks', 'chi', 'tree'
    ]
    KERNEL_CHECK_METHODS = ['kernel_density', 'chi', 'best_lift', 'cart', 'target_bad_rate', 'or_tools', 'best_iv']

    @staticmethod
    def create_monotonic_test_data(n_samples: int = 300, random_state: int = 7):
        """构造适合约束验证的稳定数值型样本。"""
        rng = np.random.default_rng(random_state)
        x = pd.Series(rng.normal(size=n_samples), name='x')

        prob = np.where(
            x < -1.2, 0.35,
            np.where(x < -0.2, 0.22, np.where(x < 0.8, 0.11, 0.03))
        )
        y = pd.Series((rng.random(n_samples) < prob).astype(int), name='target')
        return pd.DataFrame({'x': x}), y

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = cls.create_monotonic_test_data()
        cls.n_samples = len(cls.y)
        cls.min_bin_size = 0.12
        cls.max_bin_size = 0.42
        cls.min_count = int(cls.n_samples * cls.min_bin_size)
        cls.max_count = int(np.ceil(cls.n_samples * cls.max_bin_size))

    def _fit_binner(self, method: str) -> OptimalBinning:
        binner = OptimalBinning(
            method=method,
            max_n_bins=5,
            min_n_bins=2,
            min_bin_size=self.min_bin_size,
            max_bin_size=self.max_bin_size,
            monotonic='descending',
            verbose=False,
        )
        binner.fit(self.X, self.y)
        return binner

    @staticmethod
    def _valid_bin_table(binner: OptimalBinning) -> pd.DataFrame:
        table = binner.get_bin_table('x')
        return table[table['分箱'] >= 0].reset_index(drop=True)

    def test_monotonic_constraint_is_enforced_for_affected_methods(self):
        """受影响方法应真正满足单调递减约束。"""
        collected_splits = {}
        for method in self.MONOTONIC_METHODS:
            with self.subTest(method=method):
                binner = self._fit_binner(method)
                table = self._valid_bin_table(binner)
                bad_rates = table['坏样本率'].to_numpy(dtype=float)
                collected_splits[method] = tuple(np.round(np.asarray(binner.splits_['x'], dtype=float), 6))

                self.assertGreaterEqual(len(bad_rates), 2, msg=f'{method} 应至少保留 2 个有效分箱')
                self.assertTrue(
                    np.all(np.diff(bad_rates) <= 1e-10),
                    msg=f'{method} 未满足 monotonic=descending，坏样本率序列: {bad_rates.tolist()}'
                )

        self.assertGreater(
            len(set(collected_splits.values())),
            1,
            msg=f'不同 method 在 monotonic 后不应退化为完全相同的切分点，当前结果: {collected_splits}'
        )

    def test_min_bin_size_constraint_is_enforced_for_affected_methods(self):
        """受影响方法应满足每箱最小样本量约束。"""
        for method in self.SIZE_CONSTRAINT_METHODS:
            with self.subTest(method=method):
                binner = self._fit_binner(method)
                table = self._valid_bin_table(binner)
                counts = table['样本总数'].to_numpy(dtype=int)

                self.assertTrue(
                    np.all(counts >= self.min_count),
                    msg=f'{method} 未满足 min_bin_size={self.min_bin_size}，分箱样本数: {counts.tolist()}'
                )

    def test_max_bin_size_constraint_is_enforced_for_affected_methods(self):
        """受影响方法应满足每箱最大样本量约束。"""
        for method in self.SIZE_CONSTRAINT_METHODS:
            with self.subTest(method=method):
                binner = self._fit_binner(method)
                table = self._valid_bin_table(binner)
                counts = table['样本总数'].to_numpy(dtype=int)

                self.assertTrue(
                    np.all(counts <= self.max_count),
                    msg=f'{method} 未满足 max_bin_size={self.max_bin_size}，分箱样本数: {counts.tolist()}'
                )


if __name__ == '__main__':
    unittest.main()

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / 'examples'
TARGET_DATA_PATH = None
for _fname in ('hengshucredit_yyp.xlsx', 'hscredit_yyp.xlsx'):
    _p = _EXAMPLES_DIR / _fname
    if _p.is_file():
        TARGET_DATA_PATH = _p
        break
if TARGET_DATA_PATH is None:
    TARGET_DATA_PATH = _EXAMPLES_DIR / 'hscredit_yyp.xlsx'


class TestORBinningConstraintRegression(unittest.TestCase):
    """验证 OR-Tools 方法在可用环境下也遵守单调性约束。"""

    @unittest.skipUnless(ORTOOLS_AVAILABLE, 'OR-Tools 未安装')
    def test_or_tools_monotonic_constraint(self):
        rng = np.random.default_rng(7)
        x = pd.Series(rng.normal(size=300), name='x')
        prob = np.where(
            x < -1.2, 0.35,
            np.where(x < -0.2, 0.22, np.where(x < 0.8, 0.11, 0.03))
        )
        y = pd.Series((rng.random(300) < prob).astype(int), name='target')
        X = pd.DataFrame({'x': x})

        binner = OptimalBinning(
            method='or_tools',
            max_n_bins=5,
            min_n_bins=2,
            monotonic='descending',
            time_limit=5,
            verbose=False,
            lift_refine=False,
        )
        binner.fit(X, y)

        table = binner.get_bin_table('x')
        valid = table[table['分箱'] >= 0].reset_index(drop=True)
        bad_rates = valid['坏样本率'].to_numpy(dtype=float)

        self.assertGreaterEqual(len(valid), 2)
        self.assertLessEqual(len(valid), 5)
        self.assertTrue(np.all(np.diff(bad_rates) <= 1e-10), msg=bad_rates.tolist())
    @unittest.skipUnless(ORTOOLS_AVAILABLE and TARGET_DATA_PATH.exists(), '目标数据或 OR-Tools 不可用')
    def test_or_tools_uses_more_than_three_bins_on_target_field(self):
        df = pd.read_excel(TARGET_DATA_PATH)
        X = df[['中智小牛分C3']].copy()
        y = (df['MOB1'] > 3).astype(int)

        binner = OptimalBinning(
            method='or_tools',
            max_n_bins=5,
            min_n_bins=2,
            monotonic='descending',
            time_limit=10,
            verbose=False,
            lift_refine=False,
        )
        binner.fit(X, y)

        table = binner.get_bin_table('中智小牛分C3')
        valid = table[table['分箱'] >= 0].reset_index(drop=True)
        bad_rates = valid['坏样本率'].to_numpy(dtype=float)

        self.assertGreaterEqual(len(valid), 4, msg=table.to_dict('records'))
        self.assertLessEqual(len(valid), 5, msg=table.to_dict('records'))
        self.assertTrue(np.all(np.diff(bad_rates) <= 1e-10), msg=bad_rates.tolist())

class TestBinningMethodBenchmark(unittest.TestCase):
    """验证分箱方法基准只依赖 hscredit 内部实现。"""

    def test_benchmark_binning_methods_returns_hscredit_methods_only(self):
        X, y = TestBinningConstraintRegression.create_monotonic_test_data(n_samples=240, random_state=19)
        df = X.copy()
        df['MOB1'] = np.where(y.to_numpy() > 0, 5, 0)

        result = benchmark_binning_methods(
            df,
            feature='x',
            overdue_col='MOB1',
            dpds=[3],
            max_n_bins=5,
            min_bin_size=0.05,
            monotonic='descending',
            hscredit_methods=['chi', 'cart', 'mdlp'],
        )

        self.assertFalse(result.empty)
        self.assertTrue(result['method'].str.startswith('hscredit-').all(), msg=result['method'].tolist())
        self.assertFalse(result['method'].str.contains('toad|optbinning', regex=True).any())
        self.assertEqual(set(result['dpd'].tolist()), {3})
        self.assertTrue({'n_bins', 'head_lift', 'tail_lift', 'edge_gap', 'turns'}.issubset(result.columns))

class TestNotebookTargetFieldRegression(unittest.TestCase):
    """覆盖 notebook 中暴露的目标字段问题。"""

    @unittest.skipUnless(TARGET_DATA_PATH.exists(), '目标数据不存在')
    def test_notebook_methods_do_not_error_and_do_not_leave_adjacent_zero_bad_rate_bins(self):
        df = pd.read_excel(TARGET_DATA_PATH)
        X = df[['中智小牛分C3']].copy()
        y = (df['MOB1'] > 3).astype(int)

        methods = ['kernel_density', 'chi', 'best_lift', 'cart', 'target_bad_rate', 'best_iv']
        if ORTOOLS_AVAILABLE:
            methods.append('or_tools')

        for method in methods:
            with self.subTest(method=method):
                kwargs = dict(
                    method=method,
                    max_n_bins=5,
                    min_n_bins=2,
                    monotonic='descending',
                    verbose=False,
                    lift_refine=False,
                )
                if method == 'or_tools':
                    kwargs['time_limit'] = 10

                binner = OptimalBinning(**kwargs)
                binner.fit(X, y)

                table = binner.get_bin_table('中智小牛分C3')
                valid = table[table['分箱'] >= 0].reset_index(drop=True)
                bad_rates = valid['坏样本率'].to_numpy(dtype=float)

                zero_pairs = int(np.sum((bad_rates[:-1] <= 1e-12) & (bad_rates[1:] <= 1e-12))) if len(bad_rates) > 1 else 0
                self.assertEqual(zero_pairs, 0, msg=f'{method} bad_rates={bad_rates.tolist()}')

                if len(valid) >= 3:
                    bins = binner.transform(X)['中智小牛分C3'].to_numpy()
                    quad = quadratic_curve_coefficient(
                        bins=bins,
                        y=y.to_numpy(),
                        metric='lift',
                        monotonic='descending',
                    )
                    self.assertGreaterEqual(quad, 0.0, msg=f'{method} lift={valid["LIFT值"].tolist()}')

