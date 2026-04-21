"""Tests for model_report module."""

import numpy as np
import pandas as pd
import pytest

from hscredit.report.model_report import QuickModelReport


class MockModel:
    """Minimal mock model for testing QuickModelReport."""

    def __init__(self, feature_names=None):
        self._feature_names = feature_names or ['f0']
        self._coef = np.array([0.5] * len(self._feature_names))
        self._intercept = np.array([-0.5])

    def predict_proba(self, X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_feat = min(arr.shape[1], len(self._coef))
        scores = arr[:, :n_feat] @ self._coef[:n_feat] + self._intercept[0]
        prob = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - prob, prob])

    def get_feature_importances(self):
        return pd.Series(dict(zip(self._feature_names, [0.5] * len(self._feature_names))))

    @property
    def feature_importances_(self):
        return np.array([0.5] * len(self._feature_names))


class TestQuickModelReportTarget:
    """Test target parameter handling."""

    def test_target_str(self):
        """target as string column name."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'label': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target='label', feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_target_dict_overdue_dpds(self):
        """target as dict with overdue+dpds."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'overdue': [0, 0, 1, 1, 1, 1],
            'dpds': [0, 2, 3, 5, 6, 10],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target={'overdue': 'overdue', 'dpds': 'dpds', 'threshold': 3},
            feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 0, 1, 1, 1]

    def test_target_dict_overdue_only(self):
        """target as dict with overdue only (no dpds): overdue col > 0 → y=1."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'overdue': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target={'overdue': 'overdue'},
            feature_names=['f0']
        )
        # overdue > 0 → [0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_datasets_y_none(self):
        """datasets dict with y=None derives y from target config."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'target': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, datasets={'train': (X, None)},
            target='target', feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_datasets_y_none_dict_target(self):
        """datasets dict with y=None and dict target."""
        X_train = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'overdue': [0, 1, 1, 1],
            'dpds': [0, 1, 5, 6],
        })
        X_test = pd.DataFrame({
            'f0': [5, 6],
            'overdue': [1, 1],
            'dpds': [4, 8],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model,
            datasets={'train': (X_train, None), 'test': (X_test, None)},
            target={'overdue': 'overdue', 'dpds': 'dpds', 'threshold': 3},
            feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]
        assert report._datasets['test'].y.tolist() == [1, 1]

    def test_y_proba_produced(self):
        """Model produces y_proba after init."""
        X = pd.DataFrame({'f0': [1, 2, 3, 4], 'label': [0, 0, 1, 1]})
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target='label', feature_names=['f0']
        )
        proba = report._datasets['train'].y_proba
        assert proba is not None
        assert len(proba) == 4
        assert proba.min() >= 0 and proba.max() <= 1

    def test_get_metrics(self):
        """get_metrics returns DataFrame with expected columns."""
        X = pd.DataFrame({'f0': [1, 2, 3, 4], 'label': [0, 0, 1, 1]})
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target='label', feature_names=['f0']
        )
        metrics = report.get_metrics()
        assert '统计项' in metrics.columns
        assert 'KS' in metrics['统计项'].values
        assert 'AUC' in metrics['统计项'].values
        assert '样本数' in metrics['统计项'].values
        assert '坏样本率' in metrics['统计项'].values

    def test_target_default_column_fallback(self):
        """When target=None, searches for common column names."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'flag': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target=None, feature_names=['f0']
        )
        # Should find 'flag' column as fallback
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]


class TestQuickModelReportOverdueDpdsSeparate:
    """Test overdue/dpds as separate __init__ parameters (not inside target dict)."""

    def test_overdue_dpds_single_col_single_threshold(self):
        """overdue as str + dpds as int is equivalent to target='col'."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'dpds': [0, 1, 5, 10],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            overdue='dpds', dpds=3, feature_names=['f0']
        )
        # dpds > 3 → [0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_overdue_dpds_single_col_list_thresholds(self):
        """overdue as str + dpds as list thresholds."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'dpds': [0, 3, 7, 15, 20, 30],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            overdue='dpds', dpds=[15, 7, 0], feature_names=['f0']
        )
        # dpds > 15 or > 7 or > 0:
        #   0: 0>15? 0>7? 0>0? → false → 0
        #   3: 3>15? 3>7? 3>0? → false, false, true → 1 ← FAILS: test says [0,1,...]
        # Actually dpds > 0 for all values >= 1, so:
        #   [0, 1, 1, 1, 1, 1]  (only index 0 is false for >0)
        assert report._datasets['train'].y.tolist() == [0, 1, 1, 1, 1, 1]

    def test_overdue_dpds_multi_col(self):
        """overdue as list of str + dpds as list."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'dpds_m1': [0, 0, 0, 0, 0, 0],
            'dpds_m3': [0, 0, 0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            overdue=['dpds_m1', 'dpds_m3'], dpds=[3, 0],
            feature_names=['f0']
        )
        # dpds_m1 > 3 or > 0 → [0, 0, 0, 0, 0, 0]
        # dpds_m3 > 3 or > 0 → [0, 0, 0, 0, 1, 1]
        # any true → y=1 → [0, 0, 0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 0, 0, 1, 1]

    def test_overdue_dpds_override_target(self):
        """overdue/dpds takes priority over target when both provided."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'label': [1, 1, 1, 1],  # would give all 1s
            'dpds': [0, 0, 5, 10],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target='label',  # ignored because overdue/dpds provided
            overdue='dpds', dpds=3, feature_names=['f0']
        )
        # dpds > 3 → [0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_overdue_dpds_with_datasets_dict(self):
        """overdue/dpds works with datasets dict."""
        X_train = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'dpds': [0, 1, 5, 10],
        })
        X_test = pd.DataFrame({
            'f0': [5, 6],
            'dpds': [3, 15],
        })
        model = MockModel(feature_names=['f0'])
        report = QuickModelReport(
            model=model,
            datasets={'train': (X_train, None), 'test': (X_test, None)},
            overdue='dpds', dpds=3, feature_names=['f0']
        )
        # train: dpds > 3 → [0, 0, 1, 1]
        # test:  dpds > 3 → [0, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]
        assert report._datasets['test'].y.tolist() == [0, 1]

    def test_overdue_dpds_auto_model_report(self):
        """auto_model_report with overdue/dpds separate parameters."""
        from hscredit.report.model_report import auto_model_report
        X = pd.DataFrame({
            'f0': list(range(100)),
            'dpds': list(range(100)),
        })
        model = MockModel(feature_names=['f0'])
        report = auto_model_report(
            model=model, X_train=X,
            overdue='dpds', dpds=[30, 15, 7],
            feature_names=['f0'],
            verbose=False, with_plots=False,
        )
        # dpds > 30 or > 15 or > 7 → dpds > 7 → rows 8-99 → 92 out of 100
        assert report._datasets['train'].y.sum() == 92
        assert report._datasets['train'].y.mean() == 0.92

    def test_overdue_dpds_equivalent_to_dict_target(self):
        """overdue/dpds as separate params should produce same y as dict target."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'dpds': [0, 2, 5, 8, 12, 20],
        })
        model = MockModel(feature_names=['f0'])

        # via dict target
        r1 = QuickModelReport(
            model=model, X_train=X, y_train=None,
            target={'overdue': 'dpds', 'dpds': [10, 5, 0]},
            feature_names=['f0']
        )

        # via separate params
        r2 = QuickModelReport(
            model=model, X_train=X, y_train=None,
            overdue='dpds', dpds=[10, 5, 0],
            feature_names=['f0']
        )

        assert r1._datasets['train'].y.tolist() == r2._datasets['train'].y.tolist()
