import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from hscredit.core.selectors import NullImportanceSelector, VarianceSelector


class MeanDifferenceImportanceClassifier(BaseEstimator, ClassifierMixin):
    """测试用分类器：用正负样本均值差作为 feature_importances_。"""

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        if len(self.classes_) < 2:
            self.feature_importances_ = np.zeros(X_arr.shape[1], dtype=float)
            return self

        positive = X_arr[y_arr == self.classes_[-1]]
        negative = X_arr[y_arr == self.classes_[0]]
        self.feature_importances_ = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_null_importance_score_uses_actual_minus_null_percentage():
    rng = np.random.RandomState(7)
    y = np.array([0, 1] * 60)
    X = pd.DataFrame({
        'strong': y + rng.normal(0, 0.02, size=len(y)),
        'noise': rng.normal(0, 1.0, size=len(y)),
    })

    selector = NullImportanceSelector(
        MeanDifferenceImportanceClassifier(),
        threshold=0.0,
        cv=3,
        n_runs=4,
        random_state=42,
        n_jobs=1,
    )
    selector.fit(X, y)

    actual_pct = selector.actual_importances_ / selector.actual_importances_.sum()
    null_pct = selector.null_importances_ / selector.null_importances_.sum()
    expected_scores = actual_pct - null_pct
    pd.testing.assert_series_equal(selector.scores_, expected_scores, check_names=False)
    assert list(selector.importance_details_.columns) == [
        '特征', '实际重要性', 'Null重要性', '实际重要性%', 'Null重要性%', '特征得分'
    ]
    details = selector.importance_details_.set_index('特征')
    pd.testing.assert_series_equal(details['实际重要性%'], actual_pct, check_names=False)
    pd.testing.assert_series_equal(details['Null重要性%'], null_pct, check_names=False)
    pd.testing.assert_series_equal(
        selector.actual_importances_, selector.actual_importance_runs_.mean(), check_names=False
    )
    pd.testing.assert_series_equal(
        selector.null_importances_, selector.null_importance_runs_.mean(), check_names=False
    )
    assert selector.get_importance_details().equals(selector.importance_details_)
    assert selector.scores_['strong'] > selector.scores_['noise']
    assert 'strong' in selector.selected_features_


@pytest.mark.parametrize('actual_scale,null_scale', [(1.0, 1.0), (100.0, 0.5)])
@pytest.mark.parametrize('threshold,selected', [(0.0, ['strong']), (0.5, []), (-1.0, ['strong', 'noise'])])
def test_null_importance_normalizes_mean_totals_before_scoring(monkeypatch, actual_scale, null_scale, threshold, selected):
    """原始差值均为负时仍按占比筛选，且两组重要性的独立缩放不影响结果。"""
    def experiment(task):
        run = task[0]
        actual = np.array([[4.0, 12.0], [1.0, 3.0]]) * actual_scale
        null = np.array([[20.0, 40.0], [40.0, 100.0]]) * null_scale
        return run, actual, null

    monkeypatch.setattr('hscredit.core.selectors.null_importance_selector._run_null_importance_experiment', experiment)
    X = pd.DataFrame({'strong': [0, 1] * 4, 'noise': [1, 0] * 4})
    selector = NullImportanceSelector(
        MeanDifferenceImportanceClassifier(), threshold=threshold, cv=2, n_runs=2, n_jobs=1
    ).fit(X, np.array([0, 1] * 4))

    details = selector.get_importance_details().set_index('特征')
    np.testing.assert_allclose(details['实际重要性'], np.array([8.0, 2.0]) * actual_scale)
    np.testing.assert_allclose(details['Null重要性'], np.array([30.0, 70.0]) * null_scale)
    np.testing.assert_allclose(details['实际重要性%'], [0.8, 0.2])
    np.testing.assert_allclose(details['Null重要性%'], [0.3, 0.7])
    np.testing.assert_allclose(selector.scores_, [0.5, -0.5])
    np.testing.assert_allclose(details['特征得分'], selector.scores_)
    assert selector.selected_features_ == selected
    assert selector.transform(X).columns.tolist() == selected

    dropped = selector.get_dropped_df().set_index('特征')
    assert dropped.index.tolist() == [col for col in X if col not in selected]
    if not dropped.empty:
        assert {'实际重要性%', 'Null重要性%'}.issubset(dropped.columns)
        pd.testing.assert_frame_equal(dropped[details.columns], details.loc[dropped.index])
        assert dropped['剔除原因'].eq(f'实际重要性%-Null重要性% <= {threshold}').all()
        assert dropped['阈值'].eq(threshold).all()

    scores = selector.get_scores_df().set_index('特征')
    np.testing.assert_allclose(scores.loc[X.columns, '得分'], [0.5, -0.5])
    assert scores.loc[selected, '状态'].eq('选中').all()
    report = selector.get_selection_report()
    assert report['特征得分'] == selector.scores_.to_dict()
    if not dropped.empty:
        pd.testing.assert_frame_equal(pd.DataFrame(report['剔除详情']), selector.get_dropped_df())


@pytest.mark.parametrize(
    'actual,null,expected_actual,expected_null',
    [
        ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
        ([0.0, 0.0], [1.0, 3.0], [0.0, 0.0], [0.25, 0.75]),
        ([3.0, 1.0], [0.0, 0.0], [0.75, 0.25], [0.0, 0.0]),
    ],
)
def test_null_importance_zero_totals_are_finite(monkeypatch, actual, null, expected_actual, expected_null):
    """任一组总重要性为零时，其占比为零，避免 NaN 影响筛选和报告。"""
    def experiment(task):
        return task[0], np.tile(actual, (2, 1)).T, np.tile(null, (2, 1)).T

    monkeypatch.setattr('hscredit.core.selectors.null_importance_selector._run_null_importance_experiment', experiment)
    X = pd.DataFrame({'first': [0, 1] * 4, 'second': [1, 0] * 4})
    selector = NullImportanceSelector(
        MeanDifferenceImportanceClassifier(), cv=2, n_runs=1, n_jobs=1
    ).fit(X, np.array([0, 1] * 4))

    details = selector.get_importance_details()
    np.testing.assert_allclose(details['实际重要性%'], expected_actual)
    np.testing.assert_allclose(details['Null重要性%'], expected_null)
    expected_scores = np.array(expected_actual) - expected_null
    np.testing.assert_allclose(selector.scores_, expected_scores)
    assert np.isfinite(selector.scores_).all()
    assert selector.selected_features_ == X.columns[expected_scores > 0].tolist()


def test_null_importance_unfitted_details_include_percentage_columns():
    selector = NullImportanceSelector(MeanDifferenceImportanceClassifier())
    details = selector.get_importance_details()
    assert details.empty
    assert list(details.columns) == ['特征', '实际重要性', 'Null重要性', '实际重要性%', 'Null重要性%', '特征得分']


def test_include_feature_is_removed_from_dropped_report_after_force_keep():
    X = pd.DataFrame({
        'keep_me': [1, 1, 1, 1],
        'drop_me': [1, 1, 1, 1],
    })

    selector = VarianceSelector(threshold=0.0, include=['keep_me'])
    selector.fit(X)

    assert 'keep_me' in selector.selected_features_
    assert 'keep_me' not in selector.get_dropped_df()['特征'].tolist()
    assert 'keep_me' not in selector.removed_features_
