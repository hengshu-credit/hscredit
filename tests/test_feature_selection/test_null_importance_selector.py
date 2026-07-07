import numpy as np
import pandas as pd
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


def test_null_importance_score_uses_actual_minus_null_importance():
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
    )
    selector.fit(X, y)

    expected_scores = selector.actual_importances_ - selector.null_importances_
    pd.testing.assert_series_equal(selector.scores_, expected_scores, check_names=False)
    assert list(selector.importance_details_.columns) == ['特征', '实际重要性', 'Null重要性', '特征得分']
    assert selector.get_importance_details().equals(selector.importance_details_)
    assert selector.scores_['strong'] > selector.scores_['noise']
    assert 'strong' in selector.selected_features_


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
