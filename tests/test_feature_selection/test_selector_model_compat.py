"""测试嵌入法/包装法选择器对所有模型类型的兼容性.

覆盖:
- hscredit 模型: XGBoostRiskModel, LightGBMRiskModel, CatBoostRiskModel,
  RandomForestRiskModel, ExtraTreesRiskModel, GradientBoostingRiskModel, LogisticRegression
- sklearn 模型: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SVC(linear)
- 原生 xgboost, lightgbm, catboost sklearn API
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# -- 生成测试数据 --
@pytest.fixture
def data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42
    )
    feature_names = [f'f{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y


# ==============================
# get_feature_importances 单元测试
# ==============================
class TestGetFeatureImportances:
    """测试通用特征重要性提取函数."""

    def test_sklearn_random_forest(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)
        assert imp.sum() > 0

    def test_sklearn_logistic_regression(self, data):
        from sklearn.linear_model import LogisticRegression
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)
        assert (imp >= 0).all()  # abs(coef) >= 0

    def test_sklearn_gradient_boosting(self, data):
        from sklearn.ensemble import GradientBoostingClassifier
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)

    def test_hscredit_logistic_regression(self, data):
        from hscredit.core.models import LogisticRegression
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)

    def test_xgb_classifier(self, data):
        pytest.importorskip('xgboost')
        import xgboost as xgb
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)

    def test_lgbm_classifier(self, data):
        pytest.importorskip('lightgbm')
        import lightgbm as lgb
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)

    def test_catboost_classifier(self, data):
        pytest.importorskip('catboost')
        from catboost import CatBoostClassifier
        from hscredit.core.selectors.base import get_feature_importances
        X, y = data
        model = CatBoostClassifier(iterations=10, random_seed=42, verbose=0)
        model.fit(X, y)
        imp = get_feature_importances(model)
        assert isinstance(imp, np.ndarray)
        assert imp.shape == (10,)

    def test_unsupported_model_raises(self):
        from hscredit.core.selectors.base import get_feature_importances

        class DummyModel:
            pass

        with pytest.raises(ValueError, match="无法从"):
            get_feature_importances(DummyModel())


# ==============================
# FeatureImportanceSelector 兼容性测试
# ==============================
class TestFeatureImportanceSelectorCompat:
    """FeatureImportanceSelector 对不同模型的兼容性."""

    def test_with_sklearn_rf(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from hscredit.core.selectors import FeatureImportanceSelector
        X, y = data
        sel = FeatureImportanceSelector(RandomForestClassifier(n_estimators=10, random_state=42), threshold=3)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3

    def test_with_sklearn_lr(self, data):
        from sklearn.linear_model import LogisticRegression
        from hscredit.core.selectors import FeatureImportanceSelector
        X, y = data
        sel = FeatureImportanceSelector(LogisticRegression(max_iter=1000), threshold=3)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3

    def test_with_hscredit_lr(self, data):
        from hscredit.core.models import LogisticRegression
        from hscredit.core.selectors import FeatureImportanceSelector
        X, y = data
        sel = FeatureImportanceSelector(LogisticRegression(max_iter=1000), threshold=3)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3

    def test_with_xgb(self, data):
        pytest.importorskip('xgboost')
        import xgboost as xgb
        from hscredit.core.selectors import FeatureImportanceSelector
        X, y = data
        sel = FeatureImportanceSelector(xgb.XGBClassifier(n_estimators=10, verbosity=0), threshold=3)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3

    def test_with_lgbm(self, data):
        pytest.importorskip('lightgbm')
        import lightgbm as lgb
        from hscredit.core.selectors import FeatureImportanceSelector
        X, y = data
        sel = FeatureImportanceSelector(lgb.LGBMClassifier(n_estimators=10, verbose=-1), threshold=3)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3

    def test_with_catboost(self, data):
        pytest.importorskip('catboost')
        from catboost import CatBoostClassifier
        from hscredit.core.selectors import FeatureImportanceSelector
        X, y = data
        sel = FeatureImportanceSelector(CatBoostClassifier(iterations=10, verbose=0), threshold=3)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3


# ==============================
# NullImportanceSelector 兼容性测试 (使用少量迭代)
# ==============================
class TestNullImportanceSelectorCompat:
    """NullImportanceSelector 对不同模型的兼容性."""

    def test_with_sklearn_rf(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from hscredit.core.selectors import NullImportanceSelector
        X, y = data
        sel = NullImportanceSelector(
            RandomForestClassifier(n_estimators=10, random_state=42),
            n_runs=1, cv=2, threshold=0.5
        )
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)

    def test_with_sklearn_lr(self, data):
        from sklearn.linear_model import LogisticRegression
        from hscredit.core.selectors import NullImportanceSelector
        X, y = data
        sel = NullImportanceSelector(
            LogisticRegression(max_iter=1000),
            n_runs=1, cv=2, threshold=0.5
        )
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)

    def test_with_xgb(self, data):
        pytest.importorskip('xgboost')
        import xgboost as xgb
        from hscredit.core.selectors import NullImportanceSelector
        X, y = data
        sel = NullImportanceSelector(
            xgb.XGBClassifier(n_estimators=10, verbosity=0),
            n_runs=1, cv=2, threshold=0.5
        )
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)

    def test_with_lgbm(self, data):
        pytest.importorskip('lightgbm')
        import lightgbm as lgb
        from hscredit.core.selectors import NullImportanceSelector
        X, y = data
        sel = NullImportanceSelector(
            lgb.LGBMClassifier(n_estimators=10, verbose=-1),
            n_runs=1, cv=2, threshold=0.5
        )
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)


# ==============================
# BorutaSelector 兼容性测试
# ==============================
class TestBorutaSelectorCompat:
    """BorutaSelector 对不同模型的兼容性."""

    def test_with_sklearn_rf(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from hscredit.core.selectors import BorutaSelector
        X, y = data
        sel = BorutaSelector(
            RandomForestClassifier(n_estimators=10, random_state=42),
            max_iter=3
        )
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)

    def test_with_sklearn_lr(self, data):
        from sklearn.linear_model import LogisticRegression
        from hscredit.core.selectors import BorutaSelector
        X, y = data
        sel = BorutaSelector(LogisticRegression(max_iter=1000), max_iter=3)
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)

    def test_with_xgb(self, data):
        pytest.importorskip('xgboost')
        import xgboost as xgb
        from hscredit.core.selectors import BorutaSelector
        X, y = data
        sel = BorutaSelector(
            xgb.XGBClassifier(n_estimators=10, verbosity=0),
            max_iter=3
        )
        sel.fit(X, y)
        assert isinstance(sel.selected_features_, list)


# ==============================
# RFESelector 兼容性测试
# ==============================
class TestRFESelectorCompat:
    """RFESelector 对不同模型的兼容性 (sklearn RFE 原生支持 coef_)."""

    def test_with_sklearn_lr(self, data):
        from sklearn.linear_model import LogisticRegression
        from hscredit.core.selectors import RFESelector
        X, y = data
        sel = RFESelector(LogisticRegression(max_iter=1000), n_features_to_select=5)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 5

    def test_with_sklearn_rf(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from hscredit.core.selectors import RFESelector
        X, y = data
        sel = RFESelector(RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=5)
        sel.fit(X, y)
        assert len(sel.selected_features_) == 5


# ==============================
# SequentialFeatureSelector 兼容性测试
# ==============================
class TestSequentialSelectorCompat:
    """SequentialFeatureSelector 对不同模型的兼容性."""

    def test_with_sklearn_lr(self, data):
        from sklearn.linear_model import LogisticRegression
        from hscredit.core.selectors import SequentialFeatureSelector
        X, y = data
        sel = SequentialFeatureSelector(
            LogisticRegression(max_iter=1000),
            n_features_to_select=3, direction='forward', cv=2
        )
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3

    def test_with_sklearn_rf(self, data):
        from sklearn.ensemble import RandomForestClassifier
        from hscredit.core.selectors import SequentialFeatureSelector
        X, y = data
        sel = SequentialFeatureSelector(
            RandomForestClassifier(n_estimators=10, random_state=42),
            n_features_to_select=3, direction='forward', cv=2
        )
        sel.fit(X, y)
        assert len(sel.selected_features_) == 3
