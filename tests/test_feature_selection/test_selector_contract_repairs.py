"""筛选器审计缺陷的端到端回归测试。"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

import hscredit.core.selectors as selectors
from hscredit.core.selectors import (
    BorutaSelector,
    Chi2Selector,
    CompositeFeatureSelector,
    CorrSelector,
    FTestSelector,
    FeatureImportanceSelector,
    LiftSelector,
    ModeSelector,
    MutualInfoSelector,
    PSISelector,
    ScorecardFeatureSelection,
    StabilityAwareSelector,
    StepwiseSelector,
    VarianceSelector,
    VIFSelector,
)
from hscredit.core.selectors.base import BaseFeatureSelector, SelectionReportCollector, get_feature_importances
from hscredit.core.selectors.corr_selector import DEFAULT_BINNING_PARAMS
from hscredit.core.selectors.iv_selector import _compute_iv_single
from hscredit.core.selectors.psi_selector import _compute_psi_single
from hscredit.exceptions import ValidationError


class CustomImportanceClassifier(ClassifierMixin, BaseEstimator):
    """只通过自定义属性公开重要性的最小真实 estimator。"""

    def fit(self, X, y):
        self.custom_importances_ = np.arange(1, X.shape[1] + 1, dtype=float)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


def test_composite_sequential_propagates_empty_selection():
    X = pd.DataFrame({"甲": [1.0, 1.0, 1.0], "乙": [2.0, 2.0, 2.0]})

    selector = CompositeFeatureSelector([VarianceSelector(threshold=0.0)], strategy="sequential", n_jobs=1).fit(X)

    assert selector.selected_features_ == []
    assert selector.transform(X).shape == (3, 0)


def test_include_keeps_original_input_order():
    X = pd.DataFrame({"甲": [1.0, 1.0, 1.0], "乙": [0.0, 1.0, 2.0], "丙": [2.0, 3.0, 4.0]})

    selector = VarianceSelector(threshold=0.0, include=["甲"]).fit(X)

    assert selector.selected_features_ == ["甲", "乙", "丙"]


def test_transform_validates_schema_and_preserves_ndarray_type():
    X = pd.DataFrame({"甲": [0.0, 1.0], "乙": [1.0, 1.0]})
    selector = VarianceSelector(threshold=0.0).fit(X)

    with pytest.raises(ValidationError, match="缺少"):
        selector.transform(pd.DataFrame({"甲": [2.0]}))

    transformed = selector.transform(X.to_numpy())
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (2, 1)
    assert selector.get_support().tolist() == [True, False]
    assert selector.get_support(indices=True).tolist() == [0]
    assert selector.get_feature_names_out().tolist() == ["甲"]


def test_constructors_only_store_public_parameters():
    concrete = []
    for name in selectors.__all__:
        cls = getattr(selectors, name)
        if (
            isinstance(cls, type)
            and issubclass(cls, BaseFeatureSelector)
            and cls not in {BaseFeatureSelector, CompositeFeatureSelector}
        ):
            required = {
                "RegexSelector": {"pattern": "甲"},
                "FeatureImportanceSelector": {"estimator": CustomImportanceClassifier()},
                "NullImportanceSelector": {"estimator": CustomImportanceClassifier()},
                "RFESelector": {"estimator": CustomImportanceClassifier()},
                "SequentialFeatureSelector": {"estimator": CustomImportanceClassifier()},
            }
            concrete.append(cls(**required.get(name, {})))

    assert all("method_name" not in selector.__dict__ for selector in concrete)
    assert BaseFeatureSelector.mro().index(TransformerMixin) < BaseFeatureSelector.mro().index(BaseEstimator)


@pytest.mark.parametrize(
    "selector",
    [
        Chi2Selector(k=0),
        FTestSelector(k=0),
        FeatureImportanceSelector(CustomImportanceClassifier(), threshold=0),
    ],
)
def test_top_k_zero_is_rejected(selector):
    X = pd.DataFrame({"甲": [0, 1, 0, 1], "乙": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])

    with pytest.raises((ValueError, ValidationError), match="大于"):
        selector.fit(X, y)


@pytest.mark.parametrize("selector", [Chi2Selector(k=1, threshold=1e9), FTestSelector(k=1, threshold=1e9)])
def test_top_k_and_threshold_are_both_enforced(selector):
    X = pd.DataFrame({"甲": np.arange(8), "乙": [1, 4, 2, 5, 3, 0, 7, 6]})
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])

    assert selector.fit(X, y).selected_features_ == []


def test_feature_importance_honors_custom_getter_and_preserves_input_order():
    X = pd.DataFrame({"甲": [0, 1, 0, 1], "乙": [1, 0, 1, 0], "丙": [0, 0, 1, 1]})
    y = np.array([0, 1, 0, 1])

    selector = FeatureImportanceSelector(
        CustomImportanceClassifier(),
        threshold=2,
        importance_getter="custom_importances_",
    ).fit(X, y)

    assert selector.scores_.tolist() == [1.0, 2.0, 3.0]
    assert selector.selected_features_ == ["乙", "丙"]


def test_multiclass_coefficients_use_every_class():
    estimator = SimpleNamespace(coef_=np.array([[1.0, 0.0], [0.0, 4.0], [0.0, 3.0]]))

    assert get_feature_importances(estimator).tolist() == pytest.approx([1.0, 5.0])


def test_mode_dropna_excludes_missing_values_from_denominator():
    X = pd.DataFrame({"字段": [1.0] * 4 + [np.nan] * 6})

    drop_missing = ModeSelector(threshold=0.9, dropna=True).fit(X)
    count_missing = ModeSelector(threshold=0.9, dropna=False).fit(X)

    assert drop_missing.scores_["字段"] == pytest.approx(1.0)
    assert drop_missing.selected_features_ == []
    assert count_missing.scores_["字段"] == pytest.approx(0.6)
    assert count_missing.selected_features_ == ["字段"]


def test_mutual_info_treats_categorical_codes_as_discrete():
    y = np.array([0, 1] * 50)
    X = pd.DataFrame({"类别": pd.Categorical(np.where(y == 1, "坏", "好"))})

    selector = MutualInfoSelector(threshold=0.6, n_jobs=1, random_state=7).fit(X, y)

    assert selector.scores_["类别"] == pytest.approx(np.log(2), abs=1e-6)
    assert selector.selected_features_ == ["类别"]


def test_iv_uses_number_of_bins_in_additive_smoothing():
    x = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([0, 0, 0, 1, 1, 1])

    assert _compute_iv_single(x, y, regularization=1.0) == pytest.approx(0.7324081924454066)


def test_psi_detects_out_of_range_and_missing_rate_drift():
    shifted = _compute_psi_single(np.arange(100.0), np.arange(100.0) + 1000.0)
    missing_shift = _compute_psi_single(
        np.array([1.0, 1.0, 1.0, np.nan]),
        np.array([1.0, np.nan, np.nan, np.nan]),
    )

    assert shifted > 1.0
    assert missing_shift > 0.1


def test_psi_selector_supports_real_oot_and_categorical_values():
    train = pd.DataFrame({"数值": np.arange(100.0), "类别": ["甲"] * 90 + ["乙"] * 10})
    oot = pd.DataFrame({"数值": np.arange(100.0) + 1000.0, "类别": ["甲"] * 10 + ["乙"] * 90})

    selector = PSISelector(threshold=0.25, oot_df=oot, n_jobs=1).fit(train)

    assert selector.scores_["数值"] > 0.25
    assert selector.scores_["类别"] > 0.25
    assert selector.selected_features_ == []


@pytest.mark.parametrize("psi_bins", [True, 1, 2.5])
def test_psi_selector_validates_bin_count(psi_bins):
    X = pd.DataFrame({"A": np.arange(10, dtype=float)})

    with pytest.raises(ValueError, match="psi_bins"):
        PSISelector(psi_bins=psi_bins, n_jobs=1).fit(X)


def test_psi_selector_validates_oot_dataframe_type():
    X = pd.DataFrame({"A": np.arange(10, dtype=float)})

    with pytest.raises(ValidationError, match="DataFrame"):
        PSISelector(oot_df={"A": np.arange(10)}, n_jobs=1).fit(X)


def test_stability_selector_rejects_missing_oot_columns():
    X = pd.DataFrame({"甲": [0, 0, 1, 1], "乙": [0, 1, 0, 1]})
    y = np.array([0, 0, 1, 1])

    with pytest.raises(ValidationError, match="缺少"):
        StabilityAwareSelector(oot_df=pd.DataFrame({"甲": [0, 1]}), n_jobs=1).fit(X, y)


def test_vif_detects_affine_collinearity():
    base = np.linspace(0.0, 1.0, 100)
    X = pd.DataFrame({"原值": base, "平移值": base + 1000.0, "噪声": np.sin(base * 17)})

    selector = VIFSelector(threshold=5.0, n_jobs=1).fit(X)

    assert len({"原值", "平移值"}.intersection(selector.selected_features_)) == 1
    assert "噪声" in selector.selected_features_


def test_lift_only_rewards_the_requested_direction():
    X = pd.DataFrame({"风险值": np.arange(100.0), "安全值": -np.arange(100.0)})
    y = np.array([1] * 10 + [0] * 90)

    bad = LiftSelector(direction="bad", ratio=0.1, threshold=0.5, n_jobs=1).fit(X[["风险值"]], y)
    good = LiftSelector(direction="good", ratio=0.1, threshold=0.5, n_jobs=1).fit(X[["安全值"]], y)

    assert bad.scores_["风险值"] == pytest.approx(0.0)
    assert bad.selected_features_ == []
    assert good.scores_["安全值"] == pytest.approx(1.0)
    assert good.selected_features_ == ["安全值"]


def test_backward_stepwise_enforces_max_features_as_final_cap():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=list("ABCDE"))
    y = (X["A"] + 0.5 * X["B"] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)

    selector = StepwiseSelector(
        direction="backward",
        criterion="aic",
        max_features=2,
        max_iter=20,
        n_jobs=1,
    ).fit(X, y)

    assert len(selector.selected_features_) <= 2


def test_backward_stepwise_max_features_is_hard_cap_with_one_iteration():
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("ABCDEF"))
    y = (X["A"] + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)

    selector = StepwiseSelector(
        direction="backward",
        criterion="aic",
        max_features=2,
        max_iter=1,
        n_jobs=1,
    ).fit(X, y)

    assert len(selector.selected_features_) <= 2


def test_stepwise_rejects_boolean_max_features():
    X = pd.DataFrame({"A": [0.0, 1.0, 2.0], "B": [1.0, 0.0, 1.0]})
    y = pd.Series([0, 1, 1])

    with pytest.raises(ValueError, match="布尔值"):
        StepwiseSelector(max_features=True, n_jobs=1).fit(X, y)


def test_stepwise_rejects_unknown_criterion():
    X = pd.DataFrame({"甲": [0.0, 1.0, 0.0, 1.0]})
    y = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="criterion"):
        StepwiseSelector(criterion="unknown", n_jobs=1).fit(X, y)


def test_scorecard_does_not_reuse_iv_weights_for_ks_correlation():
    X = pd.DataFrame({"甲": [0, 1], "乙": [1, 0]})
    selector = ScorecardFeatureSelection(corr_metric="ks")

    weights = selector._resolve_corr_weights(X, np.array([0, 1]), pd.Series({"甲": 1.0, "乙": 0.5}))

    assert weights is None


def test_corr_default_binning_config_cannot_cross_contaminate_instances():
    first = CorrSelector()
    second = CorrSelector()
    original = DEFAULT_BINNING_PARAMS["method"]
    mutated = False
    try:
        try:
            first.binning_params["method"] = "cart"
            mutated = True
        except TypeError:
            pass
        if mutated:
            assert second.binning_params["method"] == original
    finally:
        if mutated:
            first.binning_params["method"] = original


def test_corr_treats_explicit_default_valued_dict_as_explicit(monkeypatch):
    called = []

    def record_apply(self, X, y=None):
        called.append(True)
        return X

    monkeypatch.setattr(CorrSelector, "_apply_binner", record_apply)
    X = pd.DataFrame({"甲": [0.0, 1.0, 2.0], "乙": [0.0, 1.0, 2.0]})

    CorrSelector(
        binning_params=dict(DEFAULT_BINNING_PARAMS),
        weights={"甲": 1.0, "乙": 0.5},
        n_jobs=1,
    ).fit(X)

    assert called == [True]


def test_boruta_exposes_statistical_decisions_across_all_iterations():
    rng = np.random.RandomState(7)
    signal = rng.normal(size=240)
    X = pd.DataFrame({"信号": signal, "噪声": rng.normal(size=240)})
    y = (signal > 0).astype(int)

    selector = BorutaSelector(
        RandomForestClassifier(n_estimators=30, random_state=3),
        max_iter=20,
        n_jobs=1,
    ).fit(X, y)

    assert len(selector.history_) == 20
    assert selector.hits_["信号"] > selector.hits_["噪声"]
    assert selector.p_values_.index.tolist() == ["信号", "噪声"]
    assert "信号" in selector.selected_features_


def test_empty_report_summary_is_printable(capsys):
    collector = SelectionReportCollector()

    collector.print_summary()

    assert "无筛选记录" in capsys.readouterr().out


def test_feature_trace_preserves_first_appearance_order():
    collector = SelectionReportCollector()
    collector.reports = [
        {
            "stage_name": "阶段1",
            "筛选器": "测试",
            "选中特征": ["乙", "甲"],
            "剔除特征": ["丙"],
            "特征得分": {},
            "剔除原因": ["测试"],
        }
    ]

    assert collector.get_feature_trace()["特征"].tolist() == ["乙", "甲", "丙"]
