"""筛选报告快照、目标列输出约定和组合执行状态的公开接口回归。"""

import inspect
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hscredit.core import selectors
from hscredit.core.selectors import (
    BaseFeatureSelector,
    CompositeFeatureSelector,
    NullSelector,
    ScorecardFeatureSelection,
    SelectionReportCollector,
    VarianceSelector,
    collect_selection_report,
)
from hscredit.core.selectors.reporting import DETAIL_COLUMNS, SUMMARY_COLUMNS
from hscredit.exceptions import NotFittedError, ValidationError


def test_every_public_selector_exposes_explicit_target_policy():
    for name in selectors.__all__:
        cls = getattr(selectors, name)
        if inspect.isclass(cls) and issubclass(cls, BaseFeatureSelector):
            parameters = inspect.signature(cls.__init__).parameters
            assert parameters["target_rm"].default is False
            assert "passthrough_target" not in parameters


PUBLIC_SELECTOR_CLASSES = [
    getattr(selectors, name)
    for name in selectors.__all__
    if inspect.isclass(getattr(selectors, name))
    and issubclass(getattr(selectors, name), BaseFeatureSelector)
    and getattr(selectors, name) is not BaseFeatureSelector
]


def _target_policy_selector(kind, **options):
    """让所有公开筛选器以小数据验证同一输出约定，不依赖实际入选阈值。"""
    kwargs = {"target": "目标", "include": ["甲", "乙"], "n_jobs": 1}
    if kind in {
        selectors.FeatureImportanceSelector,
        selectors.NullImportanceSelector,
        selectors.RFESelector,
        selectors.SequentialFeatureSelector,
    }:
        kwargs["estimator"] = LogisticRegression()
    if kind in {selectors.RFESelector, selectors.SequentialFeatureSelector}:
        kwargs["n_features_to_select"] = 2
    if kind is selectors.RegexSelector:
        kwargs["pattern"] = "甲|乙"
    if kind is selectors.CorrSelector:
        kwargs.update(weights={"甲": 2, "乙": 1}, binning_params=None)
    if kind is selectors.StepwiseSelector:
        kwargs.update(estimator="ols", max_features=2)
    if kind is CompositeFeatureSelector:
        kwargs["selectors"] = [("缺失", NullSelector(target="目标", n_jobs=1))]
    kwargs.update(options)
    return kind(**kwargs)


@pytest.fixture
def target_policy_frame():
    rng = np.random.default_rng(183)
    return pd.DataFrame(
        {"甲": rng.normal(size=40), "乙": rng.normal(size=40), "目标": np.tile([0, 1], 20)},
        index=np.arange(100, 140),
    )


@pytest.mark.parametrize("kind", PUBLIC_SELECTOR_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("target_rm", [False, True])
@pytest.mark.parametrize("fit_style", ["embedded", "separate"])
def test_all_selector_target_output_contracts(kind, target_rm, fit_style, target_policy_frame):
    frame = target_policy_frame
    X = frame.drop(columns="目标")
    # False 不显式传入，用实际默认参数覆盖用户所要求的默认行为。
    selector = _target_policy_selector(kind, **({"target_rm": True} if target_rm else {}))
    assert selector.target_rm is target_rm
    assert clone(selector).target_rm is target_rm
    if fit_style == "embedded":
        transformed = selector.fit_transform(frame)
    else:
        selector.fit(X, frame["目标"])
        transformed = selector.transform(frame)
    expected_columns = ["甲", "乙"] + ([] if target_rm else ["目标"])
    pd.testing.assert_frame_equal(transformed, frame[expected_columns])
    pd.testing.assert_frame_equal(selector.transform(X), X)
    assert selector.selected_features_ == ["甲", "乙"]
    assert selector.feature_names_in_.tolist() == ["甲", "乙"]
    assert selector.n_features_in_ == 2
    assert selector.get_support().tolist() == [True, True]
    assert "目标" not in collect_selection_report(selector).details["特征"].tolist()
    expected_names = ["甲", "乙"] + (["目标"] if fit_style == "embedded" and not target_rm else [])
    assert selector.get_feature_names_out().tolist() == expected_names
    assert selector.get_feature_names_out(frame.columns).tolist() == expected_columns
    selector.set_params(target_rm=not target_rm)
    assert ("目标" in selector.transform(frame)) is target_rm
    assert clone(selector).target_rm is not target_rm


@pytest.mark.parametrize("kind", PUBLIC_SELECTOR_CLASSES, ids=lambda cls: cls.__name__)
def test_removed_passthrough_parameter_is_not_a_selector_option(kind):
    selector = _target_policy_selector(kind)
    assert "passthrough_target" not in inspect.signature(kind.__init__).parameters
    assert "passthrough_target" not in selector.get_params()
    assert not hasattr(selector, "passthrough_target")
    with pytest.raises(TypeError, match="passthrough_target"):
        _target_policy_selector(kind, passthrough_target=False)
    with pytest.raises(ValueError, match="passthrough_target"):
        selector.set_params(passthrough_target=False)


@pytest.mark.parametrize("target_rm", [None, 0, 1, "yes", "False", []])
def test_target_removal_requires_boolean_on_fit_and_runtime(target_rm, target_policy_frame):
    with pytest.raises(ValidationError, match="target_rm"):
        NullSelector(target="目标", target_rm=target_rm).fit(target_policy_frame)
    selector = NullSelector(target="目标").fit(target_policy_frame)
    selector.set_params(target_rm=target_rm)
    with pytest.raises(ValidationError, match="target_rm"):
        selector.transform(target_policy_frame)
    with pytest.raises(ValidationError, match="target_rm"):
        selector.get_feature_names_out()


@pytest.mark.parametrize("kind", [NullSelector, CompositeFeatureSelector, ScorecardFeatureSelection])
@pytest.mark.parametrize("fit_has_target", [False, True])
@pytest.mark.parametrize("transform_has_target", [False, True])
@pytest.mark.parametrize("target_rm", [False, True])
@pytest.mark.parametrize("configuration", ["selector", "global"])
def test_pandas_output_uses_current_target_without_mutating_fit_metadata(
    kind, fit_has_target, transform_has_target, target_rm, configuration, target_policy_frame
):
    frame = target_policy_frame
    features = frame.drop(columns="目标")
    selector = _target_policy_selector(kind, target_rm=target_rm)
    if configuration == "selector":
        selector.set_output(transform="pandas")
    with config_context(transform_output="pandas" if configuration == "global" else "default"):
        if fit_has_target:
            selector.fit(frame)
        else:
            selector.fit(features, frame["目标"])
        fitted_names = selector.get_feature_names_out().copy()
        fitted_input = list(selector.input_columns_at_fit_)
        fitted_report = selector.get_selection_details()
        transformed = selector.transform(frame if transform_has_target else features)
        expected_columns = ["甲", "乙"] + (["目标"] if transform_has_target and not target_rm else [])
        pd.testing.assert_frame_equal(transformed, frame[expected_columns])
        # 同一对象交替处理有／无标签的表格，不改写拟合状态或报告快照。
        selector.transform(features if transform_has_target else frame)
        np.testing.assert_array_equal(selector.get_feature_names_out(), fitted_names)
        assert selector.input_columns_at_fit_ == fitted_input
        pd.testing.assert_frame_equal(selector.get_selection_details(), fitted_report)


@pytest.mark.parametrize("configuration", ["selector", "global"])
def test_pandas_output_converts_ndarray_and_survives_transactional_fit(configuration):
    values = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    selector = NullSelector(n_jobs=1)
    if configuration == "selector":
        selector.set_output(transform="pandas")
    with config_context(transform_output="pandas" if configuration == "global" else "default"):
        result = selector.fit_transform(values)
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == selector.get_feature_names_out().tolist()
        np.testing.assert_array_equal(result.to_numpy(), values)
        if configuration == "selector":
            assert clone(selector)._sklearn_output_config == {"transform": "pandas"}
        selector.set_output(transform="default")
        assert isinstance(selector.transform(values), np.ndarray)


def test_pipeline_target_is_not_an_input_to_native_model():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"x": rng.normal(size=100), "目标": rng.integers(0, 2, 100)})
    pipeline = Pipeline(
        [("筛选", NullSelector(target="目标", target_rm=True, n_jobs=1)), ("模型", LogisticRegression())]
    )
    pipeline.fit(frame, frame["目标"])
    assert pipeline[-1].feature_names_in_.tolist() == ["x"]
    assert len(pipeline.predict(frame[["x"]])) == len(frame)
    report = collect_selection_report(pipeline)
    assert report.details["特征"].tolist() == ["x"]
    default = NullSelector(target="目标").fit(frame)
    assert default.transform(frame).columns.tolist() == ["x", "目标"]
    assert clone(default).target_rm is False


def test_report_snapshot_does_not_mutate_model_and_keeps_fitted_threshold():
    frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, np.nan, 3, np.nan]})
    selector = NullSelector(threshold=0.3, n_jobs=1).fit(frame)
    legacy = selector.get_selection_report()
    assert legacy["得分统计"]["中位数"] == 0.25
    legacy["选中特征"].clear()
    selector.get_dropped_df().loc[:, "特征"] = "已修改"
    details = selector.get_selection_details()
    details.loc[:, "特征"] = "已修改"
    whole = selector.get_selection_result()
    metadata = whole.metadata
    metadata["阶段快照"].clear()
    assert selector.selected_features_ == ["a"]
    assert selector.get_dropped_df()["特征"].tolist() == ["b"]
    selector.set_params(threshold=0.9)
    assert selector.get_selection_report()["阈值"] == 0.3
    assert selector.get_selection_details()["有效阈值"].tolist() == [0.3, 0.3]
    assert selector.get_selection_result().metadata["阶段快照"]


def test_forced_only_report_has_all_features_and_fixed_schema():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    selector = NullSelector(include=["a"], exclude=["b"], n_jobs=1).fit(frame)
    details = selector.get_selection_report_df()
    assert details.columns.tolist() == DETAIL_COLUMNS
    assert details["特征"].tolist() == ["a", "b"]
    assert details["评估状态"].tolist() == ["未计算", "未计算"]
    assert details["决策来源"].tolist() == ["强制保留", "强制剔除"]
    assert details["指标值"].isna().all()
    assert selector.get_scores_df()["特征"].tolist() == ["a", "b"]
    assert selector.get_selection_report_df(kind="summary").shape == (1, 7)
    with pytest.raises(NotFittedError):
        NullSelector().get_selection_report_df()


def test_conflicting_forced_operations_report_effective_decision():
    selector = NullSelector(include=["a"], exclude=["a"]).fit(pd.DataFrame({"a": [1, 2], "b": [2, 3]}))
    row = selector.get_selection_details().set_index("特征").loc["a"]
    assert row["决策来源"] == "强制剔除"
    assert "覆盖" in row["筛选原因"]
    assert "a" not in selector.get_selection_report()["强制操作"]["强制保留"]


def test_composite_parent_forced_operations_and_mixed_labels():
    frame = pd.DataFrame({1: [1.0, 2.0, 3.0], "b": [4.0, 4.0, 4.0], "a": [5.0, 6.0, 7.0]})
    selector = CompositeFeatureSelector([("内层", VarianceSelector(n_jobs=1))], include=["b"], exclude=["a"]).fit(frame)
    root = selector.get_selection_details().query("阶段路径 == 'selector'")
    assert root["特征"].tolist() == [1, "b", "a"]
    assert selector.feature_names_in_.tolist() == [1, "b", "a"]
    assert selector.get_feature_names_out().tolist() == [1, "b"]
    assert selector.get_support().tolist() == [True, True, False]
    assert collect_selection_report(selector).metadata["完整"]


@pytest.mark.parametrize("kwargs", [{"strategy": "typo"}, {"selectors": []}])
def test_invalid_composite_configuration_even_if_forced_only(kwargs):
    options = dict(selectors=[NullSelector()], include=["a"])
    options.update(kwargs)
    with pytest.raises(ValidationError):
        CompositeFeatureSelector(**options).fit(pd.DataFrame({"a": [1, 2]}))


def test_composite_empty_upstream_is_legitimate_skipped_stage():
    selector = CompositeFeatureSelector([NullSelector(threshold=0, n_jobs=1), VarianceSelector(n_jobs=1)]).fit(
        pd.DataFrame({"a": [1, 2]})
    )
    assert selector.executed_stages_ == [0]
    assert selector.skipped_stages_ == {1: "上游无剩余特征"}
    report = collect_selection_report(selector)
    assert report.metadata["完整"]
    assert report.summary.iloc[-1]["执行状态"] == "未执行"
    assert "[SUMMARY]" not in report.details["特征"].tolist()


def test_scorecard_disabled_and_forced_stages_preserve_execution_status():
    frame = pd.DataFrame({"a": [1, 2], "目标": [0, 1]})
    selector = ScorecardFeatureSelection(
        null_threshold=None, iv_threshold=None, corr_threshold=None, mode_threshold=None, target="目标"
    ).fit(frame)
    assert selector.transform(frame).columns.tolist() == ["a", "目标"]
    assert selector.select_columns == ["a", "目标"]
    report = collect_selection_report(selector)
    assert len(report.summary) == 5
    assert report.summary.iloc[1:]["执行状态"].tolist() == ["未执行"] * 4
    forced = ScorecardFeatureSelection(include=["a"]).fit(frame[["a"]])
    assert forced.selected_features_ == ["a"]
    assert collect_selection_report(forced).metadata["完整"]


def test_collector_independent_sources_do_not_invent_final_set():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0], "c": [4.0, 4.0]})
    first, second = [VarianceSelector(n_jobs=1).fit(frame) for _ in range(2)]
    collector = SelectionReportCollector().add_report(first).add_report(second)
    assert collector.get_summary()["最终特征数"] is None
    assert collector.get_summary()["累计剔除特征数"] is None
    assert collector.to_dataframe().columns.tolist() == SUMMARY_COLUMNS
    assert len(collector.get_details()) == 6
    collector.reports[0]["选中特征"].clear()
    assert first.selected_features_ == ["a", "b"]
    first.fit(frame[["a"]])
    assert len(collector.get_details()) == 6


def test_collector_accepts_pipeline_and_declared_sequential_list_without_refit():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 3.0]})
    pipeline = Pipeline([("缺失", NullSelector(n_jobs=1)), ("方差", VarianceSelector(n_jobs=1))]).fit(frame)
    with patch.object(pipeline[0], "fit", side_effect=AssertionError("读取报告不能fit")):
        collector = SelectionReportCollector(source=pipeline)
        assert collector.get_summary()["最终特征数"] == 1
        assert len(collector.get_details()) == 4
    sequential = SelectionReportCollector(relation="sequential").add_report(pipeline[0]).add_report(pipeline[1])
    assert sequential.get_summary()["累计剔除特征数"] == 1


def test_collector_rejects_unfitted_by_default():
    with pytest.raises(NotFittedError):
        SelectionReportCollector().add_report(NullSelector())
    partial = SelectionReportCollector(source=NullSelector(), strict=False)
    assert not partial.get_report().metadata["完整"]


def test_tuple_feature_identifiers_remain_one_dimensional_and_reports_frozen():
    frame = pd.DataFrame([[1.0, np.nan], [2.0, np.nan]], columns=[("组", "a"), ("组", "b")])
    selector = NullSelector(n_jobs=1).fit(frame)
    assert selector.feature_names_in_.shape == (2,)
    assert selector.get_feature_names_out().shape == (1,)
    assert selector.get_feature_names_out(frame.columns).tolist() == [("组", "a")]
    assert selector.transform(frame).columns.tolist() == [("组", "a")]
    assert selector.get_scores_df()["特征"].tolist() == list(frame.columns)
    selector.selected_features_.clear()
    selector.dropped_.loc[:, "剔除原因"] = "外部改写"
    assert selector.get_scores_df()["状态"].tolist() == ["选中", "剔除"]
    assert selector.get_dropped_df()["剔除原因"].iloc[0] != "外部改写"


def test_history_budget_is_validated_for_forced_only_model_selector():
    from hscredit.core.selectors import VIFSelector

    with pytest.raises(ValueError, match="report_history"):
        VIFSelector(include=["a"], report_history="typo").fit(pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(ValueError, match="regularization"):
        ScorecardFeatureSelection(include=["a"], iv_regularization=0).fit(pd.DataFrame({"a": [1, 2]}))


def test_rfe_fixed_budget_marks_unsearched_features_as_not_computed():
    from hscredit.core.selectors import RFESelector

    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [2.0, 1.0, 3.0, 0.0]})
    selector = RFESelector(LogisticRegression(), n_features_to_select=1, include=["a"], n_jobs=1).fit(
        frame, [0, 0, 1, 1]
    )
    detail = selector.get_selection_details().set_index("特征")
    assert detail.loc["b", "评估状态"] == "未计算"
    assert pd.isna(detail.loc["b", "指标值"])
