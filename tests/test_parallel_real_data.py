"""真实放款数据上的串行、线程与进程一致性验收。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from hscredit.core.binning import QuantileBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.rules import Rule
from hscredit.core.selectors import IVSelector
from hscredit.report import (
    ModelReport,
    feature_binning_summary,
    overdue_prediction_report,
    rule_group_compare,
    ruleset_analysis,
)
from hscredit.report.mining import SingleFeatureRuleMiner


pytestmark = pytest.mark.integration

WORKBOOK = Path(__file__).resolve().parents[1] / "examples" / "hscredit_yyp.xlsx"
FEATURES = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
TARGET = "FPD"
OVERDUE = ["MOB1"]
DPDS = [7, 3, 0]
AMOUNT = "放款金额"
DATE = "放款时间"
CATEGORY = "商品类别"
BACKENDS = ("threading", "loky")


@pytest.fixture(scope="session")
def yyp_data():
    """整个测试会话只读加载一次真实工作簿。"""
    data = pd.read_excel(WORKBOOK)
    required = FEATURES + [TARGET, *OVERDUE, AMOUNT, DATE, CATEGORY]
    assert data.shape[0] == 970
    assert not set(required).difference(data.columns)
    return data


@pytest.fixture(scope="session")
def yyp_model(yyp_data):
    """固定样本顺序和随机种子的真实数据模型。"""
    train = yyp_data.iloc[:650]
    return LogisticRegression(
        solver="liblinear", random_state=20260805, max_iter=500
    ).fit(train[FEATURES], train[TARGET])


def _assert_frame_exact(left, right):
    """显式核对轴、dtype、值与 NaN，不允许退化为 shape-only。"""
    pd.testing.assert_index_equal(left.index, right.index, exact=True)
    pd.testing.assert_index_equal(left.columns, right.columns, exact=True)
    assert left.dtypes.astype(str).tolist() == right.dtypes.astype(str).tolist()
    pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=True, check_like=False)


def _assert_nested_value_exact(left, right):
    if isinstance(left, pd.DataFrame):
        _assert_frame_exact(left, right)
    elif isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, check_exact=True, check_dtype=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_nested_value_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_value_exact(left_item, right_item)
    elif pd.isna(left) and pd.isna(right):
        return
    else:
        assert left == right


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_binner_exact_across_backends(yyp_data, backend):
    """数值和类别特征分箱的状态、表格与转换结果完全一致。"""
    columns = FEATURES + [CATEGORY]
    kwargs = dict(max_n_bins=5, min_n_bins=2, random_state=20260805)
    serial = QuantileBinning(n_jobs=1, **kwargs).fit(yyp_data[columns], yyp_data[TARGET])
    parallel = QuantileBinning(n_jobs=2, parallel_backend=backend, **kwargs).fit(
        yyp_data[columns], yyp_data[TARGET]
    )

    assert list(serial.splits_) == list(parallel.splits_) == columns
    _assert_nested_value_exact(serial.splits_, parallel.splits_)
    _assert_nested_value_exact(serial._cat_bins_, parallel._cat_bins_)
    for feature in columns:
        _assert_frame_exact(serial.get_bin_table(feature), parallel.get_bin_table(feature))
    _assert_frame_exact(serial.transform(yyp_data[columns]), parallel.transform(yyp_data[columns]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_selector_exact_across_backends(yyp_data, backend):
    """高维入口的特征顺序、IV 得分和筛选输出完全一致。"""
    serial = IVSelector(threshold=0.0, n_jobs=1).fit(yyp_data[FEATURES], yyp_data[TARGET])
    parallel = IVSelector(threshold=0.0, n_jobs=2, parallel_backend=backend).fit(
        yyp_data[FEATURES], yyp_data[TARGET]
    )

    assert serial.selected_features_ == parallel.selected_features_
    pd.testing.assert_series_equal(serial.scores_, parallel.scores_, check_exact=True, check_dtype=True)
    _assert_frame_exact(serial.transform(yyp_data[FEATURES]), parallel.transform(yyp_data[FEATURES]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_encoder_exact_across_backends(yyp_data, backend):
    """真实商品类别的映射、列顺序、dtype 与编码值完全一致。"""
    inputs = yyp_data[[CATEGORY, FEATURES[0]]]
    serial = WOEEncoder(cols=[CATEGORY], regularization=1.0, n_jobs=1).fit(inputs, yyp_data[TARGET])
    parallel = WOEEncoder(
        cols=[CATEGORY], regularization=1.0, n_jobs=2, parallel_backend=backend
    ).fit(inputs, yyp_data[TARGET])

    _assert_nested_value_exact(serial.export_mapping(), parallel.export_mapping())
    _assert_frame_exact(serial.transform(inputs), parallel.transform(inputs))


def _real_rules():
    return [
        Rule(f"`{FEATURES[0]}` < 0.05", name="鉴真低分"),
        Rule(f"`{FEATURES[1]}` > 55", name="非银多头"),
        Rule(f"`{CATEGORY}` == '礼包'", name="礼包类别"),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_rule_analysis_exact_across_backends(yyp_data, backend):
    """规则、DPD、金额、月份及商品分组的指标与顺序完全一致。"""
    common = dict(
        datasets=yyp_data,
        overdue=OVERDUE,
        dpds=DPDS,
        amount=AMOUNT,
    )
    serial = ruleset_analysis(rules=_real_rules(), n_jobs=1, **common)
    parallel = ruleset_analysis(
        rules=_real_rules(), n_jobs=2, parallel_backend=backend, **common
    )
    _assert_frame_exact(serial, parallel)

    # API 约定 date_col 与 group_col 互斥，因此分别覆盖月份和商品类别两种分组。
    for grouping in ({"date_col": DATE, "freq": "M"}, {"group_col": CATEGORY}):
        serial_group = rule_group_compare(
            yyp_data,
            _real_rules()[0],
            overdue=OVERDUE,
            dpds=DPDS,
            amount=AMOUNT,
            n_jobs=1,
            **grouping,
        )
        parallel_group = rule_group_compare(
            yyp_data,
            _real_rules()[0],
            overdue=OVERDUE,
            dpds=DPDS,
            amount=AMOUNT,
            n_jobs=2,
            parallel_backend=backend,
            **grouping,
        )
        _assert_frame_exact(serial_group, parallel_group)


def _normalized_mined_rules(miner):
    return [
        (rule.expr, rule.name, rule.description, rule.weight)
        for rule in miner.get_rules()
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_rule_miner_exact_across_backends(yyp_data, backend):
    """固定种子后规则表达式、顺序、指标表和分箱状态完全一致。"""
    data = yyp_data[FEATURES + [CATEGORY, TARGET]]
    kwargs = dict(
        target=TARGET,
        method="quantile",
        max_n_bins=4,
        min_samples=10,
        min_lift=0.0,
        random_state=20260805,
    )
    serial = SingleFeatureRuleMiner(n_jobs=1, **kwargs).fit(data)
    parallel = SingleFeatureRuleMiner(n_jobs=2, parallel_backend=backend, **kwargs).fit(data)

    assert list(serial.results_) == list(parallel.results_)
    for feature in serial.results_:
        _assert_frame_exact(serial.results_[feature], parallel.results_[feature])
    assert _normalized_mined_rules(serial) == _normalized_mined_rules(parallel)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_feature_report_exact_across_backends(yyp_data, backend):
    """真实三特征报告的嵌套表、摘要轴和所有值完全一致。"""
    kwargs = dict(
        data=yyp_data,
        feature=FEATURES,
        methods=["quantile", "uniform"],
        target=TARGET,
        max_n_bins=4,
        random_state=20260805,
    )
    serial_tables, serial_summary = feature_binning_summary(n_jobs=1, **kwargs)
    parallel_tables, parallel_summary = feature_binning_summary(
        n_jobs=2, parallel_backend=backend, **kwargs
    )

    assert list(serial_tables) == list(parallel_tables)
    for feature in serial_tables:
        assert list(serial_tables[feature]) == list(parallel_tables[feature])
        for method in serial_tables[feature]:
            _assert_frame_exact(serial_tables[feature][method], parallel_tables[feature][method])
    _assert_frame_exact(serial_summary, parallel_summary)


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_overdue_report_exact_across_backends(yyp_data, backend):
    """MOB1 三个 DPD 口径及预测汇总完全一致。"""
    kwargs = dict(
        data=yyp_data,
        feature=FEATURES[0],
        overdue=OVERDUE,
        dpds=DPDS,
        rules=[0.02, 0.04, 0.06, 0.08],
        predict_data=yyp_data.iloc[:80][FEATURES],
    )
    serial = overdue_prediction_report(n_jobs=1, **kwargs)
    parallel = overdue_prediction_report(n_jobs=2, parallel_backend=backend, **kwargs)
    _assert_frame_exact(serial, parallel)


def _model_report(model, data, n_jobs, backend=None):
    datasets = {
        "train": data.iloc[:650].copy(),
        "test": data.iloc[650:].copy(),
    }
    return ModelReport(
        model,
        datasets=datasets,
        feature_names=FEATURES,
        overdue=OVERDUE,
        dpds=DPDS,
        n_jobs=n_jobs,
        parallel_backend=backend,
        parallel_config={"batch_size": 1},
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_model_report_exact_across_backends(yyp_data, yyp_model, backend):
    """真实多数据集、多 DPD 模型摘要、指标和含金额分箱表完全一致。"""
    serial = _model_report(yyp_model, yyp_data, 1)
    parallel = _model_report(yyp_model, yyp_data, 2, backend)

    _assert_frame_exact(serial.summary(), parallel.summary())
    assert serial._label_names == parallel._label_names
    for label in serial._label_names:
        _assert_frame_exact(serial.get_metrics(label=label), parallel.get_metrics(label=label))
    _assert_frame_exact(
        serial.get_bin_table(max_n_bins=5, amount_col=AMOUNT, labels=serial._label_names),
        parallel.get_bin_table(max_n_bins=5, amount_col=AMOUNT, labels=parallel._label_names),
    )
