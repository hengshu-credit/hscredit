"""规则挖掘字段选择与字段含义报告契约测试。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.report.mining import (
    DecisionTreeAnalyzer,
    ManualTreeExtractor,
    MultiFeatureRuleMiner,
    SingleFeatureRuleMiner,
    TreeRuleExtractor,
)


@pytest.fixture
def feature_data():
    """构造具有稳定树分裂和规则提升度的小型数据集。"""
    score = np.arange(60, dtype=float)
    debt = np.tile(np.arange(6, dtype=float), 10)
    return pd.DataFrame(
        {
            "score": score,
            "debt": debt,
            "noise": np.sin(score),
            "target": (score >= 30).astype(int),
        }
    )


@pytest.fixture
def mixed_feature_data(feature_data):
    """将类别字段放在数值字段前，便于验证默认与显式顺序契约。"""
    return pd.DataFrame(
        {
            "category": np.where(feature_data["debt"].mod(2).eq(0), "A", "B"),
            "score": feature_data["score"],
            "debt": feature_data["debt"],
            "target": feature_data["target"],
        }
    )


def _single(**kwargs):
    return SingleFeatureRuleMiner(
        method="quantile",
        max_n_bins=3,
        min_samples=1,
        min_lift=0.0,
        n_jobs=1,
        **kwargs,
    )


def _multi(**kwargs):
    return MultiFeatureRuleMiner(
        method="quantile",
        max_n_bins=3,
        min_samples=1,
        min_lift=0.0,
        n_jobs=1,
        **kwargs,
    )


def _tree(**kwargs):
    return TreeRuleExtractor(
        algorithm="dt",
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        test_size=0.2,
        n_jobs=1,
        **kwargs,
    )


def _analyzer(**kwargs):
    return DecisionTreeAnalyzer(tree_params={"max_depth": 2, "random_state": 0}, n_jobs=1, **kwargs)


def _manual(**kwargs):
    return ManualTreeExtractor(
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=1,
        **kwargs,
    )


MINER_CASES = [
    pytest.param(_single, "features_", id="single-feature"),
    pytest.param(_multi, "features_", id="multi-feature"),
    pytest.param(_tree, "feature_names_", id="tree-rule"),
    pytest.param(_analyzer, "features_", id="decision-tree-analyzer"),
    pytest.param(_manual, "_feature_list", id="manual-tree"),
]


@pytest.mark.parametrize("factory,state_attribute", MINER_CASES)
@pytest.mark.parametrize(
    "features,expected",
    [
        pytest.param("score", ["score"], id="str"),
        pytest.param(["debt", "score"], ["debt", "score"], id="list"),
    ],
)
def test_constructor_features_selects_only_requested_fields(
    feature_data, factory, state_attribute, features, expected
):
    """构造参数 features 若未生效，拟合状态会混入 noise 字段。"""
    miner = factory(features=features).fit(feature_data)

    assert getattr(miner, state_attribute) == expected


@pytest.mark.parametrize("factory,state_attribute", MINER_CASES)
@pytest.mark.parametrize(
    "feature_names,expected",
    [
        pytest.param("debt", ["debt"], id="str"),
        pytest.param(["debt", "score"], ["debt", "score"], id="list"),
    ],
)
def test_fit_feature_names_accepts_str_and_overrides_constructor(
    feature_data, factory, state_attribute, feature_names, expected
):
    """fit 的 feature_names 若未覆盖构造配置，最终仍会使用构造器中的 score 字段。"""
    miner = factory(features=["score"]).fit(feature_data, feature_names=feature_names)

    assert getattr(miner, state_attribute) == expected


@pytest.mark.parametrize("factory,state_attribute", MINER_CASES)
def test_missing_requested_feature_raises_clear_error(feature_data, factory, state_attribute):
    """不存在字段若未提前校验，会在 pandas/sklearn 深处产生不明确异常。"""
    del state_attribute

    with pytest.raises(ValueError, match="缺少必需的特征.*missing_field"):
        factory(features="missing_field").fit(feature_data)


def test_single_feature_default_order_remains_numeric_then_categorical(mixed_feature_data):
    """features=None 若改用原始列序，会破坏既有数值字段优先的结果顺序。"""
    miner = _single().fit(mixed_feature_data)

    assert miner.features_ == ["score", "debt", "category"]


def test_explicit_mixed_feature_order_drives_multi_feature_pair_order(mixed_feature_data):
    """显式字段顺序若被类型分组覆盖，首个交叉组合将错误地变成 score × debt。"""
    miner = _multi(
        features=["category", "score", "debt"],
        feature_map={"category": "客户类别", "score": "信用评分", "debt": "负债水平"},
    ).fit(mixed_feature_data)

    report = miner.get_all_cross_rules(top_n=1, max_feature_pairs=1, min_lift=0.0, min_samples=1)

    assert not report.empty
    assert report["特征组合"].eq("category × score").all()
    assert report["入参字段"].eq("category × score").all()


def test_single_feature_report_includes_feature_meaning(feature_data):
    """单特征规则报告若漏掉映射，业务人员无法识别入参字段含义。"""
    miner = _single(features="score", feature_map={"score": "信用评分"}).fit(feature_data)

    report = miner.get_top_rules(top_n=3, min_lift=0.0, min_samples=1)

    assert not report.empty
    assert report[["入参字段", "字段含义"]].drop_duplicates().to_dict("records") == [
        {"入参字段": "score", "字段含义": "信用评分"}
    ]


def test_single_feature_summary_includes_input_field_and_meaning(feature_data):
    """特征摘要若只有内部 feature 列，不满足报告的入参字段展示契约。"""
    miner = _single(features="score", feature_map={"score": "信用评分"}).fit(feature_data)

    report = miner.get_feature_summary()

    assert report[["入参字段", "字段含义"]].to_dict("records") == [
        {"入参字段": "score", "字段含义": "信用评分"}
    ]


def test_multi_feature_report_includes_each_feature_meaning(feature_data):
    """交叉规则报告若只显示表达式，会丢失两个入参字段的业务含义。"""
    miner = _multi(
        features=["score", "debt"],
        feature_map={"score": "信用评分", "debt": "负债水平"},
    ).fit(feature_data)

    report = miner.get_cross_rules("score", "debt", top_n=3, min_lift=0.0, min_samples=1)

    assert not report.empty
    assert report[["入参字段", "字段含义"]].drop_duplicates().to_dict("records") == [
        {"入参字段": "score × debt", "字段含义": "信用评分 × 负债水平"}
    ]


def test_tree_rule_dataframe_includes_used_feature_meanings(feature_data):
    """树规则 DataFrame 若不解析规则路径，就无法报告实际使用字段及含义。"""
    extractor = _tree(
        features=["score", "debt"],
        feature_map={"score": "信用评分", "debt": "负债水平"},
    ).fit(feature_data)

    report = extractor.get_rules_dataframe(top_n=10, min_samples=1)

    assert not report.empty
    assert {"入参字段", "字段含义"}.issubset(report.columns)
    assert report["入参字段"].str.contains("score", regex=False).all()
    assert report["字段含义"].str.contains("信用评分", regex=False).all()


def test_tree_feature_importance_includes_input_field_and_meaning(feature_data):
    """树重要性报告若只保留内部 feature 列，会漏掉统一的入参字段列。"""
    extractor = _tree(features=["score", "debt"], feature_map={"score": "信用评分"}).fit(feature_data)

    report = extractor.get_feature_importance()

    assert {"feature", "入参字段", "字段含义", "importance"}.issubset(report.columns)
    assert report["入参字段"].equals(report["feature"])
    assert report.loc[report["feature"].eq("score"), "字段含义"].eq("信用评分").all()


@pytest.mark.parametrize("factory", [_analyzer, _manual], ids=["decision-tree-analyzer", "manual-tree"])
def test_tree_node_report_supports_feature_map_override(feature_data, factory):
    """report 显式 feature_map 若未优先，会继续显示构造器中的旧字段含义。"""
    tree = factory(features="score", feature_map={"score": "旧含义"}).fit(feature_data)

    report = tree.report(feature_data, leaf_only=True, feature_map={"score": "信用评分"})

    assert not report.empty
    assert {"入参字段", "字段含义"}.issubset(report.columns)
    assert report["入参字段"].eq("score").all()
    assert report["字段含义"].eq("信用评分").all()


@pytest.mark.parametrize("factory", [_analyzer, _manual], ids=["decision-tree-analyzer", "manual-tree"])
def test_empty_feature_map_override_keeps_input_fields_with_blank_meanings(feature_data, factory):
    """显式空映射若按假值忽略，会错误回退或省略字段报告列。"""
    tree = factory(features="score", feature_map={"score": "旧含义"}).fit(feature_data)

    report = tree.report(feature_data, leaf_only=True, feature_map={})

    assert report["入参字段"].eq("score").all()
    assert report["字段含义"].eq("").all()


@pytest.mark.parametrize("factory", [_analyzer, _manual], ids=["decision-tree-analyzer", "manual-tree"])
def test_multi_label_tree_report_keeps_feature_meaning_in_detail_group(feature_data, factory):
    """多标签报告若把新增列放错层级，会破坏既有 MultiIndex 报告结构。"""
    data = feature_data.assign(MOB1=np.where(feature_data["target"].eq(1), 10, 0))
    tree = factory(features="score", feature_map={"score": "信用评分"}).fit(data)

    report = tree.report(data, overdue="MOB1", dpds=[7, 3], leaf_only=True)

    assert isinstance(report.columns, pd.MultiIndex)
    assert ("分箱详情", "入参字段") in report.columns
    assert ("分箱详情", "字段含义") in report.columns
    assert report[("分箱详情", "入参字段")].eq("score").all()
    assert report[("分箱详情", "字段含义")].eq("信用评分").all()


def test_decision_tree_analyzer_save_load_preserves_feature_context(feature_data, tmp_path):
    """模型持久化若漏掉新参数，加载后的报告会丢失字段顺序或字段含义。"""
    analyzer = _analyzer(
        features=["debt", "score"],
        feature_map={"debt": "负债水平", "score": "信用评分"},
    ).fit(feature_data)
    model_path = tmp_path / "decision_tree_analyzer.pkl"

    analyzer.save(str(model_path))
    restored = DecisionTreeAnalyzer.load(str(model_path))
    report = restored.report(feature_data, leaf_only=True)

    assert restored.features_ == ["debt", "score"]
    assert restored.feature_map == {"debt": "负债水平", "score": "信用评分"}
    assert {"入参字段", "字段含义"}.issubset(report.columns)
    assert report["字段含义"].str.contains("信用评分", regex=False).all()
