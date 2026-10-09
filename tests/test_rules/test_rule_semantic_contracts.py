"""审计回归：原始字段规则语义、目标隔离和有界统计。"""

import json

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning.spec import BinSpec
from hscredit.core.binning import OptimalBinning
from hscredit.core.rules import Rule
from hscredit.core.rules.accumulator import RuleAccumulator
from hscredit.core.rules.artifact import RuleArtifact
from hscredit.report.mining import MultiFeatureRuleMiner, MultiLabelRuleMiner, SingleFeatureRuleMiner, TreeRuleExtractor


@pytest.mark.parametrize(
    "miner", [SingleFeatureRuleMiner(n_jobs=1), MultiFeatureRuleMiner(n_jobs=1), TreeRuleExtractor(n_jobs=1)]
)
def test_miner_mixed_target_and_positional_alignment(miner):
    X = pd.DataFrame({"x": [0, 1, 2, 3], "target": [1, 1, 0, 0]}, index=["b", "b", "a", "c"])
    y = pd.Series([0, 1, 0, 1], index=[100, 101, 102, 103])
    features, labels = miner._check_input_data(X, y)
    assert features.columns.tolist() == ["x"]
    assert labels.index.tolist() == X.index.tolist()
    assert labels.tolist() == y.tolist()
    assert "target" in X


def test_tree_cannot_learn_explicit_target():
    X = pd.DataFrame({"noise": np.zeros(100), "target": np.arange(100) % 2})
    miner = TreeRuleExtractor(algorithm="dt", max_depth=2, n_jobs=1, random_state=42).fit(X, X.target)
    assert miner.feature_names_ == ["noise"]
    assert all("target" not in rule.feature_names_in_ for rule in miner.get_rules(min_samples=1))


@pytest.mark.parametrize("splits", [[], [1.5], [0.5, 1.5], [0.5, 1.5, 2.5]])
def test_single_feature_keeps_every_internal_split(splits):
    class Binner:
        def fit(self, X, y):
            self.splits_ = {"x": np.array(splits)}
            return self

    miner = SingleFeatureRuleMiner(min_samples=1, min_lift=0, n_jobs=1)
    miner._get_binning_instance = lambda: Binner()
    miner.fit(pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "target": [0, 1, 0, 1]}))
    assert len(miner.results_["x"]) == 2 * len(splits)
    if splits:
        assert set(miner.results_["x"].operator) == {"<=", ">="}
        assert set(miner.results_["x"].threshold) == set(splits)


def test_cross_rules_match_bin_masks_and_counts():
    X = pd.DataFrame(
        {
            "中文 字段": list(np.arange(80, dtype=float)) + [np.nan, -999],
            "类别": (["其他", "缺失", "a'b", "rare"] * 21)[:82],
        }
    )
    X["target"] = (np.arange(len(X)) % 3 == 0).astype(int)
    miner = MultiFeatureRuleMiner(max_n_bins=3, min_samples=1, min_lift=0, special_codes=[-999], n_jobs=1).fit(X)
    rules = miner.get_cross_rules("中文 字段", "类别", top_n=100, min_lift=0)
    assert len(rules) > 0
    count = 0
    for _, row in rules.iterrows():
        expected = (miner._prepared_features_["中文 字段"] == row["特征1箱号"]) & (
            miner._prepared_features_["类别"] == row["特征2箱号"]
        )
        actual = row["规则制品"].predict(X)
        np.testing.assert_array_equal(actual, expected)
        assert row["命中样本数"] == int(expected.sum())
        assert row["命中坏样本数"] == int(X.target.to_numpy()[expected.to_numpy()].sum())
        count += int(actual.sum())
    assert count == len(X)


def test_cross_shared_tasks_fit_each_numeric_column_once(monkeypatch):
    import hscredit.report.mining.multi_feature as module

    data = pd.DataFrame(np.random.RandomState(3).normal(size=(100, 4)), columns=list("abcd"))
    data["target"] = np.arange(100) % 2
    miner = MultiFeatureRuleMiner(max_n_bins=3, min_samples=1, min_lift=0, n_jobs=1).fit(data)
    original = module._multi_feature_pair_worker
    sources = []

    def record(task):
        sources.append((id(task[0]), id(task[0].X_)))
        return original(task)

    monkeypatch.setattr(module, "_multi_feature_pair_worker", record)
    miner.get_all_cross_rules(top_n=1, max_feature_pairs=6)
    assert len(sources) == 6
    assert len(set(sources)) == 1
    assert sources[0][1] == id(miner.X_)
    assert set(miner._binner_instances_) == set("abcd")
    old = {key: id(value) for key, value in miner._binner_instances_.items()}
    miner.get_all_cross_rules(top_n=1, max_feature_pairs=6)
    assert old == {key: id(value) for key, value in miner._binner_instances_.items()}


def test_multilabel_only_is_exclusive_and_max_rules_is_enforced():
    x = np.arange(100)
    data = pd.DataFrame({"x": x, "short": (x > 50).astype(int), "long": (x > 60).astype(int)})
    miner = MultiLabelRuleMiner(labels=["short", "long"], min_lift=0, max_rules=1, n_bins=5, n_jobs=1).fit(data)
    assert len(miner.get_rules(effectiveness="all")) == 1
    miner._rules = [{"规则": "x>0", "short_LIFT": 2, "long_LIFT": 2}, {"规则": "x>1", "short_LIFT": 2, "long_LIFT": 0}]
    assert miner.get_rules(effectiveness="short_only", min_lift_per_label=1).规则.tolist() == ["x>1"]
    assert miner.get_rules(effectiveness="long_only", min_lift_per_label=1).empty


def test_tree_raw_category_and_missing_numeric_leaf_equivalence():
    data = pd.DataFrame(
        {"类别": ["a", "b", None, "a'b"] * 30, "x": [np.nan, 1.0, -1.0, 2.0] * 30, "target": [0, 1, 0, 1] * 30}
    )
    miner = TreeRuleExtractor(
        algorithm="dt", max_depth=3, min_samples_leaf=1, min_samples_split=2, n_jobs=1, random_state=4
    ).fit(data)
    for item in miner.extract_rules():
        raw = data.loc[miner.X_train_.index]
        expected = miner._apply_conditions(item["conditions"], miner.X_train_)
        actual = Rule(miner._rule_to_string(item)).predict(raw)
        np.testing.assert_array_equal(actual, expected)


def test_tree_float32_rounding_boundary():
    data = pd.DataFrame({"x": [-1.0, -0.1, 0.1, 1.0] * 25, "target": [0, 0, 1, 1] * 25})
    miner = TreeRuleExtractor(
        algorithm="dt", max_depth=1, min_samples_leaf=1, min_samples_split=2, n_jobs=1, random_state=4
    ).fit(data)
    tree = miner.model_.get_native_model()
    boundary = float(tree.tree_.threshold[0])
    raw = pd.DataFrame({"x": [boundary, np.nextafter(boundary, np.inf), np.nextafter(boundary, -np.inf), np.nan]})
    for item in miner.extract_rules():
        transformed = raw.fillna(0).astype(np.float32)
        expected = miner._apply_conditions(item["conditions"], transformed)
        np.testing.assert_array_equal(Rule(miner._rule_to_string(item)).predict(raw), expected)


def test_bin_spec_artifact_boundary_roundtrip_and_version(tmp_path):
    spec = BinSpec("中文 字段", 2, lower=0.123456789, upper=None, exclude_values=(999,), include_missing=True)
    raw = pd.DataFrame({"中文 字段": [spec.lower, np.nextafter(spec.lower, -np.inf), np.nan, np.inf, 999]})
    artifact = RuleArtifact(spec.to_expression(), bins=(spec,), validation="原始掩码等价")
    expected = spec.mask(raw)
    np.testing.assert_array_equal(artifact.predict(raw), expected)
    assert spec.label(precision=2) != spec.label(precision=8)
    path = tmp_path / "rule.json"
    artifact.save(path)
    restored = RuleArtifact.load(path)
    np.testing.assert_array_equal(restored.predict(raw), expected)
    assert restored.bins == artifact.bins
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 999
    with pytest.raises(ValueError, match="版本"):
        RuleArtifact.from_dict(payload)


def test_rule_aggregate_merge_and_block_prediction():
    data = pd.DataFrame({"x": np.arange(23), "target": np.arange(23) % 2}, index=["重复"] * 23)
    rule = Rule("x >= 10")
    parts = list(rule.predict_batches(data, batch_size=4))
    np.testing.assert_array_equal(pd.concat(parts), rule.predict(data))
    left = RuleAccumulator(rule, target="target").update(data.iloc[:10])
    right = RuleAccumulator(rule, target="target").update(data.iloc[10:])
    merged = left.merge(right).finalize()
    assert merged == rule.aggregate([data.iloc[:3], data.iloc[3:20], data.iloc[20:]], target="target")
    assert merged["样本总数"] == 23
    assert merged["命中样本数"] == 13
    assert left.rule.result_ is None
    assert Rule("True").predict(data).all()


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("missing_separate", [False, True])
def test_specs_from_binner_match_reserved_policies(categorical, missing_separate):
    values = (
        ["a,b", "缺失", "其他", "a'b", None, "特殊"] * 10
        if categorical
        else list(np.arange(56, dtype=float)) + [np.nan, np.nan, -999.0, -999.0]
    )
    special = ["特殊"] if categorical else [-999.0]
    X = pd.DataFrame({"x": values})
    binner = OptimalBinning(
        method="quantile", max_n_bins=3, special_codes=special, missing_separate=missing_separate, n_jobs=1
    ).fit(X, np.arange(60) % 2)
    raw = pd.concat([X, pd.DataFrame({"x": ["新类别"] if categorical else [np.inf]})], ignore_index=True)
    expected = binner.transform(raw).x
    specs = BinSpec.from_binner(binner, "x")
    for key in expected.unique():
        np.testing.assert_array_equal(specs[key].mask(raw), expected == key)
        np.testing.assert_array_equal(Rule(specs[key].to_expression()).predict(raw), expected == key)


@pytest.mark.parametrize("unknown_target", [-1, 0])
def test_specs_unknown_category_can_join_existing_or_missing_bin(unknown_target):
    X = pd.DataFrame({"class": ["A", "B", None] * 10})
    binner = OptimalBinning(method="quantile", max_n_bins=2, handle_unknown=unknown_target, n_jobs=1).fit(
        X, np.arange(30) % 2
    )
    raw = pd.DataFrame({"class": ["A", "B", None, "C"]})
    expected = binner.transform(raw)["class"]
    specs = BinSpec.from_binner(binner, "class")
    for key in expected.unique():
        np.testing.assert_array_equal(specs[key].mask(raw), expected == key)
        np.testing.assert_array_equal(Rule(specs[key].to_expression()).predict(raw), expected == key)


@pytest.mark.parametrize("dtype,values", [("string", ["A", pd.NA]), ("Int64", [1, pd.NA])])
def test_nullable_missing_rules_use_isna_not_self_inequality(dtype, values):
    raw = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    for spec in (
        BinSpec("x", -1, kind="missing"),
        BinSpec("x", 0, kind="categories", values=(values[0],), include_missing=True),
    ):
        np.testing.assert_array_equal(Rule(spec.to_expression()).predict(raw), spec.mask(raw))
