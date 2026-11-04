"""报告标签、实际订单口径、冻结分箱与机器输出的业务语义回归。"""

import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from hscredit import ModelReport, feature_bin_stats
from hscredit.utils.targets import bind_targets


@pytest.mark.parametrize("policy,missing", [("drop", np.nan), ("positive", 1), ("negative", 0)])
@pytest.mark.parametrize("grey", [False, True])
def test_missing_target_is_independent_of_grey(policy, missing, grey):
    frame = pd.DataFrame({"target": [0, 1, 2, np.nan], "dpd": [0, 10, 2, np.nan]})
    for options in [{"target": "target"}, {"overdue": "dpd", "dpds": 7}]:
        data, targets = bind_targets(frame, **options, del_grey=grey, handle_missing_target=policy)
        actual = data[targets.columns[0]].to_numpy()
        np.testing.assert_allclose(actual, [0, 1, np.nan if grey else 0, missing], equal_nan=True)


def make_report(**kwargs):
    X = pd.DataFrame({"x": np.linspace(-4, 1, 200)})
    y = (np.arange(200) % 3 == 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return ModelReport(model, datasets={"train": (X, y), "test": (X + 5, y)}, n_jobs=1, **kwargs)


def test_weight_comparison_does_not_change_real_order_statistics():
    y = (np.arange(200) % 3 == 0).astype(int)
    report = make_report(
        sample_weight={k: np.where(y == 1, 20, 1) for k in ("train", "test")}, show_weighted_metrics=True
    )
    metrics = report.get_metrics().set_index("统计项")
    assert metrics.loc["坏样本率", "train"] == pytest.approx(0.335)
    assert report.get_bin_table("train").iloc[-1]["坏样本率"] == pytest.approx(0.335)
    assert list(metrics.columns) == ["train", "train（加权）", "test", "test（加权）"]
    assert np.isfinite(metrics.loc["KS", "train（加权）"])


def test_reference_bins_and_explicit_rules_are_shared():
    report = make_report()
    a = report.get_bin_table("train", max_n_bins=3)
    b = report.get_bin_table("test", max_n_bins=3)
    assert set(b["分箱标签"]) <= set(a["分箱标签"])
    assert b.iloc[0]["样本总数"] == 200  # 平移后的概率全部落入冻结的首箱。
    assert a["样本总数"].tolist() != b["样本总数"].tolist()
    independent = make_report(binning_reference="independent")
    assert independent.get_bin_table("train")["分箱标签"].tolist() != independent.get_bin_table("test")["分箱标签"].tolist()
    manual = make_report(rules=[0.3, 0.4])
    assert manual.get_bin_table("train")["分箱标签"].tolist() == manual.get_bin_table("test")["分箱标签"].tolist()


def test_missing_label_modes_and_strict_json():
    X = pd.DataFrame({"x": range(10)})
    y = np.tile([0, 1], 5)
    model = LogisticRegression().fit(X, y)
    data = X.copy()
    data["target"] = y.astype(float)
    data.loc[3, "target"] = np.nan
    first = ModelReport(model, X_train=data, y_train=data.target, n_jobs=1)
    second = ModelReport(model, X_train=data, target="target", n_jobs=1)
    pd.testing.assert_frame_equal(first.get_metrics(), second.get_metrics())
    json.loads(json.dumps(first.to_dict(), allow_nan=False))
    with pytest.raises(ValueError, match="缺失"):
        ModelReport(model, X_train=data, target="target", handle_missing_target="raise")


@pytest.mark.parametrize("amount", ["absent", [-1] * 200, [np.nan] * 200, [1] * 3, ["100"] * 200])
def test_invalid_amount_is_rejected_before_report_computation(amount):
    with pytest.raises((ValueError, KeyError)):
        make_report().get_bin_table("train", amount=amount)


def test_feature_bin_stats_missing_policy_is_public():
    data = pd.DataFrame({"x": [0, 1, 2, 3], "target": [0, 1, 0, np.nan]})
    table = feature_bin_stats(data, "x", target="target", rules=[1.5], margins=True, handle_missing_target="positive")
    assert table.iloc[-1]["坏样本率"] == 0.5


@pytest.mark.parametrize("probabilities", [[-0.1, 1.1], [0.4, 0.8], [0.2, 0.3, 0.5]])
def test_probability_contract_rejects_invalid_ranges_sums_and_class_count(probabilities):
    class InvalidModel:
        def predict_proba(self, data):
            return np.tile(probabilities, (len(data), 1))

    with pytest.raises(ValueError, match="概率"):
        ModelReport(InvalidModel(), X_train=pd.DataFrame({"x": range(10)}), y_train=np.arange(10) % 2, n_jobs=1)


def test_score_output_is_not_mistaken_for_probability():
    class ScoreModel:
        def predict_score(self, data):
            return np.linspace(-20, 20, len(data))

    report = ModelReport(
        ScoreModel(),
        X_train=pd.DataFrame({"x": range(10)}),
        y_train=np.arange(10) % 2,
        prediction_method="predict_score",
        n_jobs=1,
    )
    assert np.isfinite(report.get_metrics().set_index("统计项").loc["AUC", "训练集"])


def test_probability_validation_respects_float16_precision():
    class HalfModel:
        def predict_proba(self, data):
            return np.tile(np.array([0.3333, 0.6665], dtype=np.float16), (len(data), 1))

    report = ModelReport(HalfModel(), X_train=pd.DataFrame({"x": range(10)}), y_train=np.arange(10) % 2, n_jobs=1)
    assert report.get_metrics().set_index("统计项").loc["AUC", "训练集"] == 0.5


def test_score_temporary_column_does_not_overwrite_actual_amount():
    X = pd.DataFrame({"x": np.arange(20)})
    y = np.arange(20) % 2
    model = LogisticRegression().fit(X, y)
    data = X.assign(__score__=np.arange(1, 21) * 100.0)
    report = ModelReport(model, X_train=data, y_train=y, n_jobs=1)
    table = report.get_bin_table(amount="__score__")
    assert table.iloc[-1]["样本总数"] == data["__score__"].sum()


@pytest.mark.parametrize("cache_policy", ["full", "none"])
def test_parallel_dataset_tables_fit_training_bins_once(monkeypatch, cache_policy):
    from concurrent.futures import ThreadPoolExecutor
    import time
    import hscredit.report.feature_analyzer as analyzer

    original = analyzer.feature_bin_stats
    fitted = []

    def counting(*args, **kwargs):
        if kwargs.get("return_binner"):
            fitted.append(True)
            time.sleep(0.02)
        return original(*args, **kwargs)

    monkeypatch.setattr(analyzer, "feature_bin_stats", counting)
    report = make_report(cache_policy=cache_policy)
    with ThreadPoolExecutor(max_workers=2) as executor:
        tables = list(executor.map(report.get_bin_table, ["train", "test"]))
    assert len(fitted) == 1
    assert set(tables[1]["分箱标签"]) <= set(tables[0]["分箱标签"])


def test_dataset_prediction_snapshot_survives_model_buffer_reuse():
    class BufferedModel:
        buffer = np.empty((4, 2))

        def predict_proba(self, X):
            self.buffer[:, 1] = X["x"]
            self.buffer[:, 0] = 1 - self.buffer[:, 1]
            return self.buffer

    first = pd.DataFrame({"x": [0.1, 0.2, 0.8, 0.9]})
    second = pd.DataFrame({"x": [0.9, 0.8, 0.2, 0.1]})
    report = ModelReport(
        BufferedModel(), datasets={"train": (first, [0, 0, 1, 1]), "test": (second, [0, 0, 1, 1])}, n_jobs=1
    )
    np.testing.assert_array_equal(report._datasets["train"].prediction, first.x)
    np.testing.assert_array_equal(report._datasets["test"].prediction, second.x)


def test_serialized_report_keeps_frozen_rules_without_refitting(monkeypatch):
    import pickle
    import hscredit.report.feature_analyzer as analyzer

    report = make_report()
    training = report.get_bin_table("train")
    restored = pickle.loads(pickle.dumps(report))
    original = analyzer.feature_bin_stats

    def no_refit(*args, **kwargs):
        if kwargs.get("return_binner"):
            raise AssertionError("恢复报告不能重新拟合冻结规则")
        return original(*args, **kwargs)

    monkeypatch.setattr(analyzer, "feature_bin_stats", no_refit)
    current = restored.get_bin_table("test")
    assert set(current["分箱标签"]) <= set(training["分箱标签"])
