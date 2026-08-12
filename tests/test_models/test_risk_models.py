"""风控模型框架测试.

测试统一模型接口和各种模型实现。
"""

import importlib.util

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from hscredit.core.models import (
    BaseRiskModel,
    XGBoostRiskModel,
    LightGBMRiskModel,
    RandomForestRiskModel,
    GradientBoostingRiskModel,
    ModelReport,
)

# 可选依赖是否真实可用（懒加载类的 __module__ 始终在 hscredit 下，无法据此判断）
HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


def test_catboost_public_n_jobs_controls_native_thread_count(monkeypatch):
    """公共 n_jobs 必须覆盖旧的全核默认和冲突的原生 thread_count。"""
    import hscredit.core.models.boosting.catboost_model as catboost_module

    observed = {}

    class FakeCatBoostClassifier:
        def __init__(self, **params):
            observed.update(params)

        def fit(self, X, y, **kwargs):
            return self

        def get_best_iteration(self):
            return 0

        def get_best_score(self):
            return {}

        def get_evals_result(self):
            return {}

    monkeypatch.setattr(catboost_module, "CATBOOST_AVAILABLE", True)
    monkeypatch.setattr(
        catboost_module,
        "cb",
        type("FakeCatBoostModule", (), {"CatBoostClassifier": FakeCatBoostClassifier}),
    )
    model = catboost_module.CatBoostRiskModel(
        n_jobs=3,
        params={"thread_count": 99},
        validation_fraction=0,
    )
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0, 1])

    model.fit(X, y)

    assert observed["thread_count"] == 3


def test_ngboost_public_n_jobs_limits_native_training_threads(monkeypatch):
    """NGBoost 不公开 n_jobs 时，公共预算仍须约束其底层原生线程池。"""
    import hscredit.core.models.boosting.ngboost_model as ngboost_module

    observed = {"limits": [], "fit_inside_context": False}

    class RecordingThreadLimit:
        def __init__(self, limits):
            observed["limits"].append(limits)

        def __enter__(self):
            observed["active"] = True

        def __exit__(self, exc_type, exc, traceback):
            observed["active"] = False

    class FakeNGBClassifier:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, **kwargs):
            observed["fit_inside_context"] = observed.get("active", False)
            return self

    monkeypatch.setattr(ngboost_module, "NGBOOST_AVAILABLE", True)
    monkeypatch.setattr(ngboost_module, "NGBClassifier", FakeNGBClassifier)
    monkeypatch.setattr(ngboost_module, "Bernoulli", object())
    monkeypatch.setattr(ngboost_module, "LogScore", object())
    monkeypatch.setattr(ngboost_module, "threadpool_limits", RecordingThreadLimit)

    model = ngboost_module.NGBoostRiskModel(
        n_jobs=3,
        n_estimators=2,
        validation_fraction=0,
    )
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0, 1])

    model.fit(X, y)

    assert observed["limits"] == [3]
    assert observed["fit_inside_context"] is True


@pytest.fixture
def sample_data():
    """创建测试数据."""
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=2, n_classes=2, weights=[0.7, 0.3], random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestBaseRiskModel:
    """测试模型基类功能."""

    def test_model_info(self, sample_data):
        """测试模型信息获取."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        info = model.get_model_info()

        assert info["model_type"] == "RandomForestRiskModel"
        assert info["n_features"] == X_train.shape[1]
        assert info["n_classes"] == 2
        assert "params" in info

    def test_evaluate(self, sample_data):
        """测试模型评估."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)

        assert "AUC" in metrics
        assert "KS" in metrics
        assert 0 <= metrics["AUC"] <= 1
        assert 0 <= metrics["KS"] <= 1

    def test_predict_score(self, sample_data):
        """测试风险评分预测."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        scores = model.predict_score(X_test)

        assert len(scores) == len(X_test)
        assert 0 <= scores.min() <= 1000
        assert 0 <= scores.max() <= 1000


class TestXGBoostRiskModel:
    """测试XGBoost模型."""

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost 未安装")
    def test_fit_predict(self, sample_data):
        """测试训练和预测."""
        X_train, X_test, y_train, y_test = sample_data

        model = XGBoostRiskModel(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost 未安装")
    def test_feature_importance(self, sample_data):
        """测试特征重要性."""
        X_train, X_test, y_train, y_test = sample_data

        model = XGBoostRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        importances = model.get_feature_importances()

        assert isinstance(importances, pd.Series)
        assert len(importances) == X_train.shape[1]
        assert importances.index[0] in X_train.columns


class TestLightGBMRiskModel:
    """测试LightGBM模型."""

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM 未安装")
    def test_fit_predict(self, sample_data):
        """测试训练和预测."""
        X_train, X_test, y_train, y_test = sample_data

        model = LightGBMRiskModel(n_estimators=10, num_leaves=10, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert proba.shape == (len(X_test), 2)

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM 未安装")
    def test_early_stopping(self, sample_data):
        """测试早停功能."""
        X_train, X_test, y_train, y_test = sample_data

        model = LightGBMRiskModel(n_estimators=100, early_stopping_rounds=5, validation_fraction=0.2, random_state=42)
        model.fit(X_train, y_train)

        assert model._best_iteration is not None


class TestRandomForestRiskModel:
    """测试随机森林模型."""

    def test_fit_predict(self, sample_data):
        """测试训练和预测."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
        assert proba.shape == (len(X_test), 2)

    def test_sample_weight(self, sample_data):
        """测试样本权重."""
        X_train, X_test, y_train, y_test = sample_data

        sample_weight = np.ones(len(X_train))
        sample_weight[y_train == 1] = 2.0  # 增加正样本权重

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)


class TestModelReport:
    """测试模型评估报告（已统一为 hscredit.report.ModelReport）."""

    def test_model_report_alias(self):
        """core.models.ModelReport 应为 report.model_report.ModelReport 同一类."""
        from hscredit.report.model_report import ModelReport as CanonicalModelReport

        assert ModelReport is CanonicalModelReport

    def test_metrics(self, sample_data):
        """测试指标计算（多层：统计项 × 数据集）."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        report = ModelReport(model, X_train, y_train, X_test, y_test)
        metrics = report.get_metrics()

        assert isinstance(metrics, pd.DataFrame)
        assert "统计项" in metrics.columns
        assert "KS" in metrics["统计项"].values
        assert "AUC" in metrics["统计项"].values
        assert "训练集" in metrics.columns
        assert "测试集" in metrics.columns

    def test_feature_importance(self, sample_data):
        """测试特征重要性分析."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        report = ModelReport(model, X_train, y_train, X_test, y_test)
        importance = report.get_feature_importance(top_n=5)

        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 5
        assert "特征重要性" in importance.columns

    def test_summary_multi_index_columns(self, sample_data):
        """summary 应为「统计指标 × 数据集」多层列、逾期指标为行."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        report = ModelReport(model, X_train, y_train, X_test, y_test)
        summary = report.summary()

        assert isinstance(summary.columns, pd.MultiIndex)
        assert summary.index.name == "逾期指标"
        assert ("KS", "训练集") in summary.columns
        assert ("坏样本率", "测试集") in summary.columns

    def test_bin_table(self, sample_data):
        """评分分箱效果表（替代旧 get_score_distribution）."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        report = ModelReport(model, X_train, y_train, X_test, y_test)
        bin_table = report.get_bin_table("train", max_n_bins=5)

        assert isinstance(bin_table, pd.DataFrame)
        assert "坏样本率" in bin_table.columns

    def test_print_report(self, sample_data, capsys):
        """print_report 应输出完整文本报告."""
        X_train, X_test, y_train, y_test = sample_data

        model = RandomForestRiskModel(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)

        report = ModelReport(model, X_train, y_train, X_test, y_test)
        report.print_report()

        out = capsys.readouterr().out
        assert "模型评估快速报告" in out
        assert "模型性能指标" in out


class TestUnifiedInterface:
    """测试统一接口."""

    def test_all_models_same_interface(self, sample_data):
        """测试所有模型具有统一接口."""
        X_train, X_test, y_train, y_test = sample_data

        models = [
            RandomForestRiskModel(n_estimators=5, random_state=42),
            GradientBoostingRiskModel(n_estimators=5, random_state=42),
        ]

        for model in models:
            # 测试统一接口
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "predict_proba")
            assert hasattr(model, "get_feature_importances")
            assert hasattr(model, "evaluate")
            assert hasattr(model, "generate_report")

            # 测试功能
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            proba = model.predict_proba(X_test)
            importances = model.get_feature_importances()
            metrics = model.evaluate(X_test, y_test)

            assert len(predictions) == len(X_test)
            assert proba.shape[0] == len(X_test)
            assert len(importances) == X_train.shape[1]
            assert "AUC" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
