"""运行 README 模型演示，保存方法的完整返回表与原生图形。

从项目根目录运行 ``python scripts/readme_examples/models.py``。
可用 ``--section`` 单独复现一组示例；Optuna Study 保存在审计目录，
使用 ``--dashboard`` 可启动读取真实 Study 的本地 Optuna Dashboard。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import hscredit  # noqa: F401 - 注册 DataFrame.save 等扩展。
from hscredit import CatBoost, LightGBM, LogisticRegression, ModelTuner, ScoreCard, XGBoost
from hscredit.core.binning import OptimalBinning
from hscredit.core.models.losses import AUCMetric, FocalLoss, WeightedBCELoss, make_metric
from hscredit.core.models.tuning import (
    Categorical,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    Integer,
    Real,
    choice,
    suggest_categorical,
    suggest_int,
    uniform,
)
from hscredit.core.viz import (
    plot_model_feature_importance,
    plot_model_sample_shap,
    plot_weights,
    score_distribution_comparison_plot,
)
from hscredit.report import auto_model_report

ASSETS = ROOT / "docs" / "assets" / "readme" / "models"
AUDIT = ROOT / ".audit_tmp" / "readme-complete" / "models"
STORAGE_PATH = AUDIT / "optuna-multiloan-study.sqlite3"
STORAGE = "sqlite:///" + STORAGE_PATH.as_posix()
FEATURES = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "手机号近一个月非银多头机构数"]


def prepare_data():
    """按 README 的统一切分保留完整三特征与 FPD 标签。"""
    data = pd.read_excel(ROOT / "examples" / "hscredit_yyp.xlsx")
    X, y = data[FEATURES], data["FPD"]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


def save_table(name, result):
    """不筛列、不重排、不改名，保存原生完整 DataFrame。"""
    (ASSETS / f"{name}.html").write_text(result.to_html(), encoding="utf-8")
    (AUDIT / f"{name}.txt").write_text(result.to_string(), encoding="utf-8")
    result.to_pickle(AUDIT / f"{name}.pkl")
    export_table(name, result)
    print(f"{name}: {result.shape}", flush=True)
    return result


def export_table(name, result):
    """按指标实际含义设置原生 Excel 样式，保留所有行列和原始数值。"""
    formats = {
        "logistic-summary": {"condition_cols": ["Coef."]},
        "scorecard-points": {"condition_cols": ["score"]},
        "ensemble-report-metrics": {"percent_rows": [3], "condition_rows": [0, 1]},
        "pareto-history": {"condition_cols": ["values_AUC曲线下面积", "values_均方概率误差"]},
        "pareto-trial-review": {"condition_cols": ["AUC曲线下面积", "均方概率误差"]},
        "dashboard-history": {"condition_cols": ["value_AUC"]},
    }
    return result.save(str(ASSETS / f"{name}.xlsx"), auto_width=True, **formats.get(name, {}))


def save_value(name, result):
    """完整记录标量或字典，不提取部分键。"""
    (ASSETS / f"{name}.txt").write_text(repr(result), encoding="utf-8")
    print(f"{name}: {result!r}", flush=True)
    return result


def run_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(solver="liblinear", max_iter=1000, n_jobs=1, random_state=42)
    model.fit(X_train, y_train)
    save_table("logistic-summary", model.summary())
    save_value("logistic-evaluation", model.evaluate(X_test, y_test))
    figure = plot_weights(model, save=str(ASSETS / "logistic-weights.png"))
    plt.close(figure)

    binner = OptimalBinning(method="quantile", max_n_bins=4, n_jobs=1)
    binner.fit(X_train, y_train)
    card = ScoreCard(
        binner=binner,
        base_score=650,
        pdo=50,
        rate=2,
        base_odds=35,
        lr_kwargs={"solver": "liblinear", "max_iter": 1000, "n_jobs": 1, "random_state": 42},
    )
    card.fit(X_train, y_train, input_type="raw")
    save_table("scorecard-points", card.export(to_frame=True))
    figure = score_distribution_comparison_plot(
        {"训练集": card.predict(X_train), "测试集": card.predict(X_test)},
        save=str(ASSETS / "scorecard-distribution.png"),
    )
    plt.close(figure)
    return model, card


def fit_ensemble(X_train, y_train):
    loss = FocalLoss(alpha=0.75, gamma=2.0)
    model = LightGBM(
        objective=loss,
        n_estimators=60,
        num_leaves=7,
        learning_rate=0.05,
        validation_fraction=0.0,
        n_jobs=1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model, loss


def run_ensemble(X_train, X_test, y_train, y_test):
    model, loss = fit_ensemble(X_train, y_train)
    save_value("lightgbm-evaluation", model.evaluate(X_test, y_test))
    save_value("focal-loss-evaluation", loss.metric().evaluate(y_test, model.predict_proba(X_test)))
    save_value("focal-business-evaluation", loss.business_metric().evaluate(y_test, model.predict_proba(X_test)))

    weighted_loss = WeightedBCELoss(pos_weight=3, neg_weight=1)
    xgboost = XGBoost(
        objective=weighted_loss,
        n_estimators=40,
        max_depth=3,
        scale_pos_weight=1,
        validation_fraction=0.0,
        n_jobs=1,
        random_state=42,
    ).fit(X_train, y_train)
    save_value("xgboost-evaluation", xgboost.evaluate(X_test, y_test))
    catboost = CatBoost(
        objective=weighted_loss,
        eval_metric=weighted_loss.metric(),
        iterations=40,
        depth=3,
        validation_fraction=0.0,
        n_jobs=1,
        random_state=42,
        allow_writing_files=False,
    ).fit(X_train, y_train)
    save_value("catboost-evaluation", catboost.evaluate(X_test, y_test))

    report = auto_model_report(
        model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        excel_path=str(ASSETS / "ensemble-model-report.xlsx"),
        n_jobs=1,
        verbose=False,
    )
    save_table("ensemble-report-metrics", report.get_metrics())
    return model


def declared_spaces():
    """各库风格的空间声明均由 hscredit 转换后交给 Optuna。"""
    return {
        "sklearn-lists": {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.08]},
        "skopt-style": {"max_depth": Integer(2, 4), "learning_rate": Real(0.03, 0.08)},
        "hyperopt-style": {
            "max_depth": choice("max_depth", [2, 3, 4]),
            "learning_rate": uniform("learning_rate", 0.03, 0.08),
        },
        "bayes-bounds": {"max_depth": (2, 4), "learning_rate": (0.03, 0.08)},
        "optuna-suggest": {
            "max_depth": suggest_int("max_depth", 2, 4),
            "learning_rate": suggest_categorical("learning_rate", [0.03, 0.05, 0.08]),
        },
        "optuna-distributions": {"max_depth": IntDistribution(2, 4), "learning_rate": FloatDistribution(0.03, 0.08)},
    }


def run_tuning(X_train, X_test, y_train, y_test):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    common = dict(
        model_class=LightGBM,
        fixed_params={"n_estimators": 30, "num_leaves": 7, "validation_fraction": 0.0, "verbosity": -1},
        cv=3,
        n_jobs=1,
        random_state=42,
        early_stopping_rounds=None,
        retention="summary",
    )
    for name, space in declared_spaces().items():
        tuner = ModelTuner(**common, search_space=space, metric="auc")
        params = tuner.fit(X_train, y_train, n_trials=2, show_progress_bar=False, catch=())
        save_value(f"space-{name}-params", params)

    brier = make_metric(brier_score_loss, name="均方概率误差", greater_is_better=False)
    tuner = ModelTuner(
        **common,
        search_space={
            "max_depth": Integer(2, 4),
            "learning_rate": Real(0.02, 0.15, prior="log-uniform"),
            "reg_lambda": Categorical([0.0, 1.0, 3.0]),
        },
        metric=[AUCMetric(), brier],
        storage=STORAGE,
        study_name=f"README-多目标模型-{run_id}",
    )
    params = tuner.fit(X_train, y_train, n_trials=12, show_progress_bar=False, catch=())
    save_value("pareto-best-params", params)
    save_table("pareto-history", tuner.get_optimization_history())
    figure = tuner.plot_pareto_front()
    figure.write_html(str(ASSETS / "pareto-front.html"))
    figure = tuner.plot_optimization_history()
    figure.write_html(str(ASSETS / "optimization-history.html"))
    save_table("pareto-trial-review", tuner.evaluate_study_trials([0, 1]))
    save_value("pareto-trial-record", tuner.get_trial_result(0))
    best = tuner.get_best_model()
    save_value("pareto-best-evaluation", best.evaluate(X_test, y_test))
    tuner.save(str(AUDIT / "pareto-tuner.pkl"))

    single = ModelTuner(
        **common,
        search_space={"max_depth": Integer(2, 4), "learning_rate": Real(0.03, 0.1)},
        metric="auc",
        storage=STORAGE,
        study_name=f"README-训练过程-{run_id}",
    )
    single.fit(X_train, y_train, n_trials=10, show_progress_bar=False, catch=())
    save_table("dashboard-history", single.get_optimization_history())
    (AUDIT / "dashboard.json").write_text(
        json.dumps(
            {"storage": STORAGE, "pareto_study": tuner.study_.study_name, "training_study": single.study_.study_name},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return tuner


def run_explain(X_train, X_test, y_train, y_test):
    model, _ = fit_ensemble(X_train, y_train)
    figure = plot_model_feature_importance(
        model, X_test, y_test, save=str(ASSETS / "model-feature-importance.png"), show=False
    )
    plt.close(figure)
    figure = plot_model_sample_shap(
        model,
        sample=X_test.iloc[0],
        background_data=X_train,
        save=str(ASSETS / "model-sample-shap.png"),
        show=False,
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=["all", "regression", "ensemble", "tuning", "explain"], default="all")
    parser.add_argument("--dashboard", action="store_true", help="读取已生成 Study 并启动本地 Dashboard")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    if args.dashboard:
        from optuna_dashboard import run_server

        run_server(STORAGE, host="127.0.0.1", port=args.port)
        return
    data = prepare_data()
    stages = {"regression": run_regression, "ensemble": run_ensemble, "tuning": run_tuning, "explain": run_explain}
    for name, stage in stages.items():
        if args.section in {"all", name}:
            print(f"开始模型示例: {name}", flush=True)
            stage(*data)
            print(f"完成模型示例: {name}", flush=True)


if __name__ == "__main__":
    main()
