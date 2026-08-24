"""hscredit 可执行快速开始示例.

运行方式::

    python examples/00_quickstart.py
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.model_selection import time_train_test_split
from hscredit.core.models import RandomForest, ScoreCard
from hscredit.core.models.calibration import ProbabilityCalibrator
from hscredit.core.selectors import IVSelector, VIFSelector, CompositeFeatureSelector
from hscredit.report import auto_model_report
from hscredit.report.mining import SingleFeatureRuleMiner, TreeRuleExtractor


def make_demo_data(n_samples: int = 400) -> pd.DataFrame:
    """构造不依赖外部文件的演示数据."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        weights=[0.72, 0.28],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=["年龄", "收入", "负债率", "申请次数"])
    df["申请日期"] = pd.date_range("2024-01-01", periods=n_samples, freq="D")
    df["客户编号"] = [f"C{i // 2:04d}" for i in range(n_samples)]
    df["目标"] = y
    return df


def run_quickstart(output_dir=None):
    """执行从时间切分到模型报告的完整基础流程."""
    df = make_demo_data()
    train_df, test_df = time_train_test_split(df, "申请日期", test_size=0.25)
    features = ["年龄", "收入", "负债率", "申请次数"]
    X_train, y_train = train_df[features], train_df["目标"]
    X_test, y_test = test_df[features], test_df["目标"]
    X_model_train, X_calib, y_model_train, y_calib = sklearn_train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    binner = OptimalBinning(method="best_iv", max_n_bins=5)
    binner.fit(X_train, y_train)
    X_train_woe = binner.transform(X_train, metric="woe")

    selector = CompositeFeatureSelector(
        [
            ("iv", IVSelector(threshold=0.0)),
            ("vif", VIFSelector(threshold=20.0)),
        ]
    )
    X_selected = selector.fit_transform(X_train_woe, y_train)

    scorecard = ScoreCard(binner=binner)
    scorecard.fit(X_train_woe, y_train)
    scores = scorecard.predict(X_test)

    model = RandomForest(n_estimators=30, random_state=42)
    model.fit(X_model_train, y_model_train)
    metrics = model.evaluate(X_test, y_test)

    calibrator = ProbabilityCalibrator(
        model=model,
        method="platt",
        calib_ratio=None,
    ).fit(X_calib, y_calib)
    calibration_report = calibrator.report(X_test, y_test)

    rule_df = train_df[features + ["目标"]]
    single_miner = SingleFeatureRuleMiner(
        target="目标",
        method="best_iv",
        max_n_bins=4,
        min_samples=5,
    ).fit(rule_df)
    single_rules = single_miner.get_top_rules(top_n=5)

    tree_extractor = TreeRuleExtractor(max_depth=3, min_samples_leaf=10)
    tree_extractor.fit(X_train, y_train)
    tree_rules = tree_extractor.extract_rules()

    artifact_path = None
    report_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = model.save_artifact(output_dir / "risk_model.joblib")
        report_path = output_dir / "quickstart_model_report.xlsx"

    report = auto_model_report(
        model,
        X_train=X_model_train,
        y_train=y_model_train,
        X_test=X_test,
        y_test=y_test,
        excel_path=report_path,
        verbose=False,
        with_plots=False,
    )
    report_metrics = report.get_metrics()
    report_summary = report.summary()

    if report_path is not None:
        assert report_path.is_file() and report_path.stat().st_size > 0, "模型报告未正确生成"

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "calibration_rows": len(X_calib),
        "selected_features": list(X_selected.columns),
        "scores": scores,
        "metrics": metrics,
        "calibration_report": calibration_report,
        "single_rules": single_rules,
        "tree_rules": tree_rules,
        "report": report,
        "report_path": report_path,
        "report_metrics": report_metrics,
        "report_summary": report_summary,
        "artifact_path": artifact_path,
    }


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        result = run_quickstart(temp_dir)
        print("快速开始示例执行成功")
        print(f"训练集/测试集: {result['train_rows']}/{result['test_rows']}")
        print(f"测试集 AUC: {result['metrics']['AUC']:.4f}")
        print(f"模型报告: {result['report_path']}")
        print("模型报告核心指标:")
        print(result["report_metrics"].to_string(index=False))
        print("模型报告摘要:")
        print(result["report_summary"].to_string())
