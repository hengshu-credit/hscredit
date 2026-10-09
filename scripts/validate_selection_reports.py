"""使用约定真实样例验收全部筛选器报告，不导出逐行训练样本。

运行: python scripts/validate_selection_reports.py --output-dir .audit_tmp/selector-workflows
输出统一明细的聚合报告及验证JSON；不评估模型生产有效性。
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hscredit.core import selectors as s
from hscredit.core.selectors.reporting import DETAIL_COLUMNS, METRIC_COLUMNS, SUMMARY_COLUMNS, HISTORY_COLUMNS
from hscredit.report import load_selection_report

NUMERIC = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
CATEGORY, TARGET = "商品类别", "FPD"


def validate(input_path, directory):
    frame = pd.read_excel(input_path, usecols=NUMERIC + [CATEGORY, TARGET])
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError("验证目标必须全部为有效0/1，不能静默删行")
    X, y = frame[NUMERIC + [CATEGORY]], frame[TARGET]
    numeric = frame[NUMERIC].apply(pd.to_numeric, errors="raise")
    # 仅接口验收：模型型筛选需要有限数值；不作为生产预处理或OOT有效性证据。
    model_X = numeric.fillna(numeric.median()).fillna(0)
    cut = int(len(frame) * 0.75)
    forest = RandomForestClassifier(n_estimators=16, max_depth=3, random_state=42, n_jobs=1)
    common = {"n_jobs": 1, "target": TARGET}
    jobs = [
        (s.TypeSelector(dtype_include="number", **common), X, y),
        (s.RegexSelector(pattern="分|机构|青云|类别", **common), X, y),
        (s.NullSelector(**common), X, y),
        (s.ModeSelector(**common), X, y),
        (s.CardinalitySelector(threshold=1000, **common), X, y),
        (s.VarianceSelector(**common), numeric, y),
        (s.Chi2Selector(**common), X, y),
        (s.FTestSelector(**common), numeric, y),
        (s.MutualInfoSelector(random_state=42, **common), X, y),
        (s.IVSelector(**common), X, y),
        (s.KSSelector(**common), X, y),
        (s.LiftSelector(**common), numeric, y),
        (s.PSISelector(oot_df=X.iloc[cut:], **common), X.iloc[:cut], y.iloc[:cut]),
        (s.StabilityAwareSelector(oot_df=X.iloc[cut:], random_state=42, **common), X.iloc[:cut], y.iloc[:cut]),
        (
            s.CorrSelector(
                weights={name: len(NUMERIC) - i for i, name in enumerate(NUMERIC)}, binning_params=None, **common
            ),
            model_X,
            y,
        ),
        (s.VIFSelector(max_iter=5, **common), model_X, y),
        (s.FeatureImportanceSelector(forest, threshold=2, **common), model_X, y),
        (s.NullImportanceSelector(forest, n_runs=2, cv=2, **common), model_X, y),
        (
            s.RFESelector(
                LogisticRegression(max_iter=300), n_features_to_select=2, report_history="diagnostics", **common
            ),
            model_X,
            y,
        ),
        (
            s.SequentialFeatureSelector(
                LogisticRegression(max_iter=300), n_features_to_select=2, cv=2, report_history="diagnostics", **common
            ),
            model_X,
            y,
        ),
        (s.BorutaSelector(forest, n_estimators=16, max_iter=5, **common), model_X, y),
        (
            s.StepwiseSelector(estimator="logit", max_features=2, max_iter=4, report_history="diagnostics", **common),
            model_X,
            y,
        ),
    ]
    records, fitted = [], []
    canonical_dtypes = None
    for selector, data, target in jobs:
        name = type(selector).__name__
        try:
            selector.fit(data, target)
            report = s.collect_selection_report(selector)
            detail = selector.get_selection_report_df()
            assert detail.columns.tolist() == DETAIL_COLUMNS
            assert report.metrics.columns.tolist() == METRIC_COLUMNS
            assert report.summary.columns.tolist() == SUMMARY_COLUMNS
            assert report.history.columns.tolist() == HISTORY_COLUMNS
            if canonical_dtypes is None:
                canonical_dtypes = detail.dtypes
            else:
                pd.testing.assert_series_equal(detail.dtypes, canonical_dtypes)
            assert detail["特征"].tolist() == data.columns.tolist()
            assert detail.loc[detail["筛选结果"] == "保留", "特征"].tolist() == selector.selected_features_
            # 未提供目标列时不合成标签；提供时默认原样透传，只有 target_rm=True 才删除。
            assert TARGET not in selector.transform(data).columns
            labeled = data.assign(**{TARGET: target})
            preserved = selector.transform(labeled)
            assert preserved.columns.tolist() == selector.selected_features_ + [TARGET]
            pd.testing.assert_series_equal(preserved[TARGET], labeled[TARGET])
            selector.set_params(target_rm=True)
            assert selector.transform(labeled).columns.tolist() == selector.selected_features_
            selector.set_params(target_rm=False)
            assert report.metadata["完整"]
            fitted.append((name, selector))
            records.append(
                {
                    "筛选器": name,
                    "状态": "通过",
                    "输入行数": len(data),
                    "输入字段数": data.shape[1],
                    "保留字段数": len(selector.selected_features_),
                    "明细行数": len(detail),
                    "指标行数": len(report.metrics),
                    "历史行数": len(report.history),
                }
            )
        except Exception as exc:
            # 底层异常可能含原始值，不写到可交付验证JSON。
            records.append({"筛选器": name, "状态": "失败", "异常类型": type(exc).__name__})
    pipeline = Pipeline([("缺失", s.NullSelector(**common)), ("IV", s.IVSelector(threshold=0, **common))]).fit(frame, y)
    pipeline_report = s.collect_selection_report(pipeline)
    assert TARGET not in pipeline_report.details["特征"].tolist()
    assert TARGET in pipeline.transform(frame).columns
    assert TARGET not in pipeline.transform(frame.drop(columns=TARGET)).columns
    composite = s.CompositeFeatureSelector(
        [
            ("缺失", s.NullSelector(**common)),
            ("嵌套", s.CompositeFeatureSelector([s.VarianceSelector(**common)], n_jobs=1)),
        ],
        include=[NUMERIC[0]],
        n_jobs=1,
    ).fit(model_X, y)
    scorecard = s.ScorecardFeatureSelection(corr_threshold=None, n_jobs=1).fit(X, y)
    for selector in [composite, scorecard]:
        report = s.collect_selection_report(selector)
        assert report.details.columns.tolist() == DETAIL_COLUMNS
        assert report.metadata["完整"]
        records.append(
            {
                "筛选器": type(selector).__name__,
                "状态": "通过",
                "阶段数": len(report.summary),
                "明细行数": len(report.details),
                "保留字段数": len(selector.selected_features_),
            }
        )
    all_report = s.collect_selection_report(fitted)
    assert "最终字段" not in all_report.metadata
    artifacts = all_report.save(directory / "全部筛选器明细", by_selector=True, max_rows_per_sheet=200)
    restored = load_selection_report(artifacts["路径"]["json"])
    for attribute in ("summary", "details", "metrics", "history"):
        pd.testing.assert_frame_equal(getattr(restored, attribute), getattr(all_report, attribute))
    return {
        "状态": "通过" if all(item["状态"] == "通过" for item in records) else "失败",
        "输入行数": len(frame),
        "单项筛选器数": len(jobs),
        "组合入口数": 2,
        "逐行样本输出": False,
        "用途": "功能接口与报告对账，不是生产模型效果或容量认证",
        "筛选器验收": records,
        "Pipeline阶段数": len(pipeline_report.summary),
        "保存重载": "四表逐项等价",
        "报告产物": artifacts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/hscredit_yyp.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".audit_tmp/selector-workflows")
    options = parser.parse_args()
    directory = options.output_dir.resolve() / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    directory.mkdir(parents=True, exist_ok=False)
    result = validate(options.input, directory)
    destination = directory / "validation_result.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": result["状态"], "结果文件": str(destination)}, ensure_ascii=False))
    return 0 if result["状态"] == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
