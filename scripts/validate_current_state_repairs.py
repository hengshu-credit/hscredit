"""复核现状审查修复：真实数据全行、关键闭环、持久化和大样本探针。

从仓库根目录执行，使用 --output-dir 指定本地结果目录；不访问真实数据库。
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import hscredit as h
from openpyxl import load_workbook
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from hscredit.core.metrics.aggregation import BinStatsAccumulator
from hscredit.core.rules.accumulator import RuleAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".audit_tmp/defect-repairs/real-data")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(ROOT / "examples/hscredit_yyp.xlsx")
    features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
    results = []
    state = {}

    def check(name, action):
        started = time.perf_counter()
        try:
            value = action()
            record = {"检查": name, "状态": "通过", "结果": value}
        except Exception as exc:
            record = {"检查": name, "状态": "失败", "错误": str(exc), "详情": traceback.format_exc()}
        record["耗时秒"] = round(time.perf_counter() - started, 4)
        results.append(record)
        (output / "verification.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"{name}: {record['状态']}", flush=True)

    def modules():
        imported = []
        for path in sorted((ROOT / "hscredit").rglob("*.py")):
            if path.name == "__main__.py":
                continue
            parts = path.relative_to(ROOT).with_suffix("").parts
            name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            importlib.import_module(name)
            imported.append(name)
        return {"模块数": len(imported), "样本数": len(df), "FPD坏样本数": int(df.FPD.sum())}

    check("全部源码模块导入", modules)

    def model_workflow():
        train, test = h.time_train_test_split(df, "放款时间", test_size=0.3)
        binner = h.OptimalBinning(method="quantile", n_jobs=1).fit(train[features], train.FPD)
        X_train = binner.transform(train[features], metric="woe")
        X_test = binner.transform(test[features], metric="woe")
        model = LogisticRegression(max_iter=1000).fit(X_train, train.FPD)
        card = h.ScoreCard(binner=binner, lr_model=model, calculate_stats=False)
        assert np.isfinite(card.predict_score(test[features])).all()
        binner.save_artifact(output / "分箱器.pkl")
        restored = h.OptimalBinning.load_artifact(output / "分箱器.pkl")
        pd.testing.assert_frame_equal(restored.transform(test[features], metric="woe"), X_test)
        report = h.ModelReport(model, X_train=X_train, y_train=train.FPD, X_test=X_test, y_test=test.FPD, n_jobs=1)
        result = report.to_excel(
            str(output / "模型报告.xlsx"),
            with_plots=False,
            show_importance=False,
            include_sample_records=False,
            mode="strict",
            return_result=True,
        )
        assert result.complete
        state.update(train=train, test=test, binner=binner, X_train=X_train, X_test=X_test, model=model)
        return {
            "训练行数": len(train),
            "验证行数": len(test),
            "验证AUC": float(roc_auc_score(test.FPD, model.predict_proba(X_test)[:, 1])),
            "模型报告状态": result.status_table().to_dict("records"),
        }

    check("时间切分分箱WOE建模评分持久化及模型报告", model_workflow)

    def feature_reports():
        table, rules, binners = h.feature_bin_stats(
            df,
            features + ["商品类别"],
            target="FPD",
            amount="放款金额",
            method="quantile",
            return_rules=True,
            return_binner=True,
            n_jobs=1,
        )
        assert set(rules) == set(features + ["商品类别"])
        for feature, binner in binners.items():
            codes = binner.transform(df[[feature]], metric="indices")[feature]
            assert len(codes) == len(df)
            stats = h.compute_bin_stats(codes.to_numpy(), df.FPD.to_numpy())
            assert int(stats["样本总数"].sum()) == len(df)
        overdue = h.feature_bin_stats(
            df, features[0], overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额", method="quantile", n_jobs=1
        )
        summary = h.feature_summary(
            df, features=features + ["商品类别"], y="FPD", psi_method="date", date="放款时间", n_jobs=1
        )
        assert len(summary) == 4
        with h.ExcelWriter() as writer:
            for name, frame in (("分箱统计", table), ("多逾期阈值", overdue), ("按月特征摘要", summary)):
                writer.insert_df2sheet(writer.get_sheet_by_name(name), frame, "A1")
            writer.save(str(output / "特征验证.xlsx"))
        return {"统计表形状": table.shape, "逾期表形状": overdue.shape, "摘要形状": summary.shape}

    check("全部970行多特征类别金额与多标签统计", feature_reports)

    def rule_workflow():
        rules = [h.Rule(f"`{features[0]}` < {float(df[features[0]].median())}"), h.Rule(f"`{features[1]}` > 3")]
        for mode in ("serial", "parallel"):
            result = h.RuleFlow(rules, mode=mode, n_jobs=1).predict(df)
            assert len(result) == len(df) and (result["是否命中"] ^ result["是否通过"]).all()
        whole = RuleAccumulator(rules[0], target="FPD").update(df).finalize()
        accum = RuleAccumulator(rules[0], target="FPD")
        for i in range(0, len(df), 113):
            accum.update(df.iloc[i : i + 113])
        assert accum.finalize() == whole
        tree = h.TreeRuleExtractor(
            algorithm="dt",
            target="FPD",
            features=features + ["商品类别"],
            max_depth=2,
            min_samples_leaf=20,
            n_jobs=1,
            random_state=42,
        ).fit(df)
        extracted = tree.extract_rules()
        assert len(extracted) > 0
        return {
            "计数": whole,
            "规则报告形状": rules[0].report(df, target="FPD", amount="放款金额").shape,
            "树规则数": len(extracted),
        }

    check("规则流规则报告流式计数和树规则挖掘", rule_workflow)

    def monitor():
        train, test = state["train"], state["test"]
        result = h.population_drift(
            train,
            test,
            features + ["商品类别"],
            target="FPD",
            date="放款时间",
            n_jobs=1,
            output=str(output / "客群监控.xlsx"),
            mode="strict",
            return_result=True,
        )
        assert result.complete
        assert result["样本口径"].set_index("数据集").loc["实际", "纳入行数"] == len(test)
        baseline = h.MonitoringBaseline().fit(train[features[0]])
        whole = baseline.evaluate(test[features[0]])
        streamed = baseline.evaluate_batches(test[features[0]].iloc[i : i + 51] for i in range(0, len(test), 51))
        assert whole["PSI"] == streamed["PSI"]
        predictor = h.OverduePredictor(
            feature=features[0], overdue=["MOB1"], dpds=[7, 3, 0], method="quantile", n_jobs=1
        ).fit(df)
        predicted = predictor.transform(df)
        assert len(predicted) == len(df)
        return {"监控章节数": len(result.sections), "监控样本数": len(test), "多标签逾期估计形状": predicted.shape}

    check("冻结基准监控与多标签逾期估计", monitor)

    def extras():
        from catboost import CatBoostClassifier
        from hscredit import ModelExplainer, ProbabilityCalibrator

        train, test, X_train, X_test = state["train"], state["test"], state["X_train"], state["X_test"]
        model = CatBoostClassifier(iterations=5, depth=2, verbose=False, thread_count=1, allow_writing_files=False).fit(
            X_train, train.FPD
        )
        assert np.isfinite(model.predict_proba(X_test)).all()
        calibrated = ProbabilityCalibrator(model=state["model"], calib_ratio=None).fit(X_test, test.FPD)
        assert np.isfinite(calibrated.predict_proba(X_test)).all()
        # 线性解释使用 raw 尺度，检验结果可计算与加法一致性，不作为校准效果评价。
        explanation = ModelExplainer(
            state["model"], background_data=X_train, model_output="raw", max_background=20
        ).explain(X_test.head(10))
        assert explanation.values.shape == (10, len(features))
        assert np.isfinite(explanation.values).all()
        return {"CatBoost验证样本数": len(test), "解释形状": explanation.values.shape}

    check("修复依赖环境后的CatBoost校准与SHAP解释", extras)

    def large_inputs():
        n = 1_000_000
        bins = np.arange(n) % 10
        labels = (np.arange(n) % 7 == 0).astype(int)
        direct = h.compute_bin_stats(bins, labels)
        accum = BinStatsAccumulator()
        for i in range(0, n, 10000):
            accum.update(bins[i : i + 10000], labels[i : i + 10000])
        pd.testing.assert_frame_equal(direct, accum.finalize())
        categories = pd.DataFrame({"类别": np.arange(100000) % 1000})
        matrix = h.OneHotEncoder(n_jobs=1, sparse_output=True, return_df=False).fit_transform(categories)
        assert matrix.shape == (100000, 1000) and matrix.nnz == 100000
        return {"分块统计行数": n, "稀疏编码形状": matrix.shape, "非零元素": matrix.nnz}

    check("百万行统计及十万行千类别稀疏编码", large_inputs)

    def artifacts():
        workbooks = list(output.glob("*.xlsx"))
        assert len(workbooks) >= 3
        detail = {}
        for path in workbooks:
            workbook = load_workbook(path, read_only=True)
            assert workbook.sheetnames
            detail[path.name] = workbook.sheetnames
            workbook.close()
        return detail

    check("Excel输出内容可重新读取", artifacts)
    return int(any(record["状态"] == "失败" for record in results))


if __name__ == "__main__":
    raise SystemExit(main())
