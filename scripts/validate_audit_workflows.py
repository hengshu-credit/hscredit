"""使用约定真实样例做可重复的聚合级验收，不导出逐行记录。

示例：
    python scripts/validate_audit_workflows.py --output-dir .audit_tmp/workflow_runs
    python scripts/validate_audit_workflows.py --input examples/hscredit_yyp.xlsx --no-excel

默认按放款时间稳定排序后以 75%/25% 切分，训练期拟合分箱/编码/模型，
保留期仅转换与评估。输出唯一运行目录、严格 JSON、语义制品及聚合工作簿。
本脚本不是全库回归、跨版本认证或生产模型有效性结论。
"""

import argparse
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold

from hscredit.core.binning import OptimalBinning
from hscredit.core.binning.spec import BinSpec
from hscredit.core.encoders import OneHotEncoder, TargetEncoder, WOEEncoder
from hscredit.core.metrics import auc, ks, compute_bin_stats
from hscredit.core.metrics.aggregation import BinStatsAccumulator
from hscredit.core.metrics.monitoring import MonitoringBaseline
from hscredit.core.models import LogisticRegression, ScoreCard
from hscredit.core.rules.artifact import RuleArtifact
from hscredit.report import ModelReport, OverduePredictor, auto_feature_analysis, feature_bin_stats

FEATURES = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24"]
CATEGORY, TARGET, DATE, AMOUNT, OVERDUE = "商品类别", "FPD", "放款时间", "放款金额", "MOB1"
REQUIRED = [*FEATURES, CATEGORY, TARGET, DATE, AMOUNT, OVERDUE]


def json_value(value):
    """仅处理汇总结果；DataFrame 必须由调用方显式转换为聚合记录。"""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def require(condition, message):
    if not condition:
        raise WorkflowValidationError(message)


class WorkflowValidationError(ValueError):
    """本脚本生成的聚合级校验错误，消息不包含样本值。"""


def validate(args, run_dir):
    result = {
        "状态": "执行中",
        "运行目录": str(run_dir),
        "输入": str(args.input),
        "隐私策略": "仅读取8个约定字段；只导出聚合统计、模型指标和无样本的语义制品",
        "覆盖限定": "核心已改流程；不验证全部模块、逐行订单导出、图形布局、生产容量或模型泛化有效性",
        "环境": {
            "Python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        },
        "步骤": [],
    }
    state, aggregate_tables = {}, {}

    def run(name, function, depends=()):
        if any(key not in state for key in depends):
            result["步骤"].append({"步骤": name, "状态": "未执行", "原因": "前置步骤失败"})
            return
        started = perf_counter()
        try:
            summary = function()
            result["步骤"].append({"步骤": name, "状态": "通过", "汇总": summary, "耗时秒": perf_counter() - started})
        except Exception as exc:
            # 不导出 traceback、DataFrame 或输入值；异常摘要仅保留步骤与类型。
            reason = (
                str(exc)
                if isinstance(exc, WorkflowValidationError)
                else "业务接口或断言失败；为保护样本隐私，不导出底层异常中的输入值或逐行差异"
            )
            result["步骤"].append(
                {
                    "步骤": name,
                    "状态": "失败",
                    "异常类型": type(exc).__name__,
                    "原因": reason,
                    "耗时秒": perf_counter() - started,
                }
            )

    def load_and_split():
        require(args.input.is_file(), "验证数据文件不存在，请通过 --input 指定 hscredit_yyp.xlsx")
        headers = pd.read_excel(args.input, nrows=0).columns
        missing = [column for column in REQUIRED if column not in headers]
        require(not missing, f"验证数据缺少必需字段：{missing}")
        data = pd.read_excel(args.input, usecols=REQUIRED)
        original_rows = len(data)
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
        data = data.loc[data[TARGET].isin([0, 1])].copy()
        require(len(data) >= 20, "有效0/1标签不足20行，不能执行本验收流程")
        for column in [*FEATURES, AMOUNT, OVERDUE]:
            converted = pd.to_numeric(data[column], errors="coerce")
            require(not (data[column].notna() & converted.isna()).any(), f"字段“{column}”存在无法转为数值的非缺失内容")
            require(not np.isinf(converted.to_numpy(dtype=float)).any(), f"字段“{column}”包含无穷值")
            data[column] = converted
        require(data[AMOUNT].notna().all() and (data[AMOUNT] >= 0).all(), "放款金额必须为非缺失的非负数值")
        data[DATE] = pd.to_datetime(data[DATE], errors="coerce")
        require(data[DATE].notna().all(), "放款时间含缺失或无法解析日期，不能隐式替换时间切分口径")
        data = data.sort_values(DATE, kind="stable").reset_index(drop=True)
        cut = int(len(data) * args.train_fraction)
        train, test = data.iloc[:cut].copy(), data.iloc[cut:].copy()
        require(
            train[TARGET].nunique() == 2 and test[TARGET].nunique() == 2,
            "时间切分后的训练集/保留集必须各有两个标签；请显式调整 --train-fraction",
        )
        state.update(data=data, train=train, test=test)
        return {
            "输入行数": original_rows,
            "有效行数": len(data),
            "排除非法或缺失标签行数": original_rows - len(data),
            "读取字段数": len(REQUIRED),
            "训练行数": len(train),
            "保留行数": len(test),
            "切分口径": "放款时间稳定排序后按位置切分；同日记录可分属两侧，不代表完整业务OOT",
            "训练截止": train[DATE].max(),
            "保留起始": test[DATE].min(),
        }

    def model_chain():
        train, test = state["train"], state["test"]
        binner = OptimalBinning(method="quantile", max_n_bins=5, n_jobs=1).fit(train[FEATURES], train[TARGET])
        train_woe = binner.transform(train[FEATURES], metric="woe")
        test_woe = binner.transform(test[FEATURES], metric="woe")
        require(
            np.isfinite(train_woe.to_numpy()).all() and np.isfinite(test_woe.to_numpy()).all(), "WOE转换存在非有限值"
        )
        model = LogisticRegression(max_iter=500, statistics_level="coef", history_policy="summary", max_history=2).fit(
            train_woe, train[TARGET]
        )
        predictions = model.predict_proba(test_woe)[:, 1]
        require(np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all(), "预测概率无效")
        metrics = pd.DataFrame(
            [
                {
                    "数据集": "保留集",
                    "样本数": len(test),
                    "AUC": auc(test[TARGET], predictions, score_direction="higher_risk"),
                    "KS": ks(test[TARGET], predictions),
                }
            ]
        )
        state.update(binner=binner, model=model, train_woe=train_woe, test_woe=test_woe)
        aggregate_tables["保留集指标"] = metrics
        return {
            "特征数": len(FEATURES),
            "指标": metrics.to_dict("records"),
            "统计状态": model.statistics_status_["状态"],
            "说明": "固定参数接口验收，不用于生产授信结论",
        }

    def categorical_encoders():
        train, test = state["train"], state["test"]
        cv = KFold(3, shuffle=True, random_state=42)
        encoder = TargetEncoder(cols=[CATEGORY], training_mode="oof", cv=cv, n_jobs=1, random_state=42)
        encoded = encoder.fit_transform(train[[CATEGORY]], train[TARGET])
        require(
            encoder.oof_coverage_.all() and np.isfinite(encoded.to_numpy()).all(),
            "OOF编码未覆盖全部训练行或出现非有限值",
        )
        for fit_indices, valid_indices in cv.split(train):
            reference = (
                TargetEncoder(cols=[CATEGORY], n_jobs=1)
                .fit(train[[CATEGORY]].iloc[fit_indices], train[TARGET].iloc[fit_indices])
                .transform(train[[CATEGORY]].iloc[valid_indices])
            )
            np.testing.assert_allclose(encoded.iloc[valid_indices], reference)
        onehot = OneHotEncoder(cols=[CATEGORY], sparse_output=True, return_df=False, min_frequency=2, n_jobs=1)
        train_sparse, test_sparse = onehot.fit_transform(train[[CATEGORY]]), None
        test_sparse = onehot.transform(test[[CATEGORY]])
        require(sparse.isspmatrix_csr(train_sparse) and sparse.isspmatrix_csr(test_sparse), "独热输出不是CSR")
        return {
            "OOF训练行数": len(encoded),
            "稀疏训练形状": list(train_sparse.shape),
            "稀疏保留形状": list(test_sparse.shape),
            "训练非零项": train_sparse.nnz,
        }

    def scorecard_deployment():
        train, data = state['train'], state['data']
        features = [*FEATURES, CATEGORY]
        binner = OptimalBinning(method='quantile', max_n_bins=5, n_jobs=1).fit(train[features], train[TARGET])
        encoder = WOEEncoder(cols=features, regularization=50, training_mode='oof', cv=3, n_jobs=1)
        encoded = encoder.fit_transform(binner.transform(train[features], metric='bins'), train[TARGET])
        require(TARGET not in encoded.columns, '评分卡编码输出包含目标列')
        card = ScoreCard(binner=binner, encoder=encoder, calculate_stats=False, decimal=8, clip=False,
                         lr_kwargs={'weight_type': 'cost', 'max_iter': 500})
        card.fit(encoded, train[TARGET], sample_weight=train[AMOUNT].to_numpy())
        raw = data[features]
        expected = np.asarray(card.predict(raw, input_type='raw'))
        require(np.isfinite(expected).all(), '真实数据评分存在非有限值')
        namespace = {}
        exec(card.export_deployment_code('python'), namespace)
        deployed = np.asarray([namespace['calculate_score'](row) for row in raw.to_dict('records')])
        with sqlite3.connect(':memory:') as connection:
            raw.to_sql('your_table', connection, index=False)
            sql_scores = np.asarray(connection.execute(card.export_deployment_code('sql')).fetchall())[:, 0]
        payload = json.loads(json.dumps(card.export(), ensure_ascii=False, allow_nan=False))
        restored = ScoreCard().load_rules(payload).predict(raw, input_type='raw')
        differences = {}
        for name, scores in [('Python', deployed), ('SQLite SQL', sql_scores), ('JSON规则重载', restored)]:
            delta = float(np.max(np.abs(np.asarray(scores) - expected)))
            require(np.isfinite(delta) and delta < 1e-7, f'{name}评分与本地预测不一致')
            differences[name] = delta
        return {'训练行数': len(train), '对账行数': len(raw), '特征数': len(features),
                '权重口径': '放款金额作为成本权重', '训练编码': 'OOF，regularization=50，cv=3',
                '最大绝对分差': differences, '逐行输出': False, 'Java真实集验收': False}

    def rules_roundtrip():
        binner, data = state["binner"], state["data"]
        codes = binner.transform(data[FEATURES], metric="indices")
        rule_count = 0
        for feature in FEATURES:
            for spec in BinSpec.from_binner(binner, feature).values():
                expected = codes[feature].to_numpy() == spec.bin_id
                np.testing.assert_array_equal(spec.mask(data).to_numpy(), expected)
                artifact = RuleArtifact(
                    spec.to_expression(),
                    bins=(spec,),
                    target_spec={"字段": TARGET, "坏样本": 1},
                    metrics={"命中数": int(expected.sum())},
                    validation="原始mask与bin一致",
                )
                path = run_dir / "rules" / f"rule_{rule_count:03d}.json"
                artifact.save(path)
                restored = RuleArtifact.load(path)
                np.testing.assert_array_equal(np.asarray(restored.predict(data), dtype=bool), expected)
                np.testing.assert_array_equal(restored.bins[0].mask(data).to_numpy(), expected)
                rule_count += 1
        return {"规则与BinSpec往返数量": rule_count, "验证样本数": len(data), "制品包含逐行数据": False}

    def monitoring_and_aggregation():
        train, test = state["train"], state["test"]
        baseline = MonitoringBaseline(max_n_bins=5).fit(train[FEATURES[0]])
        path = run_dir / "monitoring_baseline.joblib"
        baseline.save_artifact(path)
        restored = MonitoringBaseline.load_artifact(path)
        observed = baseline.evaluate(test[FEATURES[0]])
        batches = restored.evaluate_batches(
            test[FEATURES[0]].iloc[start : start + 73] for start in range(0, len(test), 73)
        )
        pd.testing.assert_frame_equal(observed["分箱明细"], batches["分箱明细"])
        aggregate_tables["冻结基准漂移"] = observed["分箱明细"]
        data = state["data"]
        codes = state["binner"].transform(data[FEATURES], metric="indices")[FEATURES[0]].to_numpy()
        accumulator = BinStatsAccumulator()
        for start in range(0, len(data), 97):
            accumulator.update(codes[start : start + 97], data[TARGET].iloc[start : start + 97])
        whole = compute_bin_stats(codes, data[TARGET].to_numpy(), round_digits=False)
        pd.testing.assert_frame_equal(accumulator.finalize(round_digits=False), whole)
        aggregate_tables["分块分箱对账"] = whole
        return {
            "冻结基准版本": baseline.version_,
            "保留集PSI": observed["PSI"],
            "监控分块与重载等价": True,
            "累计行数": accumulator.rows_seen_,
            "全量分块等价": True,
        }

    def overdue_amount():
        table = feature_bin_stats(
            state["data"],
            FEATURES[0],
            overdue=[OVERDUE],
            dpds=[7, 3, 0],
            amount=AMOUNT,
            method="quantile",
            n_jobs=1,
            overdue_operator=">",
        )
        require(
            not table.empty and np.isfinite(table.select_dtypes(include=np.number).to_numpy()).all(),
            "多标签金额统计为空或包含非有限数值",
        )
        aggregate_tables["多标签金额统计"] = table
        return {"目标": OVERDUE, "阈值": [7, 3, 0], "比较符": ">", "金额字段": AMOUNT, "聚合形状": list(table.shape)}

    def overdue_mean():
        train = state["train"]
        baseline = OverduePredictor(FEATURES[0], target=TARGET, method="quantile", coefficients=None, n_jobs=1).fit(
            train
        )
        adjusted = OverduePredictor(FEATURES[0], target=TARGET, method="quantile", coefficients="auto", n_jobs=1).fit(
            train
        )
        old, new = baseline.predict(train[FEATURES]), adjusted.predict(train[FEATURES])
        difference = float(np.max(np.abs(np.asarray(old) - np.asarray(new))))
        require(difference < 2e-6, "无分布变化时auto调整超出分箱率6位取整容差")
        require(abs(float(new.mean()) - float(train[TARGET].mean())) < 2e-6, "无分布变化时auto逾期率均值不守恒")
        return {
            "参考坏样本率": float(train[TARGET].mean()),
            "预测均值": float(new.mean()),
            "auto调整前后最大差": difference,
            "口径容差": 2e-6,
            "auto调整前后容差内等价": True,
            "预测模式": adjusted.prediction_mode_,
        }

    def strict_reports():
        if args.no_excel:
            return {"状态": "未覆盖Excel发布", "原因": "显式 --no-excel；不宣称严格Excel报告已验收"}
        train = state["train"]
        feature_result = auto_feature_analysis(
            train,
            features=[*FEATURES, CATEGORY],
            target=TARGET,
            excel_writer=str(run_dir / "strict_feature_report.xlsx"),
            pictures=[],
            output_dir=str(run_dir / "feature_assets"),
            bin_params={"method": "quantile", "n_jobs": 1},
            n_jobs=1,
            show_progress=False,
            mode="strict",
            return_result=True,
        )
        require(feature_result.complete, "严格特征报告包含未完成章节")
        report = ModelReport(
            state["model"],
            X_train=state["train_woe"],
            y_train=train[TARGET],
            X_test=state["test_woe"],
            y_test=state["test"][TARGET],
            n_jobs=1,
        )
        model_result = report.to_excel(
            str(run_dir / "strict_model_report.xlsx"),
            with_plots=False,
            n_bins=5,
            mode="strict",
            return_result=True,
            transactional=True,
            include_sample_records=False,
        )
        require(model_result.complete, "严格模型报告包含未完成章节")
        require(
            model_result.sections["生产订单测试用例"].status == "不适用"
            and model_result.metadata.get("包含订单样例明细") is False,
            "模型报告未确认关闭逐行订单明细",
        )
        aggregate_tables["特征报告状态"] = feature_result.status_table()
        aggregate_tables["模型报告状态"] = model_result.status_table()
        return {
            "特征报告章节": len(feature_result.sections),
            "模型报告章节": len(model_result.sections),
            "完整": True,
            "包含个人测试记录": False,
            "隐私限定": "正式接口include_sample_records=False，不验收逐行记录导出",
            "特征制品": feature_result.artifacts,
            "模型制品": model_result.artifacts,
                "模型完成清单": model_result.metadata.get("完成清单"),
        }

    run("读取与显式切分", load_and_split)
    run("三特征分箱WOE与LR保留集评估", model_chain, ("train",))
    run("折外目标编码与原生稀疏OneHot", categorical_encoders, ("train",))
    run("评分卡折外训练与部署执行对账", scorecard_deployment, ("train",))
    run("结构化规则及BinSpec保存重载", rules_roundtrip, ("binner",))
    run("冻结监控与分块统计对账", monitoring_and_aggregation, ("binner",))
    run("MOB1多标签金额分箱", overdue_amount, ("data",))
    run("OverduePredictor自动系数均值守恒", overdue_mean, ("train",))
    run("严格特征与模型报告", strict_reports, ("model",))
    if not args.no_excel and aggregate_tables:

        def export_aggregates():
            destination = run_dir / "aggregate_checks.xlsx"
            with pd.ExcelWriter(destination, engine="openpyxl") as writer:
                for name, table in aggregate_tables.items():
                    table.to_excel(writer, sheet_name=name[:31], index=True)
            return {"文件": str(destination), "聚合表数": len(aggregate_tables)}

        run("聚合验收工作簿", export_aggregates)
    result["状态"] = "通过" if all(step["状态"] == "通过" for step in result["步骤"]) else "失败"
    result["Excel报告已覆盖"] = not args.no_excel and any(
        step["步骤"] == "严格特征与模型报告" and step["状态"] == "通过" for step in result["步骤"]
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="HSCredit真实样例聚合级验收，不输出个人逐行记录")
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "hscredit_yyp.xlsx", help="真实样例文件")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / ".audit_tmp" / "workflow_runs", help="唯一运行子目录的父目录"
    )
    parser.add_argument("--train-fraction", type=float, default=0.75, help="时间排序后训练集比例，默认0.75")
    parser.add_argument("--no-excel", action="store_true", help="仅验证计算和语义制品，明确跳过Excel发布验收")
    args = parser.parse_args()
    if not 0.1 <= args.train_fraction <= 0.9:
        parser.error("--train-fraction 必须在0.1至0.9之间")
    args.input = args.input.expanduser().resolve()
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir.expanduser().resolve() / f"run-{timestamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    result = validate(args, run_dir)
    output = run_dir / "validation_result.json"
    output.write_text(json.dumps(json_value(result), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(
        json.dumps({"状态": result["状态"], "结果文件": str(output), "步骤数": len(result["步骤"])}, ensure_ascii=False)
    )
    return 0 if result["状态"] == "通过" else 1


if __name__ == "__main__":
    raise SystemExit(main())
