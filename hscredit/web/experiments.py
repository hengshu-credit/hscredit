"""模型实验、独立样本评估与逐样本解释，统一委托模型和报告公共接口。"""

import joblib
import pandas as pd

from .analysis import train
from .parameters import library_options, model_excel_options, model_report_options
from .workflows import bound_frames, ReportBundle


def experiment(workspace, config, directory, progress):
    frames = bound_frames(workspace, config)
    training = frames.pop(next(iter(frames)))
    return train(training, config, directory, progress, extra_frames=frames)


def explanation_inputs(model, frame):
    """复用拟合时的预处理，让 SHAP 看见模型实际收到的字段与尺度。"""
    from sklearn.pipeline import Pipeline
    from hscredit.core.models import ProbabilityScoreCard

    if isinstance(model, ProbabilityScoreCard) and isinstance(getattr(model, "model_", None), Pipeline):
        model = model.model_
    if isinstance(model, Pipeline):
        if len(model.steps) > 1:
            frame = model[:-1].transform(frame)
        model = model.steps[-1][1]
    return model, frame


def evaluate_model(model, key, X_train, y_train, frames, mapped, config, directory, reference_rules):
    from hscredit.report import auto_model_report
    from hscredit.core.viz import plot_model_feature_importance

    bundle = ReportBundle(directory, prefix=key + "_")
    dimensions = [config[k] for k in ("amount", "date", "group") if config.get(k)]
    report_fields = list(dict.fromkeys([*config["features"], *dimensions]))
    for index, label in enumerate(next(iter(mapped.values()))):
        datasets = {}
        status_by_sample = {}
        for name, frame in frames.items():
            values = mapped[name][label].dropna()
            if values.nunique() < 2:
                bundle.table(f"{key} / {name} / {label}", pd.DataFrame([{"状态": "标签不足两个类别，无法计算 AUC/KS", "有效样本": len(values)}]))
                status_by_sample[name] = f"有效样本 {len(values)} 行，类别不足两类"
                continue
            datasets[name] = (frame.loc[values.index, report_fields], values)
            status_by_sample[name] = "已评估"
        title = f"跨样本指标 / {key} / {label}"
        status_row = pd.DataFrame([{"统计项": "评估状态", **status_by_sample}])
        if not datasets:
            if len(frames) > 1:
                bundle.table(title, status_row)
            continue
        report = auto_model_report(model, datasets=datasets, features=config["features"], rules=reference_rules, verbose=False, **model_report_options(config))
        metrics = report.get_metrics()
        if len(frames) > 1:
            comparison = metrics.reindex(columns=["统计项", *frames])
            bundle.table(title, pd.concat([comparison, status_row], ignore_index=True))
        for name in datasets:
            bundle.table(f"{key} / {name} / {label} / 评估指标", metrics[["统计项", name]])
        filename = f"report_{key}.xlsx" if index == 0 else f"report_{key}_label_{index}.xlsx"
        status = report.to_excel(str(directory / filename), mode="strict", return_result=True, **model_excel_options(config))
        status.ensure_publishable()
        bundle.table(f"{key} / {label} / 报告状态", status.status_table())
        bundle.workbook(filename)
    if config.get("explain"):
        X = X_train.sample(min(200, len(X_train)), random_state=config["random_state"])
        try:
            predictor, inputs = explanation_inputs(model, X)
            bundle.plot(f"{key} / 重要性、SHAP与变量交互（模型输入字段）", lambda: plot_model_feature_importance(predictor, inputs, y_train.loc[X.index], importance_source="shap", show=False, **{"background_size": min(50, len(X)), "left_top_n": 20, "right_top_n": 3, "dpi": 110, **library_options(config, "importance")}), dpi=library_options(config, "importance").get("dpi", 110))
        except (ImportError, ValueError, TypeError, RuntimeError) as exc:
            bundle.table(f"{key} / SHAP状态", pd.DataFrame([{"状态": "解释失败", "原因": str(exc)}]))
    return bundle.result()


def sample_explanation(workspace, config, directory, progress):
    from hscredit.core.viz import plot_model_sample_shap
    from .serving import job_model_path

    model = joblib.load(job_model_path(workspace, config["source_job"], config["model_file"]))
    source = workspace.get("jobs", config["source_job"])["config"]
    background_path = workspace.path("jobs", config["source_job"]) / "explanation_background.parquet"
    background = pd.read_parquet(background_path) if background_path.exists() else workspace.read_dataset(source["dataset_id"], source["dataset_revision"])[source["features"]]
    data = workspace.read_dataset(config["dataset_id"], config["dataset_revision"])[source["features"]]
    bundle = ReportBundle(directory)
    predictor, background = explanation_inputs(model, background)
    _, samples = explanation_inputs(model, data.iloc[config["sample_rows"]])
    for i, row in enumerate(config["sample_rows"]):
        progress(10 + int(i / len(config["sample_rows"]) * 80), f"解释第 {row + 1} 行样本")
        bundle.plot(f"第 {row + 1} 行预测解释", lambda: plot_model_sample_shap(predictor, samples.iloc[[i]], background, show=False, **{"background_size": min(50, len(background)), "dpi": 110, **library_options(config, "sample_shap")}), dpi=library_options(config, "sample_shap").get("dpi", 110))
    return bundle.result()
