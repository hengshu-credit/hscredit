"""工作台的探索、策略、训练和三方报告服务，复用 hscredit 公共能力。"""

import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .targets import label_map, overdue_options, source_columns, training_label, validate_columns
from .storage import records, write_json
from .explore_time import time_distribution


from .parameters import binning_params, library_options, model_excel_options, model_report_options
from .models import make_model, model_catalog, model_setting

MODEL_CATALOG = model_catalog()


def labeled_data(data, config, labels=None):
    features = config.get("features", [])
    validate_columns(data, features)
    if set(features).intersection(source_columns(config)):
        raise ValueError("标签字段不能同时作为特征")
    if labels is None:
        labels = label_map(data, config, evaluation=True)
    target, series = training_label(labels)
    # 派生标签同名字段可能已被用作特征，只在训练副本上绑定到空闲列。
    if config.get("label_mode") == "overdue" or isinstance(config.get("target"), dict):
        while target in data:
            target = "_" + target
    valid = series.notna()
    clean = data.loc[valid].copy()
    clean[target] = series.loc[valid].astype(int)
    return clean, features, target, int((~valid).sum())


def explore(data, feature=None, other=None, bins=20, targets=None, positive_labels=None, freq="M"):
    return {**explore_summary(data, targets, positive_labels), **explore_charts(data, feature, other, bins, freq=freq), "correlation": numeric_correlation(data)}


def numeric_correlation(data):
    """早期探索接口和报告使用的数值 Pearson 矩阵。"""
    matrix = data.select_dtypes(include="number").corr()
    return {"columns": matrix.columns.tolist(), "values": json.loads(matrix.to_json(orient="values"))}


def explore_summary(data, targets=None, positive_labels=None, *, val_df=None, summary_params=None, **options):
    """计算仅依赖数据及标签配置的字段概况。"""
    config = dict(kind="explore", targets=targets or [], positive_labels=positive_labels or {}, **options)
    if config.get("label_mode") == "overdue":
        label_args = overdue_options(data, config)
    else:
        labels = label_map(data, config)
        label_args = {"target": labels} if labels else {}
    features = [name for name in data if name not in source_columns(config)]
    if not features:
        raise ValueError("请至少保留一个非标签字段用于字段概况")
    summary_options = {"features": features, "n_jobs": 1, "show_progress": False, **(summary_params or {})}
    if summary_options.get("model_params"):
        summary_options["model_params"] = {key: model_setting(key, v) for key, v in summary_options["model_params"].items()}
    if val_df is not None:
        if not any(col in val_df for col in features):
            raise ValueError("PSI 对照样本缺少可比较的特征字段")
        summary_options["val_df"] = val_df
    table = data.summary(**summary_options, **label_args)
    columns = []
    if isinstance(table.columns, pd.MultiIndex):
        for group in table.columns.get_level_values(0).unique():
            columns.append({"label": str(group), "children": [{"key": f"{group} / {name}", "label": str(name)} for name in table[group].columns]})
        table.columns = [f"{group} / {name}" for group, name in table.columns]
    else:
        columns = [{"key": str(name), "label": str(name)} for name in table.columns]
    return {"summary": records(table), "summary_columns": columns, "rows": len(data), "duplicates": int(data.duplicated().sum())}


def explore_charts(data, feature=None, other=None, bins=20, *, freq="M", feature_kind=None):
    """更新变量图表时不重复计算字段概况。"""
    result = {"feature": feature, "other": other}
    if feature:
        validate_columns(data, [feature])
        series = data[feature]
        if feature_kind is None and not pd.api.types.is_numeric_dtype(series) and re.search(r"date|time|日期|时间", feature, re.I):
            sample = series.dropna().head(100)
            feature_kind = "date" if len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9 else None
        if feature_kind == "date" or pd.api.types.is_datetime64_any_dtype(series):
            result.update(kind="date", **time_distribution(series, freq))
        elif pd.api.types.is_numeric_dtype(series):
            values = series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            hist, edges = np.histogram(values, bins=bins)
            result["distribution"] = [{"区间": f"{a:.4g} ~ {b:.4g}", "样本数": int(n)} for a, b, n in zip(edges[:-1], edges[1:], hist)]
            result.update(kind="numeric", valid=len(values), missing=int(series.isna().sum()), invalid=int(series.isin([np.inf, -np.inf]).sum()))
            result["distribution_note"] = f"当前字段有效数值 {len(values)} 行；缺失 {result['missing']} 行、无穷值 {result['invalid']} 行不参与直方图。"
        else:
            counts = series.astype("string").fillna("（缺失）").value_counts()
            result["distribution"] = [{"区间": str(k), "样本数": int(v)} for k, v in counts.head(30).items()]
            if len(counts) > 30:
                result["distribution"].append({"区间": "（其余类别合计）", "样本数": int(counts.iloc[30:].sum())})
            result.update(kind="category", missing=int(series.isna().sum()), invalid=0, valid=int(series.notna().sum()))
            result["distribution_note"] = "按全量样本统计，缺失值单独计数；显示前 30 类，其余类别合并。"
        if other:
            validate_columns(data, [other])
            if not all(pd.api.types.is_numeric_dtype(data[c]) for c in (feature, other)):
                raise ValueError("散点图需要选择两个数值字段")
            points = data[[feature, other]].replace([np.inf, -np.inf], np.nan).dropna()
            sampled = points.sample(min(3000, len(points)), random_state=42)
            result["scatter"] = json.loads(sampled.to_json(orient="values"))
            result["scatter_total"] = len(points)
    return result


def display_table(table):
    """展平报告表头供网页展示，不改变报告计算与 Excel 布局。"""
    table = table.copy()
    if not isinstance(table.index, pd.RangeIndex):
        table = table.reset_index()
    table.columns = [" / ".join(str(p) for p in col if str(p)) if isinstance(col, tuple) else str(col) for col in table.columns]
    return records(table.head(2000))


def report_tables(result):
    tables = {n: display_table(t) for n, t in result.items() if isinstance(t, pd.DataFrame)}
    tables["章节状态"] = records(result.status_table())
    return tables


def collect_report_images(directory):
    """把 hscredit 已生成的报告图片登记为可在网页预览的产物。"""
    import shutil

    images = []
    for index, path in enumerate(sorted((directory / "figures").rglob("*.png"))):
        filename = f"feature_plot_{index}.png"
        shutil.copyfile(path, directory / filename)
        images.append({"title": path.stem, "file": filename})
    return images


def strategy(data, config, directory, progress):
    from hscredit.report import auto_feature_analysis, ruleset_analysis
    from hscredit.report.mining import SingleFeatureRuleMiner
    from hscredit.excel import ExcelWriter, dataframe2excel

    clean, features, target, dropped = labeled_data(data, config)
    params = binning_params(config)
    progress(15, "正在生成 hscredit 分箱分析报告")
    result = auto_feature_analysis(clean, features=features, target=target, bin_params=params, pictures=["bin"], save=str(directory / "report.xlsx"), output_dir=str(directory / "figures"), n_jobs=1, show_progress=False, mode="strict", return_result=True)
    result.ensure_publishable()
    tables, artifacts = report_tables(result), ["report.xlsx"]
    if config["mining"]:
        progress(65, "正在挖掘规则并生成规则集分析")
        miner = SingleFeatureRuleMiner(**{"n_jobs": 1, **params}, target=target, features=features, min_lift=config["min_lift"], min_samples=config["min_samples"]).fit(clean)
        top = miner.get_top_rules(top_n=config["top_n"]).drop(columns=["规则报告"], errors="ignore")
        tables["规则明细"] = display_table(top)
        rules = miner.get_rules(top_n=config["top_n"])
        if rules:
            rule_table = ruleset_analysis(clean, rules, target=target, n_jobs=1)
            tables["规则集效果"] = display_table(rule_table)
            writer = ExcelWriter()
            dataframe2excel(top, writer, sheet_name="规则明细", index=False)
            dataframe2excel(rule_table, writer, sheet_name="规则集效果", index=True)
            writer.save(str(directory / "rules.xlsx"))
            artifacts.append("rules.xlsx")
    images = collect_report_images(directory)
    return {"tables": tables, "artifacts": artifacts + [image["file"] for image in images], "images": images, "dropped_target": dropped, "note": "分箱报告来自 auto_feature_analysis，规则集效果来自 ruleset_analysis；当前指标为样本内探索结果。"}


def normalize_model_input(data, numeric, categorical):
    """训练与导出模型共用同一输入规范，保留缺失值并处理新类别。"""
    data = data.copy()
    data[numeric] = data[numeric].replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        data[col] = data[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return data


class _LegacyFramePreprocessor(TransformerMixin, BaseEstimator):
    """在缺少 pandas 输出接口的 sklearn 中保留字段名、行索引和全空字段。"""

    def __init__(self, numeric, categorical):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X, y=None):
        if self.numeric:
            values = X[self.numeric].astype(float)
            self.medians_ = values.median().fillna(0.0)
            self.scaler_ = StandardScaler().fit(values.fillna(self.medians_))
        if self.categorical:
            from hscredit.core.encoders import OneHotEncoder as CreditOneHotEncoder

            values = X[self.categorical].astype(object)
            # 全空列没有众数，保留为编码器的缺失类别。
            self.modes_ = {}
            for name in self.categorical:
                modes = values[name].mode()
                if not modes.empty:
                    self.modes_[name] = modes.iloc[0]
            self.encoder_ = CreditOneHotEncoder(features=self.categorical, max_categories=50, handle_unknown="ignore", use_cat_names=False, n_jobs=1)
            self.encoder_.fit(values.fillna(self.modes_))
        return self

    def transform(self, X):
        parts = []
        if self.numeric:
            values = X[self.numeric].astype(float).fillna(self.medians_)
            scaled = pd.DataFrame(self.scaler_.transform(values), index=X.index, columns=self.numeric)
            parts.append(scaled.add_prefix("numeric__"))
        if self.categorical:
            values = X[self.categorical].astype(object).fillna(self.modes_)
            parts.append(self.encoder_.transform(values).add_prefix("category__"))
        return pd.concat(parts, axis=1)


def _model_preprocessor(numeric, categorical):
    if not hasattr(ColumnTransformer, "set_output"):
        return _LegacyFramePreprocessor(numeric, categorical)
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("category", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50))]), categorical))
    return ColumnTransformer(transformers).set_output(transform="pandas")


def model_pipeline(key, params, numeric, categorical, seed):
    estimator, native_input = make_model(key, params, seed)
    if native_input:
        return Pipeline([("model", estimator)])
    normalize = FunctionTransformer(normalize_model_input, kw_args={"numeric": numeric, "categorical": categorical})
    return Pipeline([("normalize", normalize), ("prepare", _model_preprocessor(numeric, categorical)), ("model", estimator)])


def model_fit_options(params, data, train_index):
    """先对齐原始数据的标签过滤及留出索引，ModelTuner 再按训练折切分。"""
    from hscredit.core.models._contracts import SAMPLE_PARAMETER_NAMES

    options = {}
    for key, value in params.items():
        if key in SAMPLE_PARAMETER_NAMES and value is not None:
            values = data[value] if isinstance(value, str) else pd.Series(value, index=data.index)
            value = values.loc[train_index].to_numpy()
        options["model__" + key] = value
    return options


def train(data, config, directory, progress, extra_frames=None):
    from hscredit.utils import init_setting
    from hscredit.core.models import ModelTuner
    from hscredit.report import auto_model_report, compare_models
    from hscredit.core.viz import roc_plot, ks_plot, lift_plot, confusion_matrix_plot
    from matplotlib import pyplot as plt

    init_setting()

    labels = label_map(data, config, evaluation=True)
    primary_name = next(iter(labels))
    clean, features, target, dropped = labeled_data(data, config, labels)
    X, y = clean[features].copy(), clean[target]
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    external = next(iter(extra_frames.items()), None) if config.get("test_dataset") else None
    if external:
        evaluation, _, test_target, _ = labeled_data(external[1], config)
        X_train, y_train, X_test, y_test = X, y, evaluation[features], evaluation[test_target]
        test_rows = evaluation.index.to_numpy()
    elif config.get("use_holdout", True):
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=config["test_size"], random_state=config["random_state"], stratify=y)
        X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
        test_rows = clean.index.to_numpy()[test_idx]
    else:
        X_train, y_train, X_test, y_test = X, y, X, y
        test_rows = clean.index.to_numpy()
    if y_train.value_counts().min() < config["cv"] or y_test.nunique() != 2:
        raise ValueError("每类训练样本数必须不少于交叉验证折数，且留出集必须包含两类样本")
    cv = StratifiedKFold(config["cv"], shuffle=True, random_state=config["random_state"])
    from hscredit.core.binning import OptimalBinning

    reference_rules = {}
    if numeric:
        reference = OptimalBinning(**{"n_jobs": 1, **binning_params(config)}).fit(X_train[numeric], y_train)
        reference_rules = {name: list(reference.splits_[name]) for name in numeric}
    trials, curves, fitted, tables, visuals, scores, artifacts = [], {}, {}, {}, {}, {}, []
    if config.get("workflow"):
        report_frames = {"训练集": data.loc[X_train.index]}
        if config.get("use_holdout", True) and not external:
            report_frames["留出测试集"] = data.loc[X_test.index]
        # 先在原样本上绑定标签，再按实际训练和留出行取值，兼容按行传入的标签数组。
        report_labels = {name: {key: y.loc[frame.index] for key, y in labels.items()} for name, frame in report_frames.items()}
        report_labels.update({name: label_map(frame, config, evaluation=True) for name, frame in (extra_frames or {}).items()})
        report_frames.update(extra_frames or {})
        X_train.sample(min(50, len(X_train)), random_state=config["random_state"]).to_parquet(directory / "explanation_background.parquet")
    report_train, report_test = X_train, X_test
    dimensions = [config[k] for k in ("amount", "date", "group") if config.get(k)]
    if dimensions and not config.get("workflow"):
        report_fields = list(dict.fromkeys([*features, *dimensions]))
        report_train = clean.loc[X_train.index, report_fields]
        test_source = evaluation if external else clean
        report_test = test_source.loc[X_test.index, report_fields]
    del labels
    for index, key in enumerate(config["models"]):
        name = MODEL_CATALOG[key]["name"]
        progress(5 + int(index / len(config["models"]) * 80), f"正在训练与评估{name}")
        estimator = model_pipeline(key, config["model_params"].get(key, {}), numeric, categorical, config["random_state"])
        if config.get("selector", "none") != "none":
            from hscredit.core.selectors import NullSelector, IVSelector

            selector_opts = library_options(config, "selector")
            if config["selector"] == "null":
                selector = NullSelector(threshold=config["selector_threshold"], **selector_opts)
            else:
                iv_options = {"n_jobs": 1, "bin_params": {"min_n_bins": 1, "min_bin_size": 1}, **selector_opts}
                selector = IVSelector(threshold=config["selector_threshold"], **iv_options)
            # 插入每折训练流水线；IV 不读取测试或评估标签。
            estimator.steps.insert(len(estimator.steps) - 1, ("selection", selector))
        space = (config["search_spaces"].get(key) or MODEL_CATALOG[key]["search"]) if config["search"] else {}
        fit_options = model_fit_options(config.get("model_fit_params", {}).get(key, {}), data, X_train.index)
        tuner = ModelTuner(model_class=estimator, fit_params=fit_options, search_space={f"model__{k}": [model_setting(k, candidate) for candidate in v] for k, v in space.items()}, metric="auc", cv=cv, random_state=config["random_state"], retention="summary", verbose=False, sampler_kwargs=library_options(config, "sampler"), **{"n_jobs": 1, "sampler": "random", **library_options(config, "tuner")})
        tuner.fit(X_train, y_train, n_trials=config["n_iter"] if config["search"] else 1, show_progress_bar=False, catch=())
        estimator = tuner.get_best_model()
        scores[key] = float(tuner.best_score_)
        for trial in tuner.study_.trials:
            trials.append({"模型": name, "试验编号": trial.number, "参数": json.dumps(trial.params, ensure_ascii=False), "交叉验证AUC": trial.value})
        probabilities = estimator.predict_proba(X_test)[:, list(estimator.classes_).index(1)]
        predictions = (probabilities >= config["threshold"]).astype(int)
        fitted[key] = estimator
        pd.DataFrame({"原始行号": test_rows, "真实标签": y_test.to_numpy(), "坏样本概率": probabilities, "预测标签": predictions}).to_csv(directory / f"predictions_{key}.csv", index=False, encoding="utf-8-sig")
        report = auto_model_report(estimator, X_train=report_train, y_train=y_train, X_test=report_test, y_test=y_test, features=features, rules=reference_rules, verbose=False, **model_report_options(config))
        tables[f"{name} / 评估指标"] = display_table(report.get_metrics())
        population = "独立测试集" if external else "留出集" if config.get("use_holdout", True) else "训练集"
        tables[f"{name} / {population}分箱"] = display_table(report.get_bin_table(dataset="test", max_n_bins=config["max_n_bins"], **library_options(config, "model_bins")))
        importance = report.get_feature_importance(top_n=30)
        if importance.empty and "prepare" in estimator.named_steps:
            transformed_train = estimator[:-1].transform(X_train)
            transformed_test = estimator[:-1].transform(X_test)
            encoded_report = auto_model_report(estimator.steps[-1][1], X_train=transformed_train, y_train=y_train, X_test=transformed_test, y_test=y_test, verbose=False, **model_report_options(config))
            importance = encoded_report.get_feature_importance(top_n=30)
        tables[f"{name} / 特征重要性"] = display_table(importance)
        for section, table in report.get_training_process().items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                tables[f"{name} / {section}"] = display_table(table)
        if not config.get("workflow"):
            status = report.to_excel(str(directory / f"report_{key}.xlsx"), mode="strict", return_result=True, **model_excel_options(config))
            status.ensure_publishable()
            tables[f"{name} / 报告状态"] = records(status.status_table())
        visuals[key] = {"name": name, "images": [], "importance": display_table(importance), "bins": tables[f"{name} / {population}分箱"]}
        plots = [
            ("ROC", lambda: roc_plot(y_test, probabilities, title=f"{name} · {population} ROC", label=name, score_direction="higher_risk", **{"figsize": (8, 5), **library_options(config, "roc")})),
            ("KS", lambda: ks_plot(probabilities, y_test, title=f"{name} · {population} KS", score_direction="higher_risk", **{"figsize": (8, 5), "curve": "ks", **library_options(config, "ks")})),
            ("LIFT", lambda: lift_plot(y_test, probabilities, title=f"{name} · {population} LIFT", **{"figsize": (8, 5), **library_options(config, "lift")})),
            ("混淆矩阵", lambda: confusion_matrix_plot(y_test, predictions, title=f"{name} · 混淆矩阵", **{"figsize": (8, 5), **library_options(config, "confusion")})),
        ]
        for label, draw in plots:
            fig = draw()
            if not hasattr(fig, "savefig"):
                fig = fig.figure if hasattr(fig, "figure") else plt.gcf()
            if label == "ROC":
                line = fig.axes[0].lines[-1]
                curves[key] = {"fpr": np.asarray(line.get_xdata()).tolist(), "tpr": np.asarray(line.get_ydata()).tolist()}
            filename = f"plot_{key}_{len(visuals[key]['images'])}.png"
            fig.savefig(directory / filename, dpi=110, bbox_inches="tight")
            plt.close(fig)
            visuals[key]["images"].append({"title": label, "file": filename})
            artifacts.append(filename)
        artifacts.append(f"predictions_{key}.csv")
        if not config.get("workflow"):
            artifacts.append(f"report_{key}.xlsx")
        if config.get("workflow"):
            from .experiments import evaluate_model

            if config.get("scorecard"):
                from hscredit.core.models import ProbabilityScoreCard

                fitted[key] = ProbabilityScoreCard(model=estimator, prefit=True, base_score=config["base_score"], pdo=config["pdo"], **library_options(config, "score_transform")).fit(X_train, y_train)
                tables[f"{name} / 评分转换"] = display_table(pd.DataFrame({"坏样本概率": probabilities, "评分": fitted[key].predict_score(X_test)}))
            extra = evaluate_model(fitted[key], key, X_train, y_train, report_frames, report_labels, config, directory, reference_rules)
            tables.update(extra["tables"])
            artifacts.extend(extra["artifacts"])
            visuals[key]["images"].extend(extra["images"])
            joblib.dump(fitted[key], directory / f"model_{key}.joblib")
            artifacts.append(f"model_{key}.joblib")
        tuner.release_training_data()
    best_key = max(scores, key=scores.get)
    comparison_mode = not config.get("workflow") or config.get("model_mode") == "compare"
    if comparison_mode:
        comparison = compare_models({MODEL_CATALOG[k]["name"]: m for k, m in fitted.items()}, X_train, y_train, X_test, y_test, save=str(directory / "comparison.xlsx"), n_jobs=1, **library_options(config, "comparison"))
        tables = {"模型对比": display_table(comparison), **tables}
        artifacts.append("comparison.xlsx")
    tables["交叉验证与搜索"] = trials
    joblib.dump(fitted[best_key], directory / "model.joblib")
    derived = config.get("label_mode") == "overdue" or isinstance(config.get("target"), dict)
    positive = 1 if derived else config.get("positive_labels", {}).get(primary_name, config.get("positive_label", 1))
    write_json(directory / "model_metadata.json", {"features": features, "numeric": numeric, "categorical": categorical, "target": primary_name, "positive_label": positive, "best_model": best_key, "config": config})
    artifacts += ["model.joblib", "model_metadata.json"]
    note = "调参由 ModelTuner 执行，最佳模型按训练集交叉验证 AUC 选择；评估与分箱来自 ModelReport，图表由 hscredit 绘图函数生成。"
    if comparison_mode:
        note += " 模型对比来自 compare_models。"
    if not config.get("use_holdout", True) and not external:
        note += " 本次未绑定测试集或划分留出集，测试位置展示训练样本内结果，不能视为独立验证。"
    return {"tables": tables, "curves": curves, "visuals": visuals, "cv_scores": scores, "best_model": best_key, "artifacts": artifacts, "dropped_target": dropped, "note": note}


def third_party(data, config, directory, progress):
    from hscredit.report import auto_feature_analysis

    validate_columns(data, config["features"])
    if set(config["features"]).intersection(source_columns(config)):
        raise ValueError("标签字段不能同时作为分析特征")
    dropped = 0
    if config["label_mode"] == "overdue":
        label = overdue_options(data, config)
    else:
        labels = label_map(data, config)
        if len(labels) == 1 and not isinstance(config.get("target"), dict):
            name, values = next(iter(labels.items()))
            dropped = int(values.isna().sum())
            data = data.loc[values.notna()].assign(**{name: values.dropna().astype(int)})
            label = {"target": name}
        else:
            label = {"target": labels}
    for key in ("date", "amount"):
        if config[key]:
            validate_columns(data, [config[key]])
    progress(15, "正在运行自动特征分析，生成分箱统计与图表")
    result = auto_feature_analysis(
        data,
        features=config["features"],
        **label,
        date=config["date"],
        amount=config["amount"],
        freq=config["freq"],
        corr=config["corr"],
        pictures=config["pictures"],
        dropna=config["dropna"],
        margins=config["margins"],
        long_format=config["long_format"],
        bin_params=binning_params(config),
        writer_params=library_options(config, "excel_writer"),
        **library_options(config, "feature_report"),
        save=str(directory / "report.xlsx"),
        output_dir=str(directory / "figures"),
        n_jobs=1,
        show_progress=False,
        mode="strict",
        return_result=True,
    )
    result.ensure_publishable()
    tables = {n: records(t.head(2000)) for n, t in result.items() if isinstance(t, pd.DataFrame)}
    tables["章节状态"] = records(result.status_table())
    images = collect_report_images(directory)
    return {"tables": tables, "artifacts": ["report.xlsx"] + [image["file"] for image in images], "images": images, "complete": result.complete, "dropped_target": dropped, "note": "页面展示 hscredit 生成的分析图表与章节结果；Excel 包含完整分析结果，页面每个章节最多预览 2000 行。"}
