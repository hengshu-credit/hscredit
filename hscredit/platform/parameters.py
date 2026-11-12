"""平台可配置的底层调用、参数目录及任务作用域校验。"""

import inspect
import json
import re
from functools import lru_cache
from typing import Literal, get_args, get_origin

from hscredit.core.binning import OptimalBinning


def parameter_specs(target, *, exclude=(), include=None):
    """读取当前安装版本的具名关键字参数、可序列化默认值及中文参数说明。"""
    doc = inspect.getdoc(target) or ""
    if inspect.isclass(target):
        doc += "\n" + (inspect.getdoc(target.__init__) or "")
    descriptions = dict(re.findall(r"^\s*:param (\w+):[ \t]*(.+)$", doc, re.MULTILINE))
    specs = {}
    for name, param in inspect.signature(target).parameters.items():
        if name in exclude or (include is not None and name not in include):
            continue
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY) or param.default is param.empty:
            continue
        try:
            value = json.loads(json.dumps(param.default, allow_nan=False))
        except (TypeError, ValueError):
            continue
        specs[name] = {"default": value, "description": descriptions.get(name, "")}
        if get_origin(param.annotation) is Literal:
            specs[name]["choices"] = list(get_args(param.annotation))
    return specs


def _group(label, target, keys, **defaults):
    params = parameter_specs(target, include=keys.split())
    for name, default in defaults.items():
        params[name] = {**params.get(name, {}), "default": default}
    return {"label": label, "params": params}


BIN_COMMON = {"target", "method", "min_n_bins", "max_n_bins", "min_bin_size", "parallel_config", "custom_objective"}
SCOPES = {
    "single": ["binning", "bin_table", "bin_plot", "trend_plot", "target_plot"],
    "pair": ["binning", "pair_binning", "bin_table", "bin_plot", "trend_plot", "target_plot", "pair_plot"],
    "binary": ["binning", "binary_tree", "tree_report", "tree_plot", "bin_table", "bin_plot", "trend_plot", "target_plot"],
    "multiway": ["binning", "multiway_tree", "tree_report", "tree_plot", "bin_table", "bin_plot", "trend_plot", "target_plot"],
    "ruleset": ["binning", "ruleset_analysis", "rule_report", "swap_report"],
    "third_party": ["binning", "feature_report", "excel_writer"],
    "explore": ["summary"],
    "model": ["binning", "selector", "score_transform", "tuner", "model_report", "model_excel", "importance", "model_bins", "roc", "ks", "lift", "confusion", "sampler", "comparison"],
    "explain": ["sample_shap"],
    "legacy_strategy": ["binning"],
}


@lru_cache(maxsize=1)
def parameter_catalog():
    from hscredit.utils.pandas_extensions import _summary_method
    from sklearn.tree import DecisionTreeClassifier
    from hscredit.core.binning import OptimalBinning2D
    from hscredit.core.models import ModelTuner, TuningSampler, ProbabilityScoreCard
    from hscredit.core.selectors import IVSelector
    from hscredit.core.rules import Rule
    from hscredit.core.viz import bin_plot, bin_2d_plot, bin_overdues_plot, bin_trend_plot, roc_plot, ks_plot, lift_plot, confusion_matrix_plot
    from hscredit.core.viz.tree_plots import DecisionTreeViz
    from hscredit.report import compare_models
    import optuna
    from hscredit.core.viz import plot_model_sample_shap, plot_model_feature_importance
    from hscredit.excel import ExcelWriter
    from hscredit.report import auto_feature_analysis, ModelReport, ruleset_analysis, swap_out_report, feature_bin_stats
    from hscredit.report.mining import ManualTreeExtractor, MultiwayTreeExtractor

    methods = {}
    for name, cls in OptimalBinning._method_classes().items():
        params = parameter_specs(cls, exclude=BIN_COMMON)
        if name == "smooth":
            params["smoothing_method"] = parameter_specs(cls)["method"]
        methods[name] = params
    wrapper = parameter_specs(OptimalBinning, exclude=BIN_COMMON)
    wrapper.update({
        "lift_refine": {"description": "启用 LIFT 细化，默认随分箱方法和约束变化"},
        "lift_focus_weight": {"default": 3.0, "description": "LIFT 细化时的尾部风险权重"},
        "sample_stability_weight": {"default": 0.2, "description": "LIFT 细化时的样本稳定性权重"},
        "lift_refine_max_bins": {"description": "细化搜索的分箱数上限，默认沿用最多分箱数"},
        "monotonic_bonus_weight": {"default": 0.4, "description": "LIFT 细化时的单调性奖励权重"},
    })
    groups = {
        "binning": {"label": "一维分箱", "params": {**wrapper, "n_jobs": {**wrapper["n_jobs"], "default": 1}}, "methods": methods},
        "pair_binning": {"label": "二维分箱", "params": parameter_specs(OptimalBinning2D, exclude=BIN_COMMON)},
        "bin_plot": _group("单变量分箱图", bin_plot, "desc figsize colors anchor max_len fontdict hatch ending title show_data_points show_overall_bad_rate show_metric_summary metric_summary_layout show_rate_axis iv orientation"),
        "trend_plot": _group("时间与分组趋势图", bin_trend_plot, "sort_by sort_order max_groups figsize colors title show_overall show_stats orientation anchor max_cols"),
        "target_plot": _group("多标签分箱图", bin_overdues_plot, "figsize colors title show_stats max_cols"),
        "pair_plot": _group("二维分箱图", bin_2d_plot, "figsize colors title annot fontsize"),
        "tree_plot": _group("树图", DecisionTreeViz, "figsize title", show_stats=True, show_gini=True),
        "rule_report": _group("单规则与组合规则报告", Rule.report, "desc filter_cols margins long_format"),
        "model_bins": _group("评分分箱表", ModelReport.get_bin_table, "method margins"),
        "selector": _group("训练折内特征筛选", IVSelector, "include exclude force_drop n_jobs parallel_backend regularization bin_params"),
        "score_transform": _group("概率转评分", ProbabilityScoreCard, "method lower upper direction base_odds rate decimal clip"),
        "roc": _group("ROC 图", roc_plot, "figsize colors show_auc show_diagonal", figsize=[8, 5]),
        "ks": _group("KS 图", ks_plot, "figsize colors fontsize anchor curve", figsize=[8, 5], curve="ks"),
        "lift": _group("LIFT 图", lift_plot, "n_bins figsize colors show_baseline", figsize=[8, 5]),
        "confusion": _group("混淆矩阵图", confusion_matrix_plot, "figsize cmap normalize show_values show_metrics", figsize=[8, 5]),
        "comparison": _group("模型对比报告", compare_models, "sheet"),
        "bin_table": _group("分箱统计表", feature_bin_stats, "round_digits margins"),
        "binary_tree": {"label": "二叉树", "params": {**parameter_specs(DecisionTreeClassifier, exclude={"max_depth", "min_samples_leaf", "random_state"}), **parameter_specs(ManualTreeExtractor, include={"min_samples_split", "missing", "feature_map"})}},
        "multiway_tree": _group("多叉树", MultiwayTreeExtractor, "min_samples_split criterion alpha adjust_pvalue n_prebins max_candidates min_impurity_decrease feature_map missing_separate"),
        "tree_report": _group("树规则报告", ManualTreeExtractor.report, "leaf_only feature_map sheet project_desc data_source long_format apply_rule_filter"),
        "ruleset_analysis": _group("规则累积分析", ruleset_analysis, "filter_cols long_format"),
        "swap_report": _group("策略迭代报告", swap_out_report, "background summary describe rule_summary impact features current_pass_rate target_names theme_color sheet margins"),
        "feature_report": _group("三方特征报告", auto_feature_analysis, "data_summary_comment sheet start_col start_row feature_map image_table_gap_rows condition_color"),
        "excel_writer": _group("Excel 样式", ExcelWriter, "fontsize font theme_color opacity condition_color max_table_cells"),
        "summary": _group("字段概况", _summary_method, "max_n_bins psi_method group date freq psi_test_size percentiles random_state numeric_as_categorical force_numeric method bin_params fillna model_type model_params n_jobs parallel_backend", n_jobs=1),
        "model_report": _group("模型评估", ModelReport, "metrics score_direction show_weighted_metrics binning_reference cache_policy n_jobs parallel_backend", n_jobs=1, feature_psi_params={"min_bin_size": 1}),
        "model_excel": _group("模型 Excel 报告", ModelReport.to_excel, "sheet method with_plots model_name project_desc feature_map show_lift show_importance feature_contribution_label_max_features data_source include_sample_records training_sheet_index", with_plots=False, show_importance=False, include_sample_records=False),
        "tuner": _group("模型调参", ModelTuner, "sampler early_stopping_rounds validation_fraction n_jobs", sampler="random", n_jobs=1),
        "importance": _group("模型解释图", plot_model_feature_importance, "background_size left_top_n right_top_n show_dependence figsize title random_state shap_by_label hatch dpi", background_size=50, left_top_n=20, right_top_n=3, dpi=110),
        "sample_shap": _group("样本解释图", plot_model_sample_shap, "background_size max_display value_precision figsize title random_state dpi", background_size=50, dpi=110),
    }
    # 采样器还可接收由安装版本新增的关键字，由具体采样器的签名校验。
    samplers = {}
    for name, cls_name in TuningSampler.BUILTIN_SAMPLERS.items():
        sampler_cls = getattr(optuna.samplers, cls_name, None)
        if sampler_cls is not None:
            samplers[name] = parameter_specs(sampler_cls, exclude={"seed", "constraints_func", "after_trial_strategy", "child_generation_strategy", "elite_population_selection_strategy", "independent_sampler", "source_trials"})
    if "grid" in samplers:
        samplers["grid"]["search_space"] = {"description": "映射模型流水线参数名到候选数组，例如 model__max_depth"}
    groups["selector"]["params"]["n_jobs"]["description"] = "IV 筛选默认为 1，缺失率筛选采用筛选器默认预算"
    groups["selector"]["params"]["n_jobs"].pop("default")
    groups["sampler"] = {"label": "采样器构造参数", "params": {}, "variants": samplers, "custom": True}
    groups["tuner"]["params"]["sampler"]["choices"] = list(samplers)
    groups["pair_binning"]["params"]["n_jobs"]["default"] = 1
    for name in ("method_x", "method_y"):
        groups["pair_binning"]["params"][name]["choices"] = OptimalBinning.VALID_METHODS
    groups["summary"]["params"]["model_type"]["choices"] = [None, "randomforest", "logistic", "xgboost", "lightgbm", "catboost"]
    for group in ("summary", "model_excel", "model_bins"):
        groups[group]["params"]["method"]["choices"] = OptimalBinning.VALID_METHODS
    groups["model_report"]["params"]["binning_reference"]["choices"] = ["train", "independent"]
    groups["summary"]["params"]["psi_method"]["choices"] = ["random_split", "group", "date"]
    return {"methods": OptimalBinning.VALID_METHODS, "groups": groups, "scopes": SCOPES}


def library_options(config, scope):
    return config.get("library_params", {}).get(scope, {})


def model_report_options(config):
    defaults = {"n_jobs": 1, "feature_psi_params": {"min_bin_size": config["min_bin_size"]}}
    return {**defaults, **library_options(config, "model_report")}


def model_excel_options(config):
    defaults = {"with_plots": False, "show_importance": False, "include_sample_records": False}
    dimensions = {key: config[key] for key in ("amount", "date", "group", "freq") if config.get(key)}
    return {**defaults, **dimensions, "max_n_bins": config["max_n_bins"], **library_options(config, "model_excel")}


def binning_params(config):
    return {**{k: config[k] for k in ("method", "min_n_bins", "max_n_bins", "min_bin_size")}, **library_options(config, "binning")}


def validate_library_params(config):
    options = config.get("library_params", {})
    if not options:
        return
    kind = config["kind"]
    if kind == "strategy":
        kind = config["strategy_mode"] if config.get("workflow") else "legacy_strategy"
    allowed = set(SCOPES.get(kind, []))
    tuner = library_options(config, "tuner")
    if "sampler" in tuner and tuner["sampler"] not in parameter_catalog()["groups"]["tuner"]["params"]["sampler"]["choices"]:
        raise ValueError(f"当前版本不支持采样器 {tuner['sampler']}")
    if "selector" in options:
        selector = config.get("selector", "none")
        if selector == "none":
            raise ValueError("配置特征筛选参数时，请先选择筛选方法")
        if selector == "null" and {"regularization", "bin_params"}.intersection(options["selector"]):
            raise ValueError("regularization 和 bin_params 用于 IV 筛选")
    if "score_transform" in options and not (config.get("workflow") and config.get("scorecard")):
        raise ValueError("配置评分刻度时，请先启用概率转评分")
    if "importance" in options and not (config.get("workflow") and config.get("explain")):
        raise ValueError("配置模型解释图参数时，请先启用 SHAP 与变量交互")
    if "comparison" in options and config.get("workflow") and config.get("model_mode") != "compare":
        raise ValueError("模型对比报告参数仅用于多模型对比")
    for scope, values in options.items():
        if scope not in allowed:
            raise ValueError(f"当前任务不支持参数组 {scope}")
        group = parameter_catalog()["groups"][scope]
        supported = group["params"].keys() | group.get("methods", {}).get(config.get("method"), {}).keys()
        if scope == "summary" and values.get("model_params") is not None:
            from .models import model_catalog

            model = values.get("model_type")
            model = "random_forest" if model in ("rf", "randomforest") else model
            entry = model_catalog().get(model) if isinstance(model, str) else None
            if not entry or not isinstance(values["model_params"], dict):
                raise ValueError("概况模型参数需要模型类型及 JSON 对象")
            unknown_model = values["model_params"].keys() - (entry["params"].keys() | entry["extra_params"].keys())
            if unknown_model:
                raise ValueError(f"概况模型不支持参数：{'、'.join(sorted(unknown_model))}")
        if scope == "sampler":
            from hscredit.core.models import TuningSampler
            import optuna

            sampler = library_options(config, "tuner").get("sampler", "random")
            cls = getattr(optuna.samplers, TuningSampler.BUILTIN_SAMPLERS.get(sampler, ""), None) if isinstance(sampler, str) else None
            if cls is None:
                raise ValueError(f"当前版本不支持采样器 {sampler}")
            supported = {name for name, p in inspect.signature(cls).parameters.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)} - {"seed"}
        unknown = values.keys() - supported
        if unknown:
            raise ValueError(f"{group['label']}不支持参数：{'、'.join(sorted(unknown))}")
