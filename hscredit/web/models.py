"""从 hscredit 公开模型枚举能力，并适配评分卡和规则模型的训练输入。"""

import importlib
import importlib.util
import inspect
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from hscredit._public_api import EXPORTS
from .parameters import parameter_specs

_ALIASES = {"LogisticRegression": "logistic", "RandomForest": "random_forest", "ExtraTrees": "extra_trees", "GradientBoosting": "gradient_boosting"}
_NAMES = {"LogisticRegression": "逻辑回归", "RandomForest": "随机森林", "ExtraTrees": "极端随机树", "GradientBoosting": "梯度提升树", "SVM": "支持向量机", "DecisionTreeClassifier": "决策树", "ScoreCard": "WOE 评分卡", "RoundScoreCard": "整数评分卡", "ProbabilityScoreCard": "概率评分卡", "ProbabilityCalibrator": "概率校准模型", "RulesClassifier": "规则集分类器", "CalibratedModel": "已拟合校准组合"}
_DEPENDENCIES = {"XGBoost": ["xgboost"], "LightGBM": ["lightgbm"], "CatBoost": ["catboost"], "NGBoost": ["ngboost"], "ResNet": ["torch", "rtdl_revisiting_models"], "FTTransformer": ["torch", "rtdl_revisiting_models"], "TabM": ["torch", "tabm", "rtdl_num_embeddings"], "TabNet": ["torch", "pytorch_tabnet"], "TabPFN": ["torch", "tabpfn"]}
_DEFAULTS = {
    "LogisticRegression": {"C": 1.0, "max_iter": 1000},
    "RandomForest": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 5},
    "ExtraTrees": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 5},
    "GradientBoosting": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    "DecisionTreeClassifier": {"max_depth": 5, "min_samples_leaf": 10},
    "SVM": {"C": 1.0, "kernel": "rbf"},
    "XGBoost": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    "LightGBM": {"n_estimators": 100, "num_leaves": 15, "learning_rate": 0.05},
    "CatBoost": {"iterations": 100, "depth": 4, "learning_rate": 0.05},
    "NGBoost": {"n_estimators": 100, "learning_rate": 0.05},
    "ResNet": {"max_epochs": 30, "batch_size": 256, "learning_rate": 0.001},
    "FTTransformer": {"max_epochs": 30, "batch_size": 256, "learning_rate": 0.001},
    "TabM": {"max_epochs": 30, "batch_size": 256, "learning_rate": 0.001},
    "TabNet": {"max_epochs": 30, "batch_size": 256, "learning_rate": 0.02},
    "TabPFN": {"n_estimators": 4, "device": "cpu"},
    "ScoreCard": {"pdo": 60, "base_score": 750},
    "RoundScoreCard": {"pdo": 60, "base_score": 750},
    "ProbabilityScoreCard": {"pdo": 20, "base_score": 600},
    "ProbabilityCalibrator": {"method": "platt", "calib_ratio": 0.2},
    "RulesClassifier": {"min_lift": 1.1, "top_n": 10},
}
CHOICES = {"kernel": ["rbf", "linear", "poly", "sigmoid"], "device": ["cpu", "auto", "cuda"], "method": ["platt", "isotonic", "beta", "histogram"]}


def _extra_model_params(cls, name, defaults):
    """合并包装类与实际原生估计器的构造参数；只提交用户选择的覆盖项。"""
    if cls is None:
        return {}
    exclude = set(defaults) | {
        "self",
        "target",
        "model",
        "prefit",
        "positive_class",
        "probability",
        "objective",
        "model_path",
        "params",
        "pipeline",
        "lr_model",
        "binner",
        "encoder",
        "scorecard",
        "rules",
        "parallel_config",
        "output_mode",
        "allow_writing_files",
        "callbacks",
    }
    extra = {}
    if name in {"RandomForest", "ExtraTrees", "DecisionTreeClassifier", "GradientBoosting", "SVM"}:
        native = cls().estimator_class
        extra.update(parameter_specs(native, exclude=exclude | {"init"}))
    # 部分提升库仅通过 **kwargs 声明原生选项；这些名称没有包装器默认值。
    native_options = {
        "XGBoost": "booster grow_policy max_leaves max_bin rate_drop skip_drop sample_type normalize_type one_drop max_cached_hist_node colsample_bynode sampling_method max_cat_to_onehot monotone_constraints interaction_constraints device",
        "LightGBM": "max_bin bagging_freq bagging_fraction bagging_seed feature_fraction feature_fraction_seed path_smooth linear_tree deterministic force_col_wise force_row_wise subsample_freq feature_fraction_bynode min_data_in_bin extra_trees max_cat_threshold monotone_constraints monotone_constraints_method",
        "CatBoost": "bootstrap_type subsample rsm leaf_estimation_iterations leaf_estimation_method random_score_type one_hot_max_size auto_class_weights class_weights task_type",
        "TabPFN": "balance_probabilities softmax_temperature average_before_softmax ignore_pretraining_limits",
    }
    extra.update({key: {"description": "原生估计器关键字参数，省略时由底层库决定"} for key in native_options.get(name, "").split()})
    if name == "RulesClassifier":
        return {**parameter_specs(cls, include={"logic", "threshold", "weights", "verbose"}), "miner_params": {"default": {}, "description": "训练折内 SingleFeatureRuleMiner 的参数，如 method、max_n_bins、min_samples"}}
    if name in {"ScoreCard", "RoundScoreCard"}:
        extra["clip"] = {"default": True, "description": "按配置的分数上下限裁剪评分"}
        extra["bin_params"] = {"default": {}, "description": "训练折内 OptimalBinning 的参数，如 method、special_codes、user_splits"}
    module = importlib.import_module(cls.__module__)
    native_api = {"XGBoost": ("xgb", "XGBModel"), "LightGBM": ("lgb", "LGBMClassifier"), "CatBoost": ("cb", "CatBoostClassifier")}
    if name in native_api:
        library, class_name = native_api[name]
        backend = getattr(getattr(module, library, None), class_name, None)
        if backend:
            # 数据/文件路径、回调与类别编码由平台工作流绑定。
            managed = {"missing", "input_borders", "output_borders", "train_dir", "snapshot_file", "save_snapshot", "snapshot_interval", "callback", "classes_count", "class_names", "target_border", "loss_function", "custom_loss", "silent", "random_seed", "thread_count", "nthread", "use_label_encoder", "validate_parameters", "importance_type", "num_parallel_tree", "feature_types"}
            extra.update(parameter_specs(backend, exclude=exclude | managed))
    extra.update(parameter_specs(cls, exclude=exclude))
    if "n_jobs" in extra:
        extra["n_jobs"]["default"] = 1
    if "random_state" in extra:
        extra["random_state"] = {"description": "省略时使用任务的随机种子"}
    return extra


def _fit_parameter_specs(cls, name):
    if cls is None or name in {"ScoreCard", "RoundScoreCard", "RulesClassifier"}:
        return {}
    allowed = {"sample_weight", "eval_metric", "cat_features", "finetune", "finetune_params"}
    specs = parameter_specs(cls.fit, include=allowed)
    if any(p.kind == p.VAR_KEYWORD for p in inspect.signature(cls.fit).parameters.values()):
        extras = {
            "XGBoost": ["base_margin", "early_stopping_rounds", "verbose", "eval_metric"],
            "LightGBM": ["init_score", "early_stopping_rounds", "eval_metric"],
            "CatBoost": ["baseline", "verbose", "eval_metric"],
        }
        if name in {"RandomForest", "ExtraTrees", "DecisionTreeClassifier", "SVM", "GradientBoosting"}:
            specs.update(parameter_specs(cls().estimator_class.fit, include={"check_input"}))
            specs["eval_metric"] = {"description": "训练过程使用的评估指标"}
        for key in extras.get(name, []):
            specs[key] = {"description": "传给当前提升模型 fit 的关键字参数"}
    for key in specs.keys() & {"sample_weight", "base_margin", "init_score", "baseline"}:
        specs[key]["description"] = "输入原样本的数值字段名或与原样本等长的数值数组，训练时按行同步切分"
    return specs


def model_setting(name, value):
    """JSON 对象键为文本，训练流水线中二分类标签为整数。"""
    if name == "class_weight" and isinstance(value, dict):
        return {int(label): weight for label, weight in value.items()}
    return value


@lru_cache(maxsize=1)
def model_catalog():
    """枚举所有公开概率预测模型，不依赖平台维护的四模型白名单。"""
    catalog = {}
    for name, module in EXPORTS["hscredit.core.models"].items():
        if not module.startswith("hscredit.core.models.") or any(part in module for part in ("losses", "tuning", "explainability")):
            continue
        missing = [d for d in _DEPENDENCIES.get(name, []) if importlib.util.find_spec(d) is None]
        cls, reason = None, ""
        try:
            cls = getattr(importlib.import_module(module), name)
        except (ImportError, RuntimeError, OSError, ValueError) as exc:
            if name not in _DEFAULTS:
                continue
            reason = f"依赖不可用：{exc}"
        if cls is not None and (not inspect.isclass(cls) or inspect.isabstract(cls) or not hasattr(cls, "fit") or not hasattr(cls, "predict_proba")):
            continue
        if missing:
            reason = "缺少依赖：" + "、".join(missing)
        if name == "CalibratedModel":
            reason = "仅用于已拟合模型的推理组合；训练请选择概率校准模型"
        if module.endswith("calibration.methods"):
            reason = "概率校准器组件；训练请选择概率校准模型并配置对应方法"
        params = dict(_DEFAULTS.get(name, {}))
        if not params and cls is not None and name != "CalibratedModel":
            params = {k: p.default for k, p in inspect.signature(cls).parameters.items() if p.default is not inspect.Parameter.empty and isinstance(p.default, (str, int, float, bool)) and k not in {"target", "objective", "history_policy"}}
        search = {k: [v] for k, v in params.items()}
        for key, values in {"C": [0.01, 0.1, 1, 10], "max_depth": [3, 5, 8], "learning_rate": [0.01, 0.05, 0.1], "n_estimators": [50, 100, 200]}.items():
            if key in search:
                search[key] = values if name != "TabPFN" else [2, 4, 8]
        category = "表格神经网络" if ".net" in module else "提升树" if ".boosting" in module else "评分卡与规则" if any(x in module for x in ("scorecard", "rules", "calibration")) else "经典模型"
        catalog[_ALIASES.get(name, name.lower())] = {
            "name": _NAMES.get(name, name),
            "class_name": name,
            "module": module,
            "group": category,
            "params": params,
            "extra_params": _extra_model_params(cls, name, params),
            "fit_params": _fit_parameter_specs(cls, name),
            "search": search,
            "choices": {k: CHOICES[k] for k in params if k in CHOICES},
            "available": not reason,
            "reason": reason,
            "note": "首次运行可能下载模型权重，受上游授权、样本量和设备要求限制" if name == "TabPFN" else "大样本训练可能耗时较长" if name == "SVM" else "",
        }
    return catalog


class _ConfiguredClassifier(ClassifierMixin, BaseEstimator):
    """让训练折内创建的估计器参数参与 sklearn 克隆及搜索。"""

    def get_params(self, deep=True):
        return {**super().get_params(deep=deep), **getattr(self, "options", {})}

    def set_params(self, **params):
        explicit = self._get_param_names()
        for name, value in params.items():
            if name in explicit:
                setattr(self, name, value)
            else:
                self.options[name] = value
        return self


class RawScoreCard(_ConfiguredClassifier):
    """将原始数据拟合显式转交 hscredit 评分卡，分箱始终在当前训练折内学习。"""

    bin_params = None  # 兼容旧版模型制品

    def __init__(self, rounded=False, pdo=60, base_score=750, bin_params=None, **options):
        self.bin_params = bin_params
        self.options = options
        self.rounded, self.pdo, self.base_score = rounded, pdo, base_score

    def fit(self, X, y):
        from hscredit.core.binning import OptimalBinning
        from hscredit.core.models import ScoreCard, RoundScoreCard

        cls = RoundScoreCard if self.rounded else ScoreCard
        bin_options = {"method": "quantile", "min_n_bins": 1, "min_bin_size": 1, "n_jobs": 1, **(getattr(self, "bin_params", None) or {})}
        self.model_ = cls(pdo=self.pdo, base_score=self.base_score, binner=OptimalBinning(**bin_options), **{"calculate_stats": True, **getattr(self, "options", {})})
        self.model_.fit(X, y, input_type="raw")
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    @property
    def report_estimator_(self):
        """提供拟合评分卡的刻度、分值与回归统计。"""
        return self.model_

    def predict_proba(self, X):
        return self.model_.predict_proba(X, input_type="raw")

    def predict_score(self, X):
        return self.model_.predict(X, input_type="raw")

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MinedRulesClassifier(_ConfiguredClassifier):
    """在训练折中挖掘规则，再委托 hscredit RulesClassifier 预测。"""

    miner_params = None  # 兼容旧版模型制品

    def __init__(self, min_lift=1.1, top_n=10, miner_params=None, **options):
        self.miner_params = miner_params
        self.options = options
        self.min_lift, self.top_n = min_lift, top_n

    def fit(self, X, y):
        from hscredit.report.mining import SingleFeatureRuleMiner
        from hscredit.core.models import RulesClassifier

        miner_params = {"method": "quantile", "min_n_bins": 1, "min_bin_size": 1, "min_samples": 1, "n_jobs": 1, **(getattr(self, "miner_params", None) or {})}
        miner = SingleFeatureRuleMiner(min_lift=self.min_lift, **miner_params)
        miner.fit(X, y)
        rules = miner.get_rules(top_n=self.top_n)
        if not rules:
            raise ValueError("当前训练折没有满足条件的规则，请降低最小 LIFT")
        self.model_ = RulesClassifier(rules=rules, n_jobs=1, **getattr(self, "options", {})).fit(X, y)
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_model(key, params, seed):
    entry = model_catalog()[key]
    if not entry["available"]:
        raise ValueError(entry["reason"])
    name = entry["class_name"]
    settings = {**entry["params"], **{k: model_setting(k, v) for k, v in params.items()}}
    if name in {"ScoreCard", "RoundScoreCard"}:
        return RawScoreCard(rounded=name == "RoundScoreCard", **settings), True
    if name == "RulesClassifier":
        return MinedRulesClassifier(**settings), True
    cls = getattr(importlib.import_module(entry["module"]), name)
    accepted = inspect.signature(cls).parameters
    if "random_state" in accepted:
        settings.setdefault("random_state", seed)
    if "n_jobs" in accepted:
        settings.setdefault("n_jobs", 1)
    if name in {"ProbabilityScoreCard", "ProbabilityCalibrator"}:
        from hscredit.core.models import LogisticRegression

        settings["model"] = LogisticRegression(calculate_stats=True, max_iter=1000, random_state=seed)
    if name == "CatBoost":
        settings["allow_writing_files"] = False
    return cls(**settings), entry["group"] == "表格神经网络"
