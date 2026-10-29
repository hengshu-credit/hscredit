"""模型报告的预测方向、训练版本和统一指标契约。"""

import copy
import inspect
from functools import wraps

import numpy as np
import pandas as pd
from joblib import hash as stable_hash

from ..core.models._contracts import validate_sample_weight
from ..core.models._lifecycle import _fit_parameter_snapshot, learning_curve_table
from ..core.models.losses import BaseMetric, get_metric
from ..core.models.losses._sample_params import has_bound_sample_params, slice_sample_loss


def _constructor_params(model):
    """兼容原生 sklearn 和只实现无参 get_params 的自定义预测对象。"""
    method = getattr(model, "get_params", None)
    if not callable(method):
        try:
            parameters = inspect.signature(type(model).__init__).parameters
        except (TypeError, ValueError):
            return {}
        return {
            name: getattr(model, name)
            for name, parameter in parameters.items()
            if name != "self"
            and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and hasattr(model, name)
        }
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return method()
    return method(deep=False) if "deep" in parameters else method()


def _configuration_snapshot(value, _active=None):
    """报告配置逐项保留；不复用有长度/深度上限的训练日志摘要，也不遍历已学习权重。"""
    from enum import Enum
    from pathlib import Path

    if isinstance(value, Enum):
        return _configuration_snapshot(value.value, _active)
    if value is None or isinstance(value, (str, bool, int, float, np.number)):
        return value.item() if isinstance(value, np.generic) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type) or inspect.isroutine(value):
        return f"{value.__module__}.{value.__qualname__}"
    active = set() if _active is None else _active
    if id(value) in active:
        return "循环引用"
    active.add(id(value))
    try:
        if isinstance(value, dict):
            return {str(key): _configuration_snapshot(item, active) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            items = value.tolist() if isinstance(value, np.ndarray) else value
            if not isinstance(items, (list, tuple)):
                return _configuration_snapshot(items, active)
            return [_configuration_snapshot(item, active) for item in items]
        # 数据容器只保留元信息；构造配置不能隐式导出训练样本。
        if isinstance(value, (pd.DataFrame, pd.Series)) or hasattr(value, "tocsr"):
            return _fit_parameter_snapshot(value)
        params = _constructor_params(value)
        if params:
            return {
                "对象类型": f"{type(value).__module__}.{type(value).__qualname__}",
                "配置": _model_parameter_snapshot(value, active),
            }
        return _fit_parameter_snapshot(value)
    finally:
        active.remove(id(value))


def _model_parameter_snapshot(model, _active=None):
    """记录公开构造配置、实际原生参数及包装器训练后采用的组件配置。"""
    active = {id(model)} if _active is None else _active
    params = dict(_constructor_params(model))
    for key, value in getattr(model, "kwargs", {}).items():
        params.setdefault(key, value)
    result = _configuration_snapshot(params, active)
    native = getattr(model, "native_params_", None)
    if native:
        result["实际训练参数"] = _configuration_snapshot(native, active)
    children = {}
    for name in ("model_", "lr_model_", "binner_", "encoder_", "calibrator_", "transformer_", "score_transformer_"):
        child = getattr(model, name, None)
        if child is None or child is model or not _constructor_params(child):
            continue
        snapshot = _configuration_snapshot(child, active)
        if snapshot != result.get(name[:-1]):
            children[name] = snapshot
    if children:
        result["已拟合组件"] = children
    return result


def _model_stamp(model, _seen=None):
    _seen = set() if _seen is None else _seen
    if id(model) in _seen:
        return (type(model).__qualname__, "共享对象")
    _seen.add(id(model))
    if hasattr(model, "steps"):
        return tuple((name, _model_stamp(step, _seen)) for name, step in model.steps)
    if type(model).__module__.startswith("hscredit.core.models"):
        children = tuple(
            (key, _model_stamp(value, _seen))
            for key, value in vars(model).items()
            if key in {"model_", "base_model", "calibrator", "calibrator_", "lr_model_", "transformer_"}
            and value is not None
        )
        if hasattr(model, "fit_revision_") or children:
            return (
                type(model).__qualname__,
                getattr(model, "fit_revision_", None),
                getattr(model, "_is_fitted", True),
                children,
            )
    if type(model).__module__.split(".")[0] in {"sklearn", "xgboost", "lightgbm", "catboost", "ngboost"}:
        state = model.__getstate__() if callable(getattr(model, "__getstate__", None)) else vars(model)
        return stable_hash({key: value for key, value in state.items() if key != "tuner"})
    return None  # 任意自定义预测对象不假设其调用计数等属性代表训练状态。


def report_model_current(method):
    """在读取缓存或写文件前拒绝混用不同训练版本。"""

    @wraps(method)
    def checked(self, *args, **kwargs):
        self._assert_model_current()
        return method(self, *args, **kwargs)

    return checked


class ModelContextMixin:
    def _init_model_context(self, score_direction, metrics, sample_weight):
        if getattr(self, "task_type_", "binary") == "regression":
            from ._regression_model import configure

            configure(self, metrics, sample_weight, score_direction)
            return
        if score_direction not in (None, "higher_risk", "higher_safe"):
            raise ValueError("score_direction 必须为 higher_risk、higher_safe 或 None（按预测方法确定）")
        if score_direction is None:
            score_method = self.prediction_method == "predict_score" or (
                self.prediction_method in ("predict", "transform")
                and (hasattr(self.model, "scorecard_points") or hasattr(self.model, "direction_"))
            )
            owner = getattr(self.model, "score_transformer_", self.model)
            direction = getattr(owner, "direction_", getattr(owner, "direction", "descending"))
            score_direction = "higher_safe" if score_method and direction == "descending" else "higher_risk"
        self.score_direction = score_direction
        requested = [] if metrics is None else ([metrics] if isinstance(metrics, (str, BaseMetric)) else list(metrics))
        label_names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
        self.label_metrics_ = [
            item.lower() for item in requested if isinstance(item, str) and item.lower() in label_names
        ]
        self.metrics = [
            get_metric(item) for item in requested if not (isinstance(item, str) and item.lower() in label_names)
        ]
        names = [metric.name for metric in self.metrics] + [label_names[name] for name in self.label_metrics_]
        if len(set(names)) != len(names):
            raise ValueError("报告评估指标名称不能重复")
        if (self.metrics or self.label_metrics_) and self.prediction_method not in ("predict_proba", "predict_prob"):
            raise ValueError("统一金融指标需要坏样本概率，请使用 prediction_method='predict_proba'")
        self.sample_weight = copy.deepcopy(sample_weight)

    def _capture_model_context(self):
        self.evaluation_protocol_ = 2
        self._model_stamp_ = _model_stamp(self.model)
        model = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        self.model_context_ = {
            "模型类型": type(self.model).__name__,
            "训练修订": getattr(model, "fit_revision_", None),
            "预测方法": self.prediction_method if isinstance(self.prediction_method, str) else "自定义函数",
            "风险方向": self.score_direction,
            "构造参数": _fit_parameter_snapshot(self.model.get_params()) if hasattr(self.model, "get_params") else {},
            "实际参数": _fit_parameter_snapshot(getattr(model, "native_params_", {})),
            "内部转换": _fit_parameter_snapshot(getattr(model, "transformation_summary_", {})),
            "指标配置": [_fit_parameter_snapshot(metric) for metric in self.metrics],
            "指标权重": _fit_parameter_snapshot(self.sample_weight),
            "TOP名单规则": "人数向上取整，同分边界等比例计入；权重只用于名单内统计",
        }
        self.model_context_["最终估计器"] = type(model).__name__
        self.model_context_["指标配置"].extend(
            {"指标": name, "输入": "模型决策类别"} for name in getattr(self, "label_metrics_", [])
        )
        self.model_context_["最终模型参数"] = _model_parameter_snapshot(model)
        self.model_context_["任务类型"] = "回归" if getattr(self, "task_type_", "binary") == "regression" else "二分类"
        if getattr(self, "task_type_", "binary") == "regression":
            self.model_context_.pop("风险方向")
            self.model_context_.pop("TOP名单规则")
            self.model_context_["残差定义"] = "预测值减实际值"
        self._training_process_snapshot = self._collect_training_process()

    def _assert_model_current(self):
        if hasattr(self, "_model_stamp_") and _model_stamp(self.model) != self._model_stamp_:
            raise ValueError(
                "报告绑定的模型状态已变化（重新训练、失效或原生配置改变）；请重新创建 ModelReport，不能混用旧预测与新状态"
            )

    def _risk_values(self, values):
        values = np.asarray(values, dtype=float)
        return -values if self.score_direction == "higher_safe" else values

    def _metric_weight(self, key, label=None):
        spec = self.sample_weight
        if spec is None:
            return None
        dataset = self._datasets[key]
        if isinstance(spec, str):
            if spec not in dataset.X:
                raise ValueError(f"数据集 {key!r} 缺少评估权重列 {spec!r}")
            values = dataset.X[spec]
        elif isinstance(spec, dict):
            if key not in spec:
                raise ValueError(f"sample_weight 缺少数据集 {key!r} 的权重")
            values = spec[key]
        else:
            if len(self._datasets) != 1:
                raise ValueError("多个数据集的 sample_weight 请使用列名或 {数据集名称: 权重} 字典")
            values = spec
        weights = validate_sample_weight(values, len(dataset.y))
        if weights is not None:
            weights = weights[self._get_target_valid_mask(key, label)]
            if len(weights):
                weights = validate_sample_weight(weights, len(weights))
        return weights

    def _extra_metric_rows(self, label):
        rows = []
        for metric in self.metrics:
            row = {"统计项": metric.name}
            for key, dataset in self._datasets.items():
                y, p = self._get_valid_target_arrays(key, label)
                selected = slice_sample_loss(
                    metric, np.flatnonzero(self._get_target_valid_mask(key, label)), len(dataset.y)
                )
                row[dataset.label] = (
                    np.nan if not len(y) else selected.evaluate(y, p, sample_weight=self._metric_weight(key, label))
                )
            rows.append(row)
        if getattr(self, "label_metrics_", []):
            from ..core.models.base import _evaluate_binary_predictions

            label_rows = {}
            for key, dataset in self._datasets.items():
                if key not in self._label_prediction_cache:
                    owner = self.model
                    while getattr(owner, "steps", None):
                        owner = owner.steps[-1][1]
                    method = getattr(owner, "_predict_evaluation_labels", None)
                    if callable(method):
                        features = getattr(self.model, "feature_names_in_", self.feature_names)
                        X, pipeline = dataset.X[list(features)], self.model
                        while getattr(pipeline, "steps", None):
                            X = pipeline[:-1].transform(X)
                            pipeline = pipeline.steps[-1][1]
                        labels = np.asarray(method(X))
                    else:
                        labels = (dataset.prediction >= 0.5).astype(int)
                    self._label_prediction_cache[key] = labels
                valid = self._get_target_valid_mask(key, label)
                y, p = self._get_valid_target_arrays(key, label)
                predicted = self._label_prediction_cache[key][valid]
                if len(y):
                    values = _evaluate_binary_predictions(
                        y, p, predicted, metrics=self.label_metrics_, sample_weight=self._metric_weight(key, label)
                    )
                else:
                    names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
                    values = {names[name]: np.nan for name in self.label_metrics_}
                for name, value in values.items():
                    label_rows.setdefault(name, {"统计项": name})[dataset.label] = value
            rows.extend(label_rows.values())
        return rows

    @report_model_current
    def get_training_process(self):
        """返回可导出到Excel的配置、训练记录、最新曲线及调参过程表。

        **参考样例**

        >>> tables = report.get_training_process()
        >>> training = tables.get('训练记录')
        >>> search = tables.get('调参记录')
        """
        if not hasattr(self, "_training_process_snapshot"):
            self._training_process_snapshot = self._collect_training_process()
        return copy.deepcopy(self._training_process_snapshot)

    def _collect_training_process(self):
        model = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        result = {
            "模型口径": pd.DataFrame(
                [{"配置项": key, "配置值": str(value)} for key, value in self.model_context_.items()]
            )
        }
        preprocessing, selections, stages, preprocessing_params = [], [], [], []

        def collect_steps(pipeline, prefix=""):
            for name, step in getattr(pipeline, "steps", [])[:-1]:
                path = prefix + name
                inputs = getattr(step, "feature_names_in_", None)
                outputs = getattr(step, "selected_features_", None)
                if outputs is None and callable(getattr(step, "get_feature_names_out", None)):
                    try:
                        outputs = step.get_feature_names_out()
                    except (AttributeError, NotImplementedError):
                        outputs = None
                preprocessing.append(
                    {
                        "步骤": path,
                        "组件": type(step).__name__,
                        "输入字段": None if inputs is None else list(inputs),
                        "输出字段": None if outputs is None else list(outputs),
                    }
                )
                if hasattr(step, "get_params"):
                    from ._model_layout import configuration_table

                    config = configuration_table(_fit_parameter_snapshot(_constructor_params(step)))
                    config.insert(0, "步骤", path)
                    preprocessing_params.append(config)
                if callable(getattr(step, "get_selection_report", None)):
                    info = step.get_selection_report()
                    selections.append(
                        {
                            "步骤": path,
                            **{
                                key: info.get(key)
                                for key in ("筛选方法", "输入特征数", "选中特征数", "剔除特征数", "选中特征")
                            },
                        }
                    )
                    stages.extend({"步骤": path, **item} for item in info.get("阶段明细", []))
                if getattr(step, "steps", None):
                    collect_steps(step, path + "__")

        collect_steps(self.model)
        if preprocessing:
            result["预处理步骤"] = pd.DataFrame(preprocessing)
        if preprocessing_params:
            result["预处理参数"] = pd.concat(preprocessing_params, ignore_index=True)
        if selections:
            result["特征筛选记录"] = pd.DataFrame(selections)
        if stages:
            result["特征筛选阶段"] = pd.DataFrame(stages)
        availability = []
        for method, label in (
            ("get_training_history", "训练记录"),
            ("get_learning_curve", "训练曲线"),
            ("get_eval_results", "最终评估"),
        ):
            if callable(getattr(model, method, None)):
                table = getattr(model, method)()
                if not table.empty:
                    result[label] = table.copy(deep=True)
                availability.append(
                    {
                        "内容": label,
                        "状态": "已记录" if not table.empty else "未产生",
                        "说明": "" if not table.empty else "当前模型未保留此过程；重训无评估集时不产生验证曲线",
                    }
                )
            else:
                availability.append(
                    {"内容": label, "状态": "原生对象未提供统一过程接口", "说明": "不根据最终模型伪造历史过程"}
                )
        result["过程可用性"] = pd.DataFrame(availability)
        seen = {id(model)}
        for attr, prefix in (
            ("model_", "底层模型"),
            ("base_model", "底层模型"),
            ("lr_model_", "底层LR"),
            ("calibrator", "校准器"),
            ("calibrator_", "校准算法"),
            ("transformer_", "评分转换"),
        ):
            child = getattr(model, attr, None)
            if child is None or id(child) in seen:
                continue
            seen.add(id(child))
            for method, label in (
                ("get_training_history", "训练记录"),
                ("get_learning_curve", "训练曲线"),
                ("get_eval_results", "最终评估"),
            ):
                if callable(getattr(child, method, None)):
                    table = getattr(child, method)()
                    if not table.empty:
                        result[f"{prefix}{label}"] = table
        tuner = getattr(self.model, "tuner", None) or getattr(model, "tuner", None)
        if tuner is not None and getattr(tuner, "study_", None) is not None:
            rows = []
            fold_rows = []
            selected_curves = []
            selected_number = getattr(getattr(tuner, "best_trial_", None), "number", None)
            result["搜索指标"] = pd.DataFrame(
                [{"指标": metric.name, "方向": metric.direction} for metric in tuner.metrics]
            )
            validation = getattr(tuner, "validation_summary_", {})
            result["搜索设置"] = pd.DataFrame(
                [
                    {"配置项": "搜索指标", "配置值": "、".join(metric.name for metric in tuner.metrics)},
                    {
                        "配置项": "优化方向",
                        "配置值": "、".join(
                            f"{metric.name}：{'越大越好' if metric.direction=='maximize' else '越小越好'}"
                            for metric in tuner.metrics
                        ),
                    },
                    {"配置项": "验证方式", "配置值": validation.get("验证方式")},
                    {"配置项": "搜索评分数据集", "配置值": validation.get("搜索评分数据集", "交叉验证留出")},
                    {"配置项": "搜索选中试验", "配置值": selected_number},
                ]
            )
            relation = (
                "本报告模型是调参后重训模型"
                if getattr(tuner, "best_model_", None) is self.model
                else "搜索过程属于关联调参器，当前报告模型未替换为其重训模型"
            )
            result["模型口径"] = pd.concat(
                [result["模型口径"], pd.DataFrame([{"配置项": "与调参关系", "配置值": relation}])], ignore_index=True
            )
            public_history = getattr(tuner, "optimization_history_", None)
            history = (
                public_history.set_index("number")
                if public_history is not None and "number" in public_history
                else pd.DataFrame()
            )
            states = {"COMPLETE": "完成", "FAIL": "失败", "PRUNED": "剪枝", "RUNNING": "运行中", "WAITING": "等待"}
            for trial in tuner.study_.trials:
                rows.append(
                    {
                        "试验编号": trial.number,
                        "搜索选中": trial.number == selected_number,
                        "状态": states.get(trial.state.name, trial.state.name),
                        "各目标得分": trial.values,
                        "采样参数": trial.params,
                        "实际搜索参数": _fit_parameter_snapshot(
                            {
                                column[7:]: history.loc[trial.number, column]
                                for column in history
                                if column.startswith("params_") and trial.number in history.index
                            }
                        ),
                        "耗时秒": None if trial.duration is None else trial.duration.total_seconds(),
                        "错误信息": trial.user_attrs.get("错误信息", ""),
                    }
                )
                try:
                    details = tuner.get_trial_result(trial.number, load=False)
                except ValueError as exc:
                    fold_rows.append(
                        {"试验编号": trial.number, "折编号": None, "状态": "未保留逐折制品", "说明": str(exc)}
                    )
                    details = {"各折": []}
                for item in details.get("各折", []):
                    fold_rows.append(
                        {
                            "试验编号": trial.number,
                            "折编号": item["折编号"],
                            "状态": item.get("状态", "未完成"),
                            "训练输入样本数": len(item["训练位置"]) if "训练位置" in item else item.get("训练样本数"),
                            "验证评分样本数": len(item["验证位置"]) if "验证位置" in item else item.get("验证样本数"),
                            "评分数据集": item.get("搜索评分数据集", "交叉验证留出"),
                            "最佳迭代": item.get("最佳迭代"),
                            **item.get("指标", {}),
                        }
                    )
                    if trial.number == selected_number:
                        selected_curves.append(
                            learning_curve_table(
                                item.get("评估曲线", {}),
                                context={"试验编号": trial.number, "折编号": item["折编号"]},
                                names=item.get("训练记录", {}).get("评估集名称") or item.get("评估集名称"),
                            )
                        )
            result["调参记录"] = pd.DataFrame(rows)
            if fold_rows:
                result["各折指标"] = pd.DataFrame(fold_rows)
            if selected_curves and any(not table.empty for table in selected_curves):
                result["选中试验曲线"] = pd.concat(selected_curves, ignore_index=True)
            result["最终重训"] = pd.DataFrame(
                [
                    {"配置项": key, "配置值": str(value)}
                    for key, value in _fit_parameter_snapshot(getattr(tuner, "refit_params_", {})).items()
                ]
            )
        return copy.deepcopy(result)
