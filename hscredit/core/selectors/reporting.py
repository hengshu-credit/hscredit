"""筛选器的轻量、隔离报告协议与已拟合流程收集。

本模块不保存训练样本，不重算统计量，不导入 Excel 或报告发布依赖。
各表的列顺序和 dtype 是公开协议；算法特有指标使用长表表达。
"""

import copy
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from ...exceptions import NotFittedError, ValidationError

SCHEMA_VERSION = 1
DETAIL_COLUMNS = [
    "阶段编号",
    "阶段路径",
    "阶段名称",
    "筛选器",
    "筛选方法",
    "特征",
    "输入顺序",
    "计算域",
    "评估状态",
    "筛选结果",
    "决策来源",
    "指标名称",
    "指标值",
    "指标方向",
    "判定方式",
    "比较符",
    "有效阈值",
    "决策轮次",
    "关联特征",
    "筛选原因",
]
SUMMARY_COLUMNS = [
    "阶段编号",
    "阶段路径",
    "父路径",
    "阶段名称",
    "筛选器",
    "关系类型",
    "执行状态",
    "输入特征数",
    "选中特征数",
    "剔除特征数",
    "保留率",
    "耗时秒",
    "停止原因",
]
METRIC_COLUMNS = [
    "阶段路径",
    "特征",
    "轮次",
    "折号",
    "指标名称",
    "数值",
    "文本值",
    "统计口径",
    "比较符",
    "阈值",
    "条件结果",
    "有效性",
]
HISTORY_COLUMNS = [
    "阶段路径",
    "事件编号",
    "轮次",
    "特征",
    "动作",
    "候选标识",
    "候选子集",
    "指标名称",
    "指标值",
    "基准分",
    "候选分",
    "改善量",
    "有效阈值",
    "是否有效",
    "是否采纳",
    "关联特征",
    "原因",
    "补充信息",
]
TRACE_COLUMNS = ["阶段路径", "阶段名称", "特征", "筛选结果", "评估状态", "筛选原因"]
_INT_COLUMNS = {
    "阶段编号",
    "输入顺序",
    "决策轮次",
    "输入特征数",
    "选中特征数",
    "剔除特征数",
    "轮次",
    "折号",
    "事件编号",
}
_FLOAT_COLUMNS = {"指标值", "有效阈值", "保留率", "耗时秒", "数值", "阈值", "基准分", "候选分", "改善量"}
_BOOL_COLUMNS = {"条件结果", "是否有效", "是否采纳"}


def _table(records, columns):
    frame = pd.DataFrame(records, columns=columns, dtype=object)
    for name in columns:
        if name in _INT_COLUMNS:
            frame[name] = pd.array(frame[name], dtype="Int64")
        elif name in _FLOAT_COLUMNS:
            frame[name] = pd.array(pd.to_numeric(frame[name], errors="coerce"), dtype="Float64")
        elif name in _BOOL_COLUMNS:
            frame[name] = pd.array(frame[name], dtype="boolean")
        else:
            frame[name] = pd.Series([copy.deepcopy(value) for value in frame[name]], index=frame.index, dtype=object)
    return frame


def _detached_frame(frame):
    # pandas deep=True does not isolate nested mutable objects in object columns.
    result = frame.copy(deep=True)
    for name in result.select_dtypes(include="object").columns:
        result[name] = pd.Series([copy.deepcopy(value) for value in result[name]], index=result.index, dtype=object)
    return result


class SelectionReport:
    """完整筛选报告快照。

    **参数**

    :param summary: 固定结构的阶段汇总。
    :param details: 每阶段每个输入字段的决策明细。
    :param metrics: 多指标长表。
    :param history: 迭代事件表。
    :param metadata: 拟合快照和诊断，不含逐行训练样本。

    **属性**

    所有公开表和元数据属性均返回隔离副本，编辑报告不会改写模型。

    **参考样例**

    >>> report = collect_selection_report(fitted_selector)
    >>> report.details
    >>> report.save('筛选报告')
    """

    def __init__(self, summary=None, details=None, metrics=None, history=None, metadata=None):
        self._summary = _table([] if summary is None else summary, SUMMARY_COLUMNS)
        self._details = _table([] if details is None else details, DETAIL_COLUMNS)
        self._metrics = _table([] if metrics is None else metrics, METRIC_COLUMNS)
        self._history = _table([] if history is None else history, HISTORY_COLUMNS)
        self._metadata = copy.deepcopy(metadata or {})
        self._metadata.setdefault("schema_version", SCHEMA_VERSION)
        self._metadata.setdefault("完整", True)
        self._metadata.setdefault("诊断", [])

    @property
    def summary(self):
        return _detached_frame(self._summary)

    @property
    def details(self):
        return _detached_frame(self._details)

    @property
    def metrics(self):
        return _detached_frame(self._metrics)

    @property
    def history(self):
        return _detached_frame(self._history)

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    def copy(self):
        return SelectionReport(self.summary, self.details, self.metrics, self.history, self.metadata)

    def to_dict(self):
        """返回完整 Python 报告；JSON 类型协议由保存适配器处理。"""
        return {
            "阶段汇总": self.summary.to_dict("records"),
            "特征决策": self.details.to_dict("records"),
            "指标明细": self.metrics.to_dict("records"),
            "迭代事件": self.history.to_dict("records"),
            "元数据": self.metadata,
        }

    def get_feature_trace(self, *, max_rows=1000000):
        """按需展开全字段追踪；提前预算，避免阶段数乘字段数失控。"""
        features = list(dict.fromkeys(self._details["特征"].tolist()))
        if len(features) * len(self._summary) > max_rows:
            raise ValidationError("完整特征追踪超出 max_rows 预算，请使用稀疏特征决策表或提高预算")
        rows = []
        for _, stage in self._summary.iterrows():
            part = self._details[self._details["阶段路径"] == stage["阶段路径"]]
            mapping = {row["特征"]: row for row in part.to_dict("records")}
            for feature in features:
                source = mapping.get(feature)
                rows.append(
                    {
                        "阶段路径": stage["阶段路径"],
                        "阶段名称": stage["阶段名称"],
                        "特征": feature,
                        "筛选结果": (
                            source["筛选结果"] if source else ("未执行" if stage["执行状态"] != "已完成" else "未进入")
                        ),
                        "评估状态": source["评估状态"] if source else "未计算",
                        "筛选原因": source["筛选原因"] if source else stage["停止原因"],
                    }
                )
        return _table(rows, TRACE_COLUMNS)

    def save(self, path, **kwargs):
        """延迟加载保存适配器，以版本目录及原子清单发布完整报告。"""
        from ...report.selection_report import save_selection_report

        return save_selection_report(self, path, **kwargs)

    def to_report_result(self):
        """显式转换为报告层公共结果；仅此调用复制各章节表。"""
        from ...report.selection_report import to_report_result

        return to_report_result(self)


# 指标名称、方向、比较符、判定方式。主指标和有效阈值必须属于同一统计量。
_ADAPTERS = {
    "NullSelector": ("缺失率", "越小越优", "<", "数值阈值"),
    "ModeSelector": ("众数占比", "越小越优", "<", "数值阈值"),
    "CardinalitySelector": ("唯一值数", "越小越优", "<=", "数值阈值"),
    "TypeSelector": ("类型匹配", "不适用", "", "类型条件"),
    "RegexSelector": ("最终匹配", "不适用", "", "正则条件"),
    "VarianceSelector": ("总体方差", "越大越优", ">", "数值阈值"),
    "Chi2Selector": ("卡方统计量", "越大越优", ">=", "多条件"),
    "FTestSelector": ("F统计量", "越大越优", ">=", "多条件"),
    "MutualInfoSelector": ("互信息", "越大越优", ">=", "数值阈值"),
    "IVSelector": ("IV", "越大越优", ">=", "数值阈值"),
    "KSSelector": ("KS", "越大越优", ">=", "数值阈值"),
    "LiftSelector": ("方向改善量", "越大越优", ">=", "数值阈值"),
    "PSISelector": ("PSI", "越小越优", "<", "数值阈值"),
    "StabilityAwareSelector": ("综合分", "越大越优", ">=", "多条件"),
    "CorrSelector": ("保留优先权重", "越大越优", "", "相关冲突"),
    "VIFSelector": ("VIF", "越小越优", "<=", "迭代阈值"),
    "FeatureImportanceSelector": ("模型重要性", "越大越优", ">=", "重要性条件"),
    "NullImportanceSelector": ("零重要性得分", "越大越优", ">", "数值阈值"),
    "RFESelector": ("消除排名", "越小越优", "<=", "排名数量"),
    "SequentialFeatureSelector": ("选择标志", "不适用", "", "候选择优"),
    "BorutaSelector": ("显著性p值", "越小越优", "<=", "统计显著性"),
    "StepwiseSelector": ("1-p值", "越大越优", "", "候选模型准则"),
}
_SERIES_METRICS = {
    "scores_": "筛选得分",
    "valid_counts_": "有效样本数",
    "missing_counts_": "缺失数",
    "total_counts_": "总样本数",
    "mode_values_": "众数值",
    "mode_counts_": "众数计数",
    "cardinalities_": "唯一值数",
    "dtypes_": "原始类型",
    "matches_": "原始匹配",
    "variances_": "总体方差",
    "ranges_": "极差",
    "p_values_": "p值",
    "ranks_": "排名",
    "test_methods_": "检验方法",
    "category_counts_": "类别数",
    "metric_methods_": "计算口径",
    "psi_std_": "PSI标准差",
    "psi_max_": "PSI最大值",
    "reference_counts_": "基准样本数",
    "comparison_counts_": "比较样本数",
    "iv_scores_": "IV",
    "psi_scores_": "PSI",
    "combined_scores_": "综合分",
    "hits_": "命中次数",
    "support_": "强支持",
    "support_weak_": "弱支持",
    "decision_": "统计决策",
    "importance_scores_": "平均重要性",
    "actual_importances_": "真实重要性均值",
    "null_importances_": "随机重要性均值",
    "decision_vif_": "决策时VIF",
    "elimination_round_": "淘汰轮次",
    "candidate_scores_": "最近候选子集评分",
    "ranking_": "排名",
    "decision_importances_": "决策时重要性",
    "decision_correlations_": "冲突相关系数",
    "association_features_": "关联特征",
    "selection_reasons_": "决策解释",
    "raw_scores_": "原始检验统计量",
    "discrete_features_": "离散特征判定",
}


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number(value):
    value = _scalar(value)
    return value if isinstance(value, (int, float, bool)) else None


def _validity(value):
    number = _number(value)
    if number is None:
        return "文本" if value is not None and value is not pd.NA else "缺失"
    if np.isnan(number):
        return "缺失"
    if np.isposinf(number):
        return "正无穷"
    if np.isneginf(number):
        return "负无穷"
    return "有效"


def _mapping(value, features):
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    if isinstance(value, (np.ndarray, list, tuple)) and len(value) == len(features):
        return dict(zip(features, value))
    return {}


def _parameters(value, seen=None, depth=0):
    """不保留样本或已训练模型的参数描述；容器递归和估计器循环受控。"""
    seen = set() if seen is None else seen
    value = _scalar(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.DataFrame, pd.Series, np.ndarray)):
        result = {"对象类型": type(value).__name__, "形状": tuple(value.shape), "数据内容": "未保存"}
        if isinstance(value, pd.DataFrame):
            result["字段"] = list(value.columns)
        return result
    if id(value) in seen or depth > 6:
        return {"对象类型": type(value).__name__, "内容": "循环引用或超出深度"}
    seen.add(id(value))
    try:
        if isinstance(value, dict):
            return {copy.deepcopy(k): _parameters(v, seen, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_parameters(v, seen, depth + 1) for v in value]
        if hasattr(value, "get_params"):
            return {
                "对象类型": f"{type(value).__module__}.{type(value).__name__}",
                "参数": _parameters(value.get_params(deep=False), seen, depth + 1),
            }
        return {"对象类型": f"{type(value).__module__}.{type(value).__name__}", "内容": "未保存"}
    finally:
        seen.remove(id(value))


def _main_values(selector, features):
    cls = type(selector).__name__
    name, direction, comparison, mode = _ADAPTERS.get(cls, ("筛选得分", "由筛选器定义", "", "自定义筛选"))
    name = getattr(selector, "score_name_", name)
    direction = getattr(selector, "score_direction_", direction)
    direction = {"越大越好": "越大越优", "越小越好": "越小越优", "候选子集评分越大越好": "越大越优"}.get(
        direction, direction
    )
    values = _mapping(getattr(selector, "scores_", None), features)
    threshold = getattr(
        selector, "effective_threshold_", getattr(selector, "threshold_", getattr(selector, "threshold", None))
    )
    if cls == "VarianceSelector" and hasattr(selector, "variances_"):
        values = _mapping(selector.variances_, features)
    if cls == "FTestSelector" and hasattr(selector, "raw_scores_"):
        values = _mapping(selector.raw_scores_, features)
    if cls == "BorutaSelector":
        values = _mapping(getattr(selector, "p_values_", None), features)
        name, direction = "显著性p值", "越小越优"
        threshold = getattr(selector, "corrected_alpha_", float(selector.alpha) / max(1, len(values)))
    if cls == "StabilityAwareSelector":
        threshold = getattr(selector, "score_threshold", threshold)
    if cls == "RFESelector":
        threshold = 1
    if cls in {"TypeSelector", "RegexSelector", "CorrSelector", "SequentialFeatureSelector", "StepwiseSelector"}:
        threshold = None
    if cls == "FeatureImportanceSelector" and getattr(selector, "selection_mode_", None) == "前K个":
        threshold, comparison, mode = None, "", "排名数量"
    # VIF 的最终 scores 可能只有幸存字段；优先用拟合时保存的原始淘汰值补齐。
    if cls == "VIFSelector":
        values = dict(values)
        values.update(_mapping(getattr(selector, "decision_vif_", None), features))
        dropped = getattr(selector, "dropped_", None)
        if isinstance(dropped, pd.DataFrame) and "VIF值" in dropped:
            values.update(dict(zip(dropped["特征"], dropped["VIF值"])))
    return name, direction, comparison, mode, _number(threshold), values


def _single_report(selector):
    if not bool(getattr(selector, "_is_fitted", False)):
        raise NotFittedError("筛选器尚未拟合，无法取得筛选报告")
    features = list(getattr(selector, "_feature_names", getattr(selector, "feature_names_in_", [])))
    feature_set = set(features)
    selected = set(getattr(selector, "selected_features_", []))
    included = set(getattr(selector, "include_", []) or [])
    excluded = set(getattr(selector, "exclude_", []) or []) | set(getattr(selector, "forced_dropped_", []) or [])
    cls = type(selector).__name__
    is_composite = hasattr(selector, "selectors") or hasattr(selector, "stage_selectors_")
    relation = getattr(selector, "strategy", "sequential" if hasattr(selector, "stage_selectors_") else "single")
    metric_name, direction, comparison, mode, threshold, values = _main_values(selector, features)
    dropped = getattr(selector, "dropped_", None)
    dropped_map = (
        {row["特征"]: row for row in dropped.to_dict("records")}
        if isinstance(dropped, pd.DataFrame) and "特征" in dropped
        else {}
    )
    conditions = getattr(selector, "condition_results_", None)
    condition_map = conditions.to_dict("index") if isinstance(conditions, pd.DataFrame) else {}
    reason_map = _mapping(getattr(selector, "selection_reasons_", None), features)
    association_map = _mapping(getattr(selector, "association_features_", None), features)
    round_map = _mapping(getattr(selector, "elimination_round_", None), features)
    condition_values = {
        name: _mapping(getattr(selector, name, None), features)
        for name in ("iv_scores_", "psi_scores_", "ranks_", "percentile_values_", "scores_")
    }
    details, metrics, history = [], [], []
    domain = "分箱结果" if getattr(selector, "_binner_instance", None) is not None else "原始值"
    path = "selector"
    calculated_features = set(getattr(selector, "selection_input_features_", features))
    for position, feature in enumerate(features):
        raw = values.get(feature)
        numeric = _number(raw)
        valid = _validity(raw)
        kept = feature in selected
        source = "组合规则" if is_composite else "算法"
        reason = str(
            reason_map.get(
                feature, dropped_map.get(feature, {}).get("剔除原因", "满足筛选条件" if kept else "不满足筛选条件")
            )
        )
        evaluated = (
            "已计算" if feature in values and valid not in {"缺失"} else ("不适用" if is_composite else "未计算")
        )
        if feature in values and valid == "缺失":
            evaluated = "无效"
            if cls in {"StepwiseSelector", "SequentialFeatureSelector"} and not kept:
                evaluated = "未计算"
            if feature not in calculated_features:
                evaluated = "未计算"
        if feature in included and kept:
            source, reason = "强制保留", "用户强制保留"
        if feature in excluded and not kept:
            source = "强制剔除"
            reason = "强制剔除覆盖强制保留" if feature in included else "用户强制剔除"
        feature_conditions = condition_map.get(feature, {})
        failed = [name for name, ok in feature_conditions.items() if isinstance(ok, (bool, np.bool_)) and not ok]
        if failed and not kept and source == "算法":
            reason = "未通过条件：" + "、".join(map(str, failed))
        details.append(
            {
                "阶段编号": 1,
                "阶段路径": path,
                "阶段名称": cls,
                "筛选器": cls,
                "筛选方法": getattr(selector, "method_name", cls),
                "特征": feature,
                "输入顺序": position,
                "计算域": domain,
                "评估状态": evaluated,
                "筛选结果": "保留" if kept else "剔除",
                "决策来源": source,
                "指标名称": metric_name if not is_composite else None,
                "指标值": numeric,
                "指标方向": direction if not is_composite else None,
                "判定方式": mode if not is_composite else relation,
                "比较符": comparison,
                "有效阈值": threshold,
                "决策轮次": round_map.get(feature),
                "关联特征": association_map.get(feature, dropped_map.get(feature, {}).get("相关特征")),
                "筛选原因": reason,
            }
        )
        if feature in values:
            metrics.append(
                {
                    "阶段路径": path,
                    "特征": feature,
                    "指标名称": metric_name,
                    "数值": numeric,
                    "文本值": None if numeric is not None else raw,
                    "统计口径": domain,
                    "比较符": comparison,
                    "阈值": threshold,
                    "条件结果": None,
                    "有效性": valid,
                }
            )
        for condition, passed in feature_conditions.items():
            condition_value, condition_threshold, operator = numeric, threshold, comparison
            scope = "实际判定条件"
            if cls == "StabilityAwareSelector":
                if "IV" in condition:
                    condition_value, condition_threshold, operator = (
                        condition_values["iv_scores_"].get(feature),
                        selector.iv_threshold,
                        ">=",
                    )
                elif "PSI" in condition:
                    condition_value, condition_threshold, operator = (
                        condition_values["psi_scores_"].get(feature),
                        selector.psi_threshold,
                        "<=",
                    )
            if cls in {"Chi2Selector", "FTestSelector"}:
                if "数量" in condition:
                    condition_value = condition_values["ranks_"].get(feature)
                    condition_threshold = selector.k if selector.k != "all" else len(selector.ranks_)
                    operator, scope = "<=", "按分数降序、同分保持输入顺序的排名"
                elif "百分位" in condition:
                    condition_value = condition_values["percentile_values_"].get(feature)
                    condition_threshold = getattr(selector, "percentile_cutoff_", None)
                    operator, scope = (
                        ">或边界平局",
                        f"稠密序号百分位；边界名额={getattr(selector, 'percentile_tie_slots_', None)}；配置百分比={selector.percentile}",
                    )
                    if selector.percentile in {None, 100}:
                        operator, scope = "", "未限制百分位"
                elif "阈值" in condition:
                    condition_value = condition_values["scores_"].get(feature)
                    scope = "实际筛选统计量（原始NaN归0）" if cls == "FTestSelector" else scope
            metrics.append(
                {
                    "阶段路径": path,
                    "特征": feature,
                    "指标名称": condition,
                    "数值": _number(condition_value),
                    "比较符": operator,
                    "阈值": _number(condition_threshold),
                    "条件结果": passed if isinstance(passed, (bool, np.bool_)) else None,
                    "统计口径": scope,
                    "有效性": _validity(condition_value),
                }
            )
    for attr, label in _SERIES_METRICS.items():
        mapping = _mapping(getattr(selector, attr, None), features)
        if attr == "scores_":
            label = metric_name if cls != "BorutaSelector" else "命中比例"
            if cls not in {"BorutaSelector", "VarianceSelector", "FTestSelector"}:
                continue
            if cls == "VarianceSelector":
                label = "筛选方差"
            if cls == "FTestSelector":
                label = "实际F筛选统计量（原始NaN归0）"
        for feature, value in mapping.items():
            if feature not in feature_set:
                continue
            extra = {}
            if cls == "StabilityAwareSelector" and attr in {"iv_scores_", "psi_scores_", "combined_scores_"}:
                extra = {
                    "阈值": {
                        "iv_scores_": selector.iv_threshold,
                        "psi_scores_": selector.psi_threshold,
                        "combined_scores_": selector.score_threshold,
                    }[attr],
                    "比较符": "<=" if attr == "psi_scores_" else ">=",
                }
            metrics.append(
                {
                    "阶段路径": path,
                    "特征": feature,
                    "指标名称": label,
                    "数值": _number(value),
                    "文本值": None if _number(value) is not None else str(value),
                    "统计口径": "拟合时保存",
                    "有效性": _validity(value),
                    **extra,
                }
            )
    for attr in ("lift_detail_", "importance_details_", "fold_scores_"):
        frame = getattr(selector, attr, None)
        if not isinstance(frame, pd.DataFrame):
            continue
        if "特征" in frame:
            frame = frame.set_index("特征")
        for feature, row in frame.iterrows():
            if feature not in feature_set:
                continue
            for column, value in row.items():
                metrics.append(
                    {
                        "阶段路径": path,
                        "特征": feature,
                        "指标名称": "PSI" if attr == "fold_scores_" else str(column),
                        "折号": list(frame.columns).index(column) + 1 if attr == "fold_scores_" else None,
                        "数值": _number(value),
                        "文本值": None if _number(value) is not None else str(value),
                        "统计口径": {
                            "lift_detail_": "LIFT方向及覆盖",
                            "importance_details_": "真实及随机重要性",
                            "fold_scores_": "交叉验证分折",
                        }[attr],
                        "有效性": _validity(value),
                    }
                )
    if cls == "PSISelector":
        for attr, label in (("reference_counts_", "基准样本数"), ("comparison_counts_", "比较样本数")):
            counts = getattr(selector, attr, None)
            if isinstance(counts, pd.Series):
                for fold, (fold_name, count) in enumerate(counts.items()):
                    metrics.append(
                        {
                            "阶段路径": path,
                            "特征": None,
                            "折号": fold + 1,
                            "指标名称": label,
                            "数值": _number(count),
                            "统计口径": str(fold_name),
                            "有效性": _validity(count),
                        }
                    )
    for feature, row in dropped_map.items():
        for column, value in row.items():
            if column in {"特征", "剔除原因", "阈值", "筛选器", "筛选器类型", "筛选轮次", "筛选阶段", "筛选阶段名称"}:
                continue
            metrics.append(
                {
                    "阶段路径": path,
                    "特征": feature,
                    "指标名称": column,
                    "数值": _number(value),
                    "文本值": None if _number(value) is not None else str(value),
                    "统计口径": "淘汰时记录",
                    "阈值": _number(row.get("阈值")),
                    "有效性": _validity(value),
                }
            )
    events = getattr(selector, "selection_events_", [])
    for number, event in enumerate(events):
        record = {name: copy.deepcopy(event.get(name)) for name in HISTORY_COLUMNS}
        record.update({"阶段路径": path, "事件编号": number + 1})
        record["轮次"] = event.get("轮次", event.get("迭代"))
        history.append(record)
    final_events = {}
    for event in history:
        feature = event.get("特征")
        if feature in selected or feature in dropped_map:
            final_events[feature] = event
    for record in details:
        event = final_events.get(record["特征"])
        if event:
            record["决策轮次"] = event["轮次"]
            if event.get("关联特征") is not None:
                record["关联特征"] = event["关联特征"]
    stop = getattr(selector, "selection_stopping_reason_", "筛选完成")
    summary = [
        {
            "阶段编号": 1,
            "阶段路径": path,
            "父路径": None,
            "阶段名称": cls,
            "筛选器": cls,
            "关系类型": relation,
            "执行状态": "已完成",
            "输入特征数": len(features),
            "选中特征数": len(selected),
            "剔除特征数": len(features) - len(selected),
            "保留率": len(selected) / len(features) if features else None,
            "耗时秒": getattr(selector, "fit_duration_seconds_", None),
            "停止原因": stop,
        }
    ]
    snapshot = {
        "拟合参数": _parameters(selector.get_params(deep=False)),
        "输入字段": copy.deepcopy(features),
        "输出字段": [feature for feature in features if feature in selected],
        "拟合时间": datetime.now(timezone.utc).isoformat(),
        "fit_id": getattr(selector, "fit_id_", uuid4().hex),
        "算法指标": metric_name,
        "计算域": domain,
        "有效阈值": threshold,
        "样本数": getattr(selector, "n_samples_in_", None),
        "实际计算字段": list(getattr(selector, "selection_input_features_", values.keys())),
        "原始字段类型": {key: str(value) for key, value in getattr(selector, "input_dtypes_", {}).items()},
        "历史采集": "已采集" if hasattr(selector, "selection_events_") else "未采集",
    }
    # 筛选特征与实际 DataFrame 边界分别记录；透传目标不是一次特征选择。
    target = getattr(selector, "target", None)
    target_present = bool(getattr(selector, "target_present_at_fit_", False))
    input_columns = getattr(selector, "input_columns_at_fit_", None)
    transform_input = list(features if input_columns is None else input_columns)
    if target_present and target not in transform_input:
        transform_input.append(target)
    transform_output = list(snapshot["输出字段"])
    if target_present and not getattr(selector, "target_rm", False) and target not in transform_output:
        transform_output.append(target)
    snapshot.update(
        {
            "目标列": target,
            "拟合输入含目标": target_present,
            "目标列移除": bool(getattr(selector, "target_rm", False)),
            "目标字段": [target] if target_present else [],
            "转换输入字段": transform_input,
            "转换输出字段": transform_output,
        }
    )
    for attr in (
        "iv_normalizer_",
        "psi_normalizer_",
        "comparison_mode_",
        "corrected_alpha_",
        "report_events_truncated_",
        "report_events_bytes_",
        "report_candidates_truncated_",
        "report_decisions_truncated_",
        "decision_criteria_",
        "vif_history_truncated_",
        "vif_history_bytes_",
        "history_truncated_",
        "history_bytes_",
        "importance_runs_total_",
        "importance_runs_stored_",
        "importance_runs_truncated_",
        "selection_history_truncated_",
        "selection_history_bytes_",
        "converged_",
        "unresolved_features_",
        "n_cv_splits_",
        "candidate_failures_",
        "initial_criterion_",
        "final_criterion_",
    ):
        if hasattr(selector, attr):
            snapshot[attr] = _parameters(getattr(selector, attr))
    snapshot["历史策略"] = getattr(selector, "report_history", "未配置")
    snapshot["历史截断条数"] = int(getattr(selector, "report_events_truncated_", 0))
    snapshot["候选历史完整"] = (
        hasattr(selector, "selection_events_") and snapshot["历史策略"] == "full" and snapshot["历史截断条数"] == 0
    )
    return SelectionReport(
        summary, details, metrics, history, {"阶段快照": {path: snapshot}, "关系类型": relation, "明细完整": True}
    )


def _rebase(report, path, name=None, parent=None):
    result = report.copy()
    old_root = result._metadata.get("根路径") or (
        result._summary.iloc[0]["阶段路径"] if len(result._summary) else "selector"
    )

    def replace(value):
        if value is None or value is pd.NA:
            return value
        return path + str(value)[len(old_root) :] if str(value).startswith(old_root) else value

    for frame in (result._summary, result._details, result._metrics, result._history):
        frame["阶段路径"] = frame["阶段路径"].map(replace)
    result._summary["父路径"] = result._summary["父路径"].map(replace)
    root_mask = result._summary["阶段路径"] == path
    result._summary.loc[root_mask, "父路径"] = parent
    if name is not None:
        result._summary.loc[root_mask, "阶段名称"] = str(name)
        result._details.loc[result._details["阶段路径"] == path, "阶段名称"] = str(name)
    result._metadata["阶段快照"] = {replace(k): v for k, v in result._metadata.get("阶段快照", {}).items()}
    for boundary in result._metadata.get("字段边界", []):
        boundary["阶段路径"] = replace(boundary["阶段路径"])
    result._metadata["根路径"] = path
    return result


def _combine(reports, relation):
    records = {"summary": [], "details": [], "metrics": [], "history": []}
    metadata = {"关系类型": relation, "阶段快照": {}, "字段边界": [], "完整": True, "诊断": []}
    for report in reports:
        for name in records:
            records[name].extend(getattr(report, "_" + name).to_dict("records"))
        metadata["阶段快照"].update(report._metadata.get("阶段快照", {}))
        metadata["完整"] = metadata["完整"] and report._metadata.get("完整", True)
        metadata["诊断"].extend(report._metadata.get("诊断", []))
        metadata["字段边界"].extend(copy.deepcopy(report._metadata.get("字段边界", [])))
    positions = {row["阶段路径"]: index + 1 for index, row in enumerate(records["summary"])}
    for name in ("summary", "details"):
        for row in records[name]:
            row["阶段编号"] = positions[row["阶段路径"]]
    return SelectionReport(metadata=metadata, **records)


def _skipped(selector, path, name, parent, reason, *, incomplete=False, selector_type=None):
    summary = [
        {
            "阶段编号": 1,
            "阶段路径": path,
            "父路径": parent,
            "阶段名称": str(name),
            "筛选器": selector_type or type(selector).__name__,
            "关系类型": "single",
            "执行状态": "未拟合" if incomplete else "未执行",
            "停止原因": reason,
        }
    ]
    return SelectionReport(summary=summary, metadata={"完整": not incomplete, "诊断": [reason] if incomplete else []})


def _children(selector):
    if hasattr(selector, "selectors"):
        return [
            (str(item[0]), item[1]) if isinstance(item, tuple) and len(item) == 2 else (type(item).__name__, item)
            for item in selector.selectors
        ]
    if hasattr(selector, "stage_selectors_"):
        planned = getattr(selector, "planned_stages_", None)
        if planned is not None:
            return [(item["name"], selector.stage_selectors_.get(item["key"])) for item in planned]
        return list(selector.stage_selectors_.items())
    return []


def capture_selection_report(selector):
    """拟合完成时捕获结果；供基类在候选对象提交前调用。"""
    root = _single_report(selector)
    children = _children(selector)
    if not children:
        return root
    reports = [root]
    skipped = getattr(selector, "skipped_stages_", {})
    executed = getattr(selector, "executed_stages_", None)
    for position, (name, child) in enumerate(children):
        path = f"selector/{position + 1:02d}_{name}"
        reason = skipped.get(position) if isinstance(skipped, dict) else None
        if reason is not None or (executed is not None and position not in executed):
            planned = getattr(selector, "planned_stages_", None)
            selector_type = planned[position].get("selector") if planned is not None else None
            reports.append(
                _skipped(child, path, name, "selector", reason or "上游无剩余特征，未执行", selector_type=selector_type)
            )
        elif bool(getattr(child, "_is_fitted", False)):
            child_report = getattr(child, "selection_report_", None)
            if not isinstance(child_report, SelectionReport):
                child_report = capture_selection_report(child)
            reports.append(_rebase(child_report, path, name, "selector"))
        else:
            # Legacy composites did not persist skip metadata. Do not fabricate successful stages.
            reports.append(
                _skipped(child, path, name, "selector", "子筛选器未拟合，缺少明确的未执行记录", incomplete=True)
            )
    return _combine(reports, root._metadata["关系类型"])


def _selector_report(selector, strict):
    report = getattr(selector, "selection_report_", None)
    if isinstance(report, SelectionReport):
        return report.copy()
    if bool(getattr(selector, "_is_fitted", False)):
        report = capture_selection_report(selector)
        report._metadata["诊断"].append("旧对象未保存拟合报告快照；根据当前已拟合状态捕获，不能保证拟合后配置未变化")
        report._metadata["历史快照"] = False
        return report
    if strict:
        raise NotFittedError(f"{type(selector).__name__} 尚未拟合，无法取得筛选报告")
    return _skipped(selector, "selector", type(selector).__name__, None, "筛选器尚未拟合", incomplete=True)


def collect_selection_report(source, relation="auto", strict=True):
    """直接读取单筛选器、Pipeline、组合器或列表的完整报告，不执行 fit。

    列表默认独立比较。显式 sequential 必须满足字段链；intersection/union
    产生独立根决策，不把每个子报告的淘汰次数当唯一淘汰数量。
    """
    if relation not in {"auto", "independent", "sequential", "intersection", "union"}:
        raise ValidationError("relation 必须为 auto、independent、sequential、intersection 或 union")
    if isinstance(source, SelectionReport):
        return source.copy()
    if hasattr(source, "selected_features_") or hasattr(source, "get_selection_report"):
        result = _selector_report(source, strict)
        if strict and not result._metadata.get("完整", True):
            raise ValidationError("筛选报告不完整：" + "；".join(result._metadata.get("诊断", [])))
        return result
    is_pipeline = hasattr(source, "steps") and isinstance(source.steps, (list, tuple))
    if is_pipeline:
        items, effective = source.steps, "sequential"
        prefix = "pipeline"
    elif isinstance(source, (list, tuple)):
        items, effective = source, "independent" if relation == "auto" else relation
        prefix = "list"
    else:
        raise ValidationError("报告来源必须是筛选器、已拟合 Pipeline 或筛选器列表")
    reports, top, diagnostics, boundaries = [], [], [], []
    previous = None
    transform_input = None
    target_fields = set()
    for position, item in enumerate(items):
        name, child = item if isinstance(item, tuple) and len(item) == 2 else (type(item).__name__, item)
        if (
            isinstance(child, SelectionReport)
            and not (isinstance(item, tuple) and len(item) == 2)
            and len(child._summary)
        ):
            name = child._summary.iloc[0]["筛选器"]
        path = f"{prefix}/{position + 1:02d}_{name}"
        if child is None or (isinstance(child, str) and child == "passthrough"):
            continue
        is_selector = (
            isinstance(child, SelectionReport)
            or hasattr(child, "get_selection_report")
            or hasattr(child, "selected_features_")
        )
        if not is_selector and not hasattr(child, "steps"):
            if not is_pipeline:
                raise ValidationError("筛选器列表包含非筛选器对象")
            # Models terminate the chain. Transformations require an explicit field boundary.
            if hasattr(child, "transform"):
                incoming = list(getattr(child, "feature_names_in_", []))
                if transform_input is None:
                    transform_input = incoming
                try:
                    outgoing = list(child.get_feature_names_out())
                except (AttributeError, ValueError, TypeError):
                    outgoing = None
                if previous is not None and incoming and previous != incoming:
                    diagnostics.append(f"{path} 的输入字段与上游输出不一致")
                if outgoing is None:
                    diagnostics.append(f"{path} 变换后的字段边界无法验证")
                boundaries.append({"阶段路径": path, "类型": "变换", "输入字段": incoming, "输出字段": outgoing})
                previous = outgoing
            else:
                incoming = list(getattr(child, "feature_names_in_", []))
                if previous is not None and incoming and previous != incoming:
                    diagnostics.append(f"{path} 模型实际输入字段与上游转换输出不一致")
                count = getattr(child, "n_features_in_", None)
                if not incoming and count is None:
                    diagnostics.append(f"{path} 模型没有已拟合输入字段或数量，无法验证实际入模边界")
                if previous is not None and count is not None and count != len(previous):
                    diagnostics.append(f"{path} 模型实际输入数量与上游报告输出不一致")
                boundaries.append(
                    {
                        "阶段路径": path,
                        "类型": "模型",
                        "输入字段": incoming,
                        "输入数量": count,
                        "非筛选字段": [field for field in incoming if field in target_fields],
                    }
                )
            continue
        part = collect_selection_report(child, strict=strict)
        part = _rebase(part, path, name)
        stage = part._summary.iloc[0] if len(part._summary) else None
        snapshot = part._metadata.get("阶段快照", {}).get(path, {})
        incoming, outgoing = snapshot.get("输入字段"), snapshot.get("输出字段")
        actual_input = snapshot.get("转换输入字段", incoming)
        actual_output = snapshot.get("转换输出字段", outgoing)
        target_fields.update(snapshot.get("目标字段", []))
        if transform_input is None:
            transform_input = actual_input
        if effective == "sequential" and previous is not None and actual_input is not None and previous != actual_input:
            diagnostics.append(f"{path} 的输入字段不是上一阶段输出，不能解释为顺序筛选")
        previous = actual_output
        reports.append(part)
        top.append((stage, incoming, outgoing))
    if not reports:
        diagnostics.append("来源中没有可收集的筛选器报告")
    if diagnostics and strict:
        raise ValidationError("；".join(diagnostics))
    result = _combine(reports, effective)
    result._metadata["根路径"] = prefix
    result._metadata["字段边界"].extend(boundaries)
    result._metadata["诊断"].extend(diagnostics)
    result._metadata["完整"] = result._metadata["完整"] and not diagnostics
    if effective in {"intersection", "union"} and top:
        inputs = [entry[1] for entry in top]
        if any(item is None for item in inputs) or any(item != inputs[0] for item in inputs[1:]):
            raise ValidationError("集合组合要求每个筛选器有相同且可验证的原始输入字段")
        outputs = [set(entry[2] or []) for entry in top]
        final = set.intersection(*outputs) if effective == "intersection" else set.union(*outputs)
        features = inputs[0]
        details = [
            {
                "阶段编号": 1,
                "阶段路径": "list",
                "阶段名称": "列表集合决策",
                "筛选器": "集合组合",
                "筛选方法": effective,
                "特征": f,
                "输入顺序": i,
                "评估状态": "不适用",
                "筛选结果": "保留" if f in final else "剔除",
                "决策来源": "组合规则",
                "判定方式": effective,
                "筛选原因": "按声明的集合规则组合",
            }
            for i, f in enumerate(features)
        ]
        summary = [
            {
                "阶段编号": 1,
                "阶段路径": "list",
                "阶段名称": "列表集合决策",
                "筛选器": "集合组合",
                "关系类型": effective,
                "执行状态": "已完成",
                "输入特征数": len(features),
                "选中特征数": len(final),
                "剔除特征数": len(features) - len(final),
                "保留率": len(final) / len(features) if features else None,
            }
        ]
        root = SelectionReport(
            summary,
            details,
            metadata={"阶段快照": {"list": {"输入字段": features, "输出字段": [f for f in features if f in final]}}},
        )
        result._summary.loc[result._summary["父路径"].isna(), "父路径"] = "list"
        result = _combine([root, result], effective)
    if effective != "independent" and top:
        result._metadata["最终字段"] = (
            [f for f in top[0][1] if f in final] if effective in {"intersection", "union"} else top[-1][2]
        )
    if effective == "sequential" and top:
        incoming = (
            [field for field in transform_input if field not in target_fields] if transform_input is not None else None
        )
        outgoing = [field for field in previous if field not in target_fields] if previous is not None else None
        comparable = incoming is not None and outgoing is not None and set(outgoing).issubset(incoming)
        result._metadata["阶段快照"][prefix] = {
            "输入字段": incoming,
            "输出字段": outgoing,
            "转换输入字段": transform_input,
            "转换输出字段": previous,
            "目标字段": [field for field in transform_input or [] if field in target_fields],
        }
        # A flow node makes nested pipelines rebase as one subtree. It is not a selector decision stage.
        root_summary = [
            {
                "阶段编号": 1,
                "阶段路径": prefix,
                "阶段名称": "顺序流程",
                "筛选器": "流程",
                "关系类型": effective,
                "执行状态": "已完成" if result._metadata["完整"] else "不完整",
                "输入特征数": len(incoming) if incoming is not None else None,
                "选中特征数": len(outgoing) if outgoing is not None else None,
                "剔除特征数": len(incoming) - len(outgoing) if comparable else None,
                "保留率": len(outgoing) / len(incoming) if incoming and comparable else None,
            }
        ]
        result._summary.loc[result._summary["父路径"].isna(), "父路径"] = prefix
        result = _combine([SelectionReport(summary=root_summary), result], effective)
        result._metadata["根路径"] = prefix
        result._metadata["最终字段"] = outgoing
        result._metadata["最终转换字段"] = previous
    return result
