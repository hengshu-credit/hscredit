"""特征筛选器基类.

定义特征筛选器的统一接口和通用方法。
所有筛选器都继承此类,确保API的一致性。

报告系统设计:
1. 单个/组合筛选器报告: 统一通过 report() 返回汇总和具体明细两个 DataFrame
2. 全局报告收集器: SelectionReportCollector 手动聚合多个筛选器的结果
3. 报告格式: 统一的中文格式,包含统计信息、选中/剔除特征、得分等

**参考样例**

>>> from hscredit.core.selectors.base import BaseFeatureSelector, SelectionReportCollector
>>> from hscredit.core.selectors import VarianceSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
>>> y = pd.Series(np.random.randint(0, 2, 100))
>>> selector = VarianceSelector(threshold=0.1)
>>> selector.fit(X, y)
>>> summary, details = selector.report()
>>> summary["选中特征数"]
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Union, List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
import copy
import inspect
from time import perf_counter
from uuid import uuid4
import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype
from pandas.api.types import is_numeric_dtype
from joblib import parallel_backend as joblib_parallel_backend
from sklearn.exceptions import NotFittedError as SklearnNotFittedError
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn import get_config
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import issparse

from ...exceptions import NotFittedError, ValidationError, DependencyError
from ...utils.data_contracts import prepare_xy
from ...utils.parallel import (
    ParallelWorkload,
    ParallelizableMixin,
    _resolve_current_n_jobs,
    _validate_parallel_backend,
    resolve_n_jobs,
    validate_parallel_config,
)

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
PROCESS_COLUMNS = list(dict.fromkeys(["记录类型"] + DETAIL_COLUMNS + METRIC_COLUMNS + HISTORY_COLUMNS))
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

    :param summary: 默认值为 ``None`` 。固定结构的阶段汇总
    :param details: 默认值为 ``None`` 。每阶段每个输入字段的决策明细
    :param metrics: 默认值为 ``None`` 。多指标长表
    :param history: 默认值为 ``None`` 。迭代事件表
    :param metadata: 默认值为 ``None`` 。拟合快照和诊断，不含逐行训练样本

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

    def report(self, *, strict: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按统一中文表头返回筛选汇总和具体明细两张表。

        :param strict: 默认 True，拒绝不完整快照；False 保留已有记录和 attrs 诊断。
        :return: ``(summary, details)``，列顺序分别为 SUMMARY_COLUMNS、PROCESS_COLUMNS。
            明细用“记录类型”区分特征决策、指标明细、迭代事件，保留不同算法的
            判定指标、阈值、条件、关联特征与轮次。同阶段按上述类型排列，类型内
            保持原始记录顺序。返回独立副本，不舍入、不重算、不生成文件。
        :raises ValidationError: strict 不是布尔值或严格模式下快照不完整。
        """
        if not isinstance(strict, bool):
            raise ValidationError("strict 必须为布尔值")
        metadata = self.metadata
        if strict and not metadata.get("完整", True):
            raise ValidationError("筛选报告不完整：" + "；".join(map(str, metadata.get("诊断", []))))
        summary = self.summary
        stage_info = summary.set_index("阶段路径")
        parts = []
        for attribute, kind in (("details", "特征决策"), ("metrics", "指标明细"), ("history", "迭代事件")):
            frame = getattr(self, attribute)
            if frame.empty:
                continue
            frame.insert(0, "记录类型", kind)
            for column in ("阶段编号", "阶段名称", "筛选器"):
                if column not in frame:
                    frame[column] = frame["阶段路径"].map(stage_info[column])
            parts.append(frame)
        process = _table(pd.concat(parts, ignore_index=True, sort=False) if parts else [], PROCESS_COLUMNS)
        if not process.empty:
            stage_order = {path: index for index, path in enumerate(summary["阶段路径"])}
            order = process["阶段路径"].map(stage_order).fillna(len(stage_order)).sort_values(kind="stable").index
            process = process.loc[order].reset_index(drop=True)
        summary.attrs = copy.deepcopy(metadata)
        process.attrs = copy.deepcopy(metadata)
        return summary, process

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

    def save(
        self,
        path,
        *,
        formats=("xlsx", "json"),
        overwrite=False,
        max_rows_per_sheet=1048575,
        max_excel_cells=5000000,
        max_sheets=200,
        by_selector=False,
        allow_incomplete=False,
    ):
        """以版本目录及原子清单归档快照，参数与 save_selection_report 一致。

        :param path: 归档名称或带 .xlsx/.json 后缀的路径；实际文件位于版本目录。
        :param formats: 默认 ('xlsx', 'json')，也支持单个格式字符串。
        :param overwrite: 默认 False；是否更新同名发布清单，旧版本仍保留。
        :param max_rows_per_sheet: 每表最大数据行数，默认 1048575，不含表头。
        :param max_excel_cells: 总单元格预算，默认 5000000，包含表头。
        :param max_sheets: 最多工作表数，默认 200。
        :param by_selector: 默认 False；是否额外按阶段导出特征决策表。
        :param allow_incomplete: 默认 False；是否允许归档不完整快照。
        :return: 包含实际路径、清单、制品、工作表映射及完整性的字典。
        """
        from ...report.selection_report import save_selection_report

        return save_selection_report(
            report=self,
            path=path,
            formats=formats,
            overwrite=overwrite,
            max_rows_per_sheet=max_rows_per_sheet,
            max_excel_cells=max_excel_cells,
            max_sheets=max_sheets,
            by_selector=by_selector,
            allow_incomplete=allow_incomplete,
        )

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


def _selector_family(selector):
    """按继承链选择算法口径，派生类保留真实类名并继承其筛选逻辑。"""
    return next(
        (base.__name__ for base in type(selector).__mro__ if base.__name__ in _ADAPTERS),
        type(selector).__name__,
    )


def _main_values(selector, features):
    cls = _selector_family(selector)
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
    family = _selector_family(selector)
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
            if family in {"StepwiseSelector", "SequentialFeatureSelector"} and not kept:
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
            if family == "StabilityAwareSelector":
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
            if family in {"Chi2Selector", "FTestSelector"}:
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
                    scope = "实际筛选统计量（原始NaN归0）" if family == "FTestSelector" else scope
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
            label = metric_name if family != "BorutaSelector" else "命中比例"
            if family not in {"BorutaSelector", "VarianceSelector", "FTestSelector"}:
                continue
            if family == "VarianceSelector":
                label = "筛选方差"
            if family == "FTestSelector":
                label = "实际F筛选统计量（原始NaN归0）"
        for feature, value in mapping.items():
            if feature not in feature_set:
                continue
            extra = {}
            if family == "StabilityAwareSelector" and attr in {"iv_scores_", "psi_scores_", "combined_scores_"}:
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
    if family == "PSISelector":
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
        "原始得分": copy.deepcopy(
            _mapping(
                getattr(selector, "scores_", None),
                getattr(selector, "feature_names_", getattr(selector, "selection_input_features_", features)),
            )
        ),
        "剔除明细列": list(dropped.columns) if isinstance(dropped, pd.DataFrame) else ["特征", "剔除原因"],
        "剔除明细": copy.deepcopy(dropped.to_dict("records")) if isinstance(dropped, pd.DataFrame) else [],
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
                raise ValidationError(f"子筛选器 {name} 缺少拟合报告快照，请重新调用 fit")
            reports.append(_rebase(child_report, path, name, "selector"))
        else:
            # 子步骤既无拟合快照又无明确跳过记录，不能视为成功。
            reports.append(
                _skipped(child, path, name, "selector", "子筛选器未拟合，缺少明确的未执行记录", incomplete=True)
            )
    return _combine(reports, root._metadata["关系类型"])


def _selector_report(selector, strict):
    report = getattr(selector, "selection_report_", None)
    if isinstance(report, SelectionReport):
        return report.copy()
    if bool(getattr(selector, "_is_fitted", False)):
        reason = "筛选器缺少拟合报告快照，请重新调用 fit"
        if strict:
            raise ValidationError(reason)
        return _skipped(selector, "selector", type(selector).__name__, None, reason, incomplete=True)
    if strict:
        raise NotFittedError(f"{type(selector).__name__} 尚未拟合，无法取得筛选报告")
    return _skipped(selector, "selector", type(selector).__name__, None, "筛选器尚未拟合", incomplete=True)


def collect_selection_report(source, relation="auto", strict=True):
    """直接读取单筛选器、收集器、Pipeline、组合器或列表的完整报告，不执行 fit。

    列表默认独立比较。显式 sequential 必须满足字段链；intersection/union
    产生独立根决策，不把每个子报告的淘汰次数当唯一淘汰数量。
    """
    if relation not in {"auto", "independent", "sequential", "intersection", "union"}:
        raise ValidationError("relation 必须为 auto、independent、sequential、intersection 或 union")
    if isinstance(source, SelectionReportCollector):
        source = source.get_report()
    if isinstance(source, SelectionReport):
        return source.copy()
    if isinstance(source, BaseFeatureSelector) or hasattr(source, "selected_features_"):
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
            or isinstance(child, BaseFeatureSelector)
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


def selection_report(
    selector: Any,
    *,
    relation: str = "auto",
    strict: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """统一收集筛选器、收集器或 Pipeline 的报告，返回汇总和具体过程两张表。

    :param selector: 已拟合的单筛选器、组合筛选器、Pipeline、筛选器列表或
        SelectionReport 快照。列表可混合上述筛选流程，也可使用 ``(名称, 筛选器)``
        为每个流程命名；不会创建筛选器，也不会调用 fit 或 transform。
    :param relation: 默认 'auto'，列表按独立筛选结果汇总，Pipeline 按实际步骤串行收集。
        串行列表须显式指定 'sequential' 并满足输入/输出字段连续；
        'intersection' / 'union' 按共同输入字段生成交集/并集根决策。
        'independent' 表示独立比较，不合成最终特征数。
    :param strict: 默认 True，拒绝未拟合或不完整的来源；False 保留已有记录，
        完整性及诊断写入两张表各自的 attrs，不添加诊断列。
    :return: ``(summary, process)`` 两个独立 DataFrame。
        summary 按阶段给出路径、父路径、筛选器、输入/选中/剔除特征数及保留率；
        根流程汇总与子阶段分别呈现，不能再次累加为唯一特征数量。
        process 用“记录类型”区分“特征决策”“指标明细”“迭代事件”，包含特征、
        指标值、阈值、筛选结果、原因及迭代轮次等原始字段。按汇总阶段顺序排列，
        同阶段按上述记录类型排列，各类型内部保持原始顺序；不做多对多拼接，
        不重复计算指标，不舍入，不修改筛选器。未记录的历史不会被补造。
        两表列名为中文，空表也保留固定列结构及 dtype。
    :raises ValidationError: 参数无效、字段链不连续或严格模式下来源不完整。
    :raises NotFittedError: 严格模式下筛选器尚未拟合。

    **参考样例**

    >>> summary, process = selection_report(selector=fitted_selector)
    >>> summary, process = selection_report([('缺失', null_selector), ('相关性', corr_selector)], relation='sequential')
    >>> summary, process = selection_report(fitted_pipeline)
    """
    if not isinstance(strict, bool):
        raise ValidationError("strict 必须为布尔值")
    if not isinstance(relation, str):
        raise ValidationError("relation 必须为筛选关系名称字符串")
    report = collect_selection_report(selector, relation=relation, strict=strict)
    return report.report(strict=strict)


def _fit_intersection_selector(task):
    """拟合一个彼此独立的 intersection 子筛选器。"""
    position, selector_name, selector, X, y, named = task
    selector.fit(X, y)
    dropped = None
    if hasattr(selector, "dropped_") and len(selector.dropped_) > 0:
        dropped = selector.dropped_.copy(deep=False)
        dropped["筛选器"] = selector_name
        dropped["筛选器类型"] = selector.__class__.__name__
    return position, selector_name, selector, named, set(selector.selected_features_), dropped


def _set_estimator_parallel_budget(estimator: Any, n_jobs: int) -> Any:
    """仅在隔离克隆上收敛估计器并行预算。

    最浅的有效并行层获得当前子预算；其下更深的并行层固定为 1，
    避免“外层候选 × meta × nested estimator”再次超额。同一最浅深度的并列分支均获得子预算。
    """
    parameters = estimator.get_params(deep=True)
    worker_aliases = {"n_jobs", "thread_count", "num_workers"}
    worker_names = [name for name in parameters if name.rsplit("__", 1)[-1] in worker_aliases]
    if not worker_names:
        return estimator

    depths = {name: name.count("__") for name in worker_names}
    shallowest_depth = min(depths.values())
    worker_parameters = {name: n_jobs if depth == shallowest_depth else 1 for name, depth in depths.items()}
    if worker_parameters:
        estimator.set_params(**worker_parameters)
    return estimator


def get_feature_importances(estimator) -> np.ndarray:
    """从任意模型中提取特征重要性.

    兼容以下模型类型:

    - 树模型 (feature_importances_): sklearn RandomForest/GradientBoosting,
      hscredit RiskModels, XGBClassifier, LGBMClassifier, CatBoostClassifier

    - 线性模型 (coef_): sklearn LogisticRegression, LinearSVC 等

    - 原生 xgboost Booster (get_score)

    - 原生 lightgbm Booster (feature_importance)

    - 原生 catboost CatBoost (get_feature_importance)

    **参数**

    :param estimator: 已训练的模型对象
    :return: 一维 numpy 数组，长度为特征数
    :raises ValidationError: 当无法从模型中提取重要性时
    """
    # 1. feature_importances_ — 最通用 (sklearn tree models, hscredit models, XGB/LGB/CB sklearn API)
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        if isinstance(importances, pd.Series):
            return importances.values.astype(float)
        return np.asarray(importances, dtype=float)

    # 2. coef_ — 线性模型 (sklearn LogisticRegression, LinearSVC, etc.)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            coef = np.linalg.norm(coef, axis=0)
        return np.abs(coef)

    # 3. 原生 xgboost Booster
    if hasattr(estimator, "get_score"):
        try:
            scores = estimator.get_score(importance_type="gain")
            if hasattr(estimator, "feature_names") and estimator.feature_names:
                return np.array([scores.get(f, 0.0) for f in estimator.feature_names], dtype=float)
            n = max(int(k.replace("f", "")) for k in scores) + 1 if scores else 0
            return np.array([scores.get(f"f{i}", 0.0) for i in range(n)], dtype=float)
        except Exception:
            pass

    # 4. 原生 lightgbm Booster
    if hasattr(estimator, "feature_importance") and callable(estimator.feature_importance):
        try:
            return np.asarray(estimator.feature_importance(importance_type="gain"), dtype=float)
        except Exception:
            pass

    # 5. 原生 catboost (Pool-based API)
    if hasattr(estimator, "get_feature_importance") and callable(estimator.get_feature_importance):
        try:
            return np.asarray(estimator.get_feature_importance(), dtype=float)
        except Exception:
            pass

    raise ValidationError(
        f"无法从 {type(estimator).__name__} 中提取特征重要性。"
        f"模型需要提供 feature_importances_、coef_ 属性，"
        f"或 get_score / feature_importance / get_feature_importance 方法。"
    )


class SelectionReportCollector:
    """收集已拟合筛选流程，并以基类的统一结构输出报告。

    **参数**

    :param name: 流程名称，默认 '特征筛选流程'。
    :param source: 初始筛选器、组合器、Pipeline、列表或快照，默认 None。
    :param relation: 默认 'auto'，多个来源独立比较；串行需指定 'sequential'。
    :param strict: 默认 True，拒绝未拟合及不完整来源。

    **属性**

    :ivar name: 流程名称。
    :ivar created_at: 收集器创建时间。

    **参考样例**

    >>> collector = SelectionReportCollector(relation='sequential')
    >>> collector.add_source(fitted_null, stage_name='缺失').add_source(fitted_iv, stage_name='IV')
    >>> summary, details = collector.report()
    """

    def __init__(self, name: str = "特征筛选流程", *, source=None, relation="auto", strict=True):
        if not isinstance(name, str):
            raise ValidationError("name 必须为字符串；报告来源请通过 source 传入")
        if not isinstance(relation, str) or relation not in {
            "auto",
            "independent",
            "sequential",
            "intersection",
            "union",
        }:
            raise ValidationError("未知筛选报告关系类型")
        if not isinstance(strict, bool):
            raise ValidationError("strict 必须为布尔值")
        self.name, self.relation, self.strict = name, relation, strict
        self.created_at = datetime.now()
        self._snapshots = []
        if source is not None:
            self.add_source(source)

    def add_source(self, source, stage_name=None):
        """保存来源的隔离快照并返回自身；不会重新拟合。

        :param source: 已拟合筛选器、组合器、Pipeline、列表或 SelectionReport。
        :param stage_name: 来源名称；None 时使用阶段序号。
        :return: 收集器自身，可链式添加来源。
        """
        if stage_name is not None and not isinstance(stage_name, str):
            raise ValidationError("stage_name 必须为字符串或 None")
        snapshot = collect_selection_report(source, relation=self.relation, strict=self.strict)
        snapshot.report(strict=self.strict)
        self._snapshots.append((stage_name or f"阶段{len(self._snapshots) + 1}", snapshot))
        return self

    def get_report(self) -> SelectionReport:
        """返回完整的隔离快照；空收集器返回标记为不完整的空快照。"""
        if not self._snapshots:
            return SelectionReport(metadata={"关系类型": "independent", "完整": False, "诊断": ["无筛选记录"]})
        return collect_selection_report(self._snapshots, relation=self.relation, strict=self.strict)

    def report(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """返回统一的汇总和明细两张表，使用构造时的 relation、strict 配置。"""
        return selection_report(self, strict=self.strict)

    def get_feature_trace(self, *, max_rows=1000000) -> pd.DataFrame:
        """按输入顺序返回跨阶段的字段追踪表；max_rows 限制展开后的行数。"""
        return self.get_report().get_feature_trace(max_rows=max_rows)

    def __repr__(self):
        return f"SelectionReportCollector(name={self.name!r}, stages={len(self._snapshots)})"


class BaseFeatureSelector(ParallelizableMixin, TransformerMixin, BaseEstimator, ABC):
    """特征筛选器基类.

    所有特征筛选器都继承此类，实现统一的 fit/transform 接口。
    支持中文筛选报告生成。支持可选的分箱器，在筛选前对数据进行分箱处理。

    **参数**

    :param target_rm: 默认False保留输入中的目标列；仅显式True时移除，不影响拟合时目标隔离
    :type target_rm: bool

    :param target: 目标变量列名，默认为 'target'。在 scorecardpipeline 风格中用于
        从 DataFrame 中提取目标列；在 sklearn 风格中若 fit 传入了 y 参数则优先使用 y

    :type target: str
    :param include: 默认值为 ``None`` 。强制保留的特征列表，这些特征无论如何都会被保留
    :type include: Optional[List[str]]
    :param exclude: 默认值为 ``None`` 。强制剔除的特征列表，这些特征无论如何都会被剔除
    :type exclude: Optional[List[str]]
    :param binner: 默认值为 ``None`` 。可选的已配置分箱器实例，支持已训练或待训练状态，不接受分箱器类
    :type binner: Optional[Any]
    :param bin_params: 默认值为 ``None`` 。可选的 ``OptimalBinning`` 构造参数字典。未传入 ``binner`` 时
        基类使用该字典创建分箱器；同时传入时 ``binner`` 优先

    :type bin_params: Optional[Dict[str, Any]]
    :param threshold: 默认值为 ``0.0`` 。筛选阈值，不同筛选器含义不同
    :type threshold: Union[float, int, str]
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Optional[Union[int, float]]
    :param force_drop: 默认值为 ``None`` 。强制剔除的特征列表，效果与 exclude 合并
    :type force_drop: Optional[List[str]]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]

    **属性**

    - selected_features_: 选中保留的特征列表

    - removed_features_: 被剔除的特征列表

    - dropped_: 被剔除的特征及原因 DataFrame，包含 '特征' 和 '剔除原因' 两列

    - scores_: 各特征的筛选得分的 Series

    - n_features_in_: fit 时输入的特征数量

    - forced_dropped_: 被强制剔除的特征列表

    - include_: 处理后的强制保留特征列表（字符串会被转为单元素列表）

    - exclude_: 处理后的强制剔除特征列表（包含 force_drop 的合并结果）

    **参考样例**

    **sklearn 风格（推荐用于纯特征矩阵）**

    ::

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1, include=['feat_0'], exclude=['feat_4'])
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> print(selector.selected_features_)
        ['feat_0', ...]

    **scorecardpipeline 风格（推荐用于完整数据框）**

    ::

        >>> from hscredit.core.selectors import IVSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> df = pd.DataFrame({
        ...     'feat1': np.random.randn(1000),
        ...     'feat2': [1] * 1000,  # 低方差，会被剔除
        ...     'target': np.random.randint(0, 2, 1000)
        ... })
        >>> selector = IVSelector(threshold=0.02, target='target')
        >>> selector.fit(df)
        IVSelector(...)
        >>> summary, details = selector.report()
        >>> print(summary[["输入特征数", "选中特征数"]])

    **引用**

    ``fit`` / ``transform`` / ``get_support`` 遵循估计器与特征筛选接口习惯，参见
    `scikit-learn 估计器开发指南 <https://scikit-learn.org/stable/developers/develop.html>`_。
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 目标列是否存在由本次transform输入决定，不能让sklearn按fit时的
        # 固定字段名再次重命名。容器转换由本类使用本次实际输出完成。
        cls._sklearn_auto_wrap_output_keys = set()

    def set_output(self, *, transform=None):
        """配置输出容器；按本次输入决定目标列，不改写拟合字段或报告快照。

        **参数**

        :param transform: 输出容器： ``None`` （默认）保持配置； ``"default"`` 使用原生输出，
            ``"pandas"`` 返回 DataFrame， ``"polars"`` 返回 Polars DataFrame（需要安装 polars）。
        """
        if transform is not None:
            if transform not in {"default", "pandas", "polars"}:
                raise ValidationError("transform 必须为 default、pandas 或 polars")
            self._sklearn_output_config = {"transform": transform}
        return self

    def _format_transform_output(self, result):
        mode = getattr(self, "_sklearn_output_config", {}).get(
            "transform", get_config().get("transform_output", "default")
        )
        if mode == "default":
            return result
        if not isinstance(result, pd.DataFrame):
            result = pd.DataFrame(result, columns=list(self.selected_features_))
        if mode == "pandas":
            return result
        if mode == "polars":
            try:
                import polars as pl
            except ImportError as exc:
                raise DependencyError("polars 输出需要安装可选依赖 polars") from exc
            return pl.from_pandas(result, include_index=False)
        raise ValidationError("输出容器只支持 default、pandas 或 polars")

    def __init__(
        self,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        binner: Optional[Any] = None,
        bin_params: Optional[Dict[str, Any]] = None,
        threshold: Union[float, int, str] = 0.0,
        n_jobs: Optional[Union[int, float]] = -1,
        force_drop: Optional[List[str]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        target_rm: bool = False,
    ):
        """初始化特征筛选器。

        **参数**

        :param target: 目标变量列名，默认为 'target'。当 fit 传入 y 参数时优先使用 y
        :param include: 强制保留的特征列表，这些特征无论如何都会被保留
        :param exclude: 强制剔除的特征列表，这些特征无论如何都会被剔除
        :param binner: 可选的已配置分箱器实例，支持已训练或未训练状态，不接受分箱器类
        :param bin_params: 可选的 ``OptimalBinning`` 构造参数字典； ``binner`` 优先
        :param threshold: 筛选阈值，不同筛选器含义不同
        :param n_jobs: 并行工作数，默认为 -1；None 沿用旧串行行为
        :param force_drop: 强制剔除的特征列表，效果与 exclude 合并
        :param parallel_backend: joblib 并行后端，默认为 None
        :param parallel_config: joblib 扩展配置，默认为 None
        :param target_rm: 默认False保留输入中的目标列；仅显式True时移除，不影响拟合时目标隔离
        """
        self.target = target
        self.include = include
        self.exclude = exclude
        self.binner = binner
        self.bin_params = bin_params
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.force_drop = force_drop
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config
        self.target_rm = target_rm

    def __setstate__(self, state):
        """补齐旧制品构造参数；未声明target_rm的对象默认保留目标列。"""
        super().__setstate__(state)
        for name, parameter in inspect.signature(type(self).__init__).parameters.items():
            if name not in self.__dict__ and parameter.default is not inspect.Parameter.empty:
                self.__dict__[name] = copy.deepcopy(parameter.default)

    def __sklearn_tags__(self):
        """声明筛选器支持缺失值，由具体算法按自身语义处理。"""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        """兼容 sklearn 1.5 及更早版本的缺失值标签接口。"""
        return {"allow_nan": True}

    def _clone_estimator_for_parallel(self, estimator: Any) -> Any:
        """克隆 estimator，并只向最浅并行层传递当前有效预算。"""
        model = clone(estimator)
        workers = _resolve_current_n_jobs(self.n_jobs) or 1
        return _set_estimator_parallel_budget(model, workers)

    @contextmanager
    def _estimator_parallel_context(self):
        """为使用 joblib 的底层 estimator 建立已验证后端上下文。"""
        config = validate_parallel_config(self.parallel_backend, self.parallel_config)
        backend_options = config.get("backend_kwargs", {}) or {}
        backend_options = dict(backend_options)
        inner_max_num_threads = config.get("inner_max_num_threads")
        backend = self.parallel_backend
        if inner_max_num_threads is not None:
            if backend == "threading":
                raise ValidationError("threading 后端不支持 inner_max_num_threads")
            backend = backend or "loky"
            backend_options["inner_max_num_threads"] = inner_max_num_threads

        _validate_parallel_backend(backend, backend_options)
        if backend is None:
            yield
            return

        with joblib_parallel_backend(backend, **backend_options):
            yield

    def _check_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, Optional[Union[pd.Series, np.ndarray]]]:
        """检查并处理输入数据，支持两种 API 风格。

        **API 风格**

        1. **sklearn 风格** : ``fit(X, y)`` — X 是特征矩阵，y 是目标变量

        2. **scorecardpipeline 风格** : ``fit(df)`` — df 是包含特征和目标列的完整 DataFrame，
           目标列名通过初始化参数 target 指定

        **优先级规则** : 如果 y 不为 None，使用 sklearn 风格；否则检查 X 中是否包含 target 列。

        **参数**

        :param X: 输入特征矩阵（DataFrame 或 numpy 数组）或包含目标列的完整数据框
        :param y: 目标变量，可选。如果不为 None 则优先使用

        :returns: 二元组 ``(处理后的特征 DataFrame, 目标变量或 None)``

        **参考样例**

        >>> from hscredit.core.selectors.base import BaseFeatureSelector
        >>> import pandas as pd
        >>> class DummySelector(BaseFeatureSelector):
        ...     def _fit_impl(self, X, y): pass
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X_df = pd.DataFrame(np.random.randn(5, 3), columns=['a', 'b', 'c'])
        >>> y_s = pd.Series([0, 1, 0, 1, 0])
        >>> sel = DummySelector()
        >>> X_out, y_out = sel._check_input(X_df, y_s)
        >>> X_out.shape
        (5, 3)
        """
        if not isinstance(self.target_rm, (bool, np.bool_)):
            raise ValidationError("target_rm 必须为布尔值")
        if issparse(X):
            raise ValidationError("Sparse input is not supported（不支持稀疏矩阵输入）")

        # 转换为DataFrame
        if not isinstance(X, pd.DataFrame):
            values = np.asarray(X)
            if values.ndim != 2:
                raise ValidationError(f"Expected 2D array, got {values.ndim}D array instead（特征矩阵必须为二维）")
            X = pd.DataFrame(values)

        if not X.columns.is_unique:
            raise ValidationError("输入字段名不能重复")
        if any(is_complex_dtype(dtype) for dtype in X.dtypes):
            raise ValidationError("Complex data not supported（不支持复数特征）")
        for column in X.columns:
            if is_numeric_dtype(X[column].dtype) and np.isinf(X[column].to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValidationError("Input X contains infinity（输入特征不能包含无穷值）")
        if X.shape[0] == 0:
            raise ValidationError("Found array with 0 sample(s)（输入数据不能为空）")
        if X.shape[1] == 0:
            raise ValidationError(
                f"0 feature(s) (shape=({len(X)}, 0)) while a minimum of 1 is required.（至少需要一个特征）"
            )

        prepared = prepare_xy(X, y, target=self.target)
        return prepared.X, prepared.y

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "BaseFeatureSelector":
        """拟合筛选器并记录中文筛选报告。

        使用隔离候选对象训练，成功后提交状态；失败时保留本次调用前的已拟合状态。

        **参数**

        :param X: DataFrame 或二维数组，形状为 ``(样本数, 特征数)`` 。
        :param y: 一维目标，默认 ``None`` 时按构造参数 ``target`` 从 DataFrame 提取；
            显式 ``y`` 优先。无监督筛选器可不提供目标。

        :return: 已拟合的筛选器自身。
        :raises ValidationError: 输入为空、字段或标签不满足筛选器要求时。

        **参考样例**

        >>> import pandas as pd
        >>> from hscredit.core.selectors import VarianceSelector
        >>> X = pd.DataFrame({"评分": [500, 550, 600], "常量": [1, 1, 1]})
        >>> selector = VarianceSelector(threshold=0.0, n_jobs=1).fit(X)
        >>> selector.transform(X).columns.tolist()
        ['评分']
        """
        started = perf_counter()
        candidate = clone(self)
        if hasattr(self, "_sklearn_output_config"):
            candidate._sklearn_output_config = copy.deepcopy(self._sklearn_output_config)
        self._prepare_fit_candidate(candidate)
        candidate._fit_once(X, y)
        candidate._finalize_fit()
        candidate.fit_id_ = uuid4().hex
        candidate.fit_duration_seconds_ = perf_counter() - started

        candidate.selection_report_ = capture_selection_report(candidate)
        self._adopt_fitted_candidate(candidate)
        return self

    def _prepare_fit_candidate(
        self,
        candidate: "BaseFeatureSelector",
        binner_memo: Optional[Dict[int, Any]] = None,
    ) -> None:
        """为隔离拟合准备候选对象。

        sklearn ``clone`` 会丢弃已拟合外部分箱器的学习状态，因此对显式传入且已拟合的
        分箱器使用受控深拷贝。未拟合分箱器仍使用 clone 生成的隔离副本。
        """
        if binner_memo is None:
            binner_memo = {}
        original_params = self.get_params(deep=False)
        candidate_params = candidate.get_params(deep=False)

        original_binner = original_params.get("binner")
        if original_binner is not None:
            memo_key = id(original_binner)
            if memo_key in binner_memo:
                candidate.binner = binner_memo[memo_key]
            else:
                candidate_binner = candidate_params.get("binner")
                if self._is_binner_fitted(original_binner):
                    candidate_binner = copy.deepcopy(original_binner)
                    candidate.binner = candidate_binner
                binner_memo[memo_key] = candidate_binner

        original_selectors = original_params.get("selectors")
        candidate_selectors = candidate_params.get("selectors")
        if original_selectors is None or candidate_selectors is None:
            return
        if len(original_selectors) != len(candidate_selectors):
            raise ValidationError("候选子筛选器数量与原配置不一致，无法准备拟合")

        for original_item, candidate_item in zip(original_selectors, candidate_selectors):
            original_child = self._selector_from_item(original_item)
            candidate_child = self._selector_from_item(candidate_item)
            if isinstance(original_child, BaseFeatureSelector) and isinstance(candidate_child, BaseFeatureSelector):
                original_child._prepare_fit_candidate(candidate_child, binner_memo)

    def _finalize_fit(self) -> None:
        """在候选对象上完成子类拟合后处理。"""

    def _adopt_fitted_candidate(self, candidate: "BaseFeatureSelector") -> None:
        """原子提交已成功拟合的候选状态。"""
        raw_plan = []
        self._build_candidate_commit_plan(candidate, raw_plan)
        plan = self._normalize_candidate_commit_plan(raw_plan)

        snapshots = {id(item["target"]): self._snapshot_target_state(item["target"]) for item in plan}
        try:
            for item in plan:
                target = item["target"]
                if item["is_selector"]:
                    target._apply_candidate_state(item["payload"])
                    self._rebind_public_params_direct(target, item["success_param_refs"])
                    if item["active_binner"]:
                        target._binner_instance = item["success_param_refs"]["binner"]
                    target._rebind_committed_parallel_children()
                else:
                    self._replace_object_state_direct(target, item["payload"])
        except Exception:
            # 回滚必须绕过可覆写、可注入故障的提交辅助方法。
            for item in plan:
                target = item["target"]
                target.__dict__.clear()
                target.__dict__.update(snapshots[id(target)])
                if item["is_selector"]:
                    self._restore_public_param_contents(item)
                    self._rebind_public_params_direct(target, item["rollback_param_refs"])
                    if item["rollback_active_binner"]:
                        target._binner_instance = item["rollback_param_refs"]["binner"]
            raise

    @classmethod
    def _snapshot_target_state(cls, target: Any) -> Dict[str, Any]:
        """建立回滚快照，但不递归复制独立提交的子筛选器和分箱器。

        Composite 的子筛选器和显式 binner 都是提交计划中的独立 target，
        各自已有快照。父级快照只需保留这些对象的引用，否则每一层都会
        再深拷贝整棵已拟合子树，在高维场景造成成倍内存放大。
        """
        memo = cls._nested_commit_reference_memo(target)
        return copy.deepcopy(target.__dict__, memo)

    @classmethod
    def _nested_commit_reference_memo(cls, target: Any) -> Dict[int, Any]:
        """收集应由事务计划独立管理、不可随父状态递归复制的对象。"""
        memo: Dict[int, Any] = {}
        seen_containers = set()

        def collect(value: Any) -> None:
            if isinstance(value, BaseFeatureSelector):
                memo[id(value)] = value
                return
            if isinstance(value, dict):
                if id(value) in seen_containers:
                    return
                seen_containers.add(id(value))
                for nested in value.values():
                    collect(nested)
                return
            if isinstance(value, (list, tuple, set)):
                if id(value) in seen_containers:
                    return
                seen_containers.add(id(value))
                for nested in value:
                    collect(nested)

        for value in getattr(target, "__dict__", {}).values():
            collect(value)

        if isinstance(target, BaseFeatureSelector):
            params = target.get_params(deep=False)
            explicit_binner = params.get("binner")
            if explicit_binner is not None:
                memo[id(explicit_binner)] = explicit_binner
            active_binner = getattr(target, "_binner_instance", None)
            if active_binner is not None:
                memo[id(active_binner)] = active_binner
        return memo

    @staticmethod
    def _rebind_public_params_direct(target: "BaseFeatureSelector", param_refs: Dict[str, Any]) -> None:
        """在可覆写 helper 返回后直接恢复公开构造参数身份。"""
        for name, value in param_refs.items():
            target.__dict__[name] = value

    @staticmethod
    def _restore_public_param_contents(item: Dict[str, Any]) -> None:
        """原地恢复回滚目标的公开可变参数内容。"""
        for name, original in item["rollback_param_refs"].items():
            snapshot = item["rollback_param_snapshots"][name]
            if isinstance(original, list):
                original.clear()
                if name == "selectors":
                    original.extend(item["rollback_param_shallow"][name])
                else:
                    original.extend(copy.deepcopy(snapshot))
            elif isinstance(original, dict):
                original.clear()
                original.update(copy.deepcopy(snapshot))
            elif isinstance(original, set):
                original.clear()
                original.update(copy.deepcopy(snapshot))

    @classmethod
    def _payloads_deep_equal(cls, left: Any, right: Any) -> bool:
        """安全比较两个候选 payload，避免 numpy/pandas 布尔歧义。"""
        if left is right:
            return True
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(cls._payloads_deep_equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return len(left) == len(right) and all(cls._payloads_deep_equal(a, b) for a, b in zip(left, right))
        if isinstance(left, np.ndarray):
            return bool(np.array_equal(left, right, equal_nan=True))
        if isinstance(left, (pd.DataFrame, pd.Series, pd.Index)):
            return bool(left.equals(right))
        try:
            result = left == right
            if isinstance(result, (np.ndarray, pd.Series, pd.DataFrame)):
                return bool(np.asarray(result).all())
            return bool(result)
        except Exception:
            return False

    @classmethod
    def _normalize_candidate_commit_plan(cls, raw_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对共享 target 去重；仅当候选来源相同或 payload 深等价时允许合并。"""
        normalized = []
        seen = {}
        for item in raw_plan:
            target_key = id(item["target"])
            previous = seen.get(target_key)
            if previous is None:
                seen[target_key] = item
                normalized.append(item)
                continue
            if previous["candidate_source"] is item["candidate_source"] or cls._payloads_deep_equal(
                previous["payload"], item["payload"]
            ):
                continue
            raise ValidationError("共享外部对象产生了冲突的候选状态，无法原子提交")
        return normalized

    def _build_candidate_commit_plan(
        self,
        candidate: "BaseFeatureSelector",
        plan: List[Dict[str, Any]],
        parallel_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """验证候选树并构建无副作用的完整提交计划。"""
        original_params = self.get_params(deep=False)
        candidate_params = candidate.get_params(deep=False)
        # candidate 已由 sklearn.clone 隔离并成功完成拟合，可直接转移其状态；
        # 再 deepcopy 会重复复制全部 scores、报告、分箱表和子筛选器树。
        candidate_state = dict(candidate.__dict__)

        original_binner = original_params.get("binner")
        candidate_binner = candidate_params.get("binner")
        if original_binner is not None and candidate_binner is not None:
            if not hasattr(original_binner, "__dict__") or not hasattr(candidate_binner, "__dict__"):
                raise ValidationError("外部对象必须支持状态复制，才能保证事务式拟合")
            plan.append(
                {
                    "target": original_binner,
                    "candidate_source": candidate_binner,
                    # candidate binner 同样已隔离，提交时直接转移状态即可。
                    "payload": dict(candidate_binner.__dict__),
                    "is_selector": False,
                }
            )

        original_selectors = original_params.get("selectors")
        candidate_selectors = candidate_params.get("selectors")
        if original_selectors is not None and candidate_selectors is not None:
            if len(original_selectors) != len(candidate_selectors):
                raise ValidationError("候选子筛选器数量与原配置不一致，无法提交拟合状态")
            for original_item, candidate_item in zip(original_selectors, candidate_selectors):
                original_child = self._selector_from_item(original_item)
                candidate_child = self._selector_from_item(candidate_item)
                if not (
                    isinstance(original_child, BaseFeatureSelector) and isinstance(candidate_child, BaseFeatureSelector)
                ):
                    raise ValidationError("候选子筛选器类型与原配置不一致，无法提交拟合状态")
                # 子级非默认配置优先；只有对应项未配置时才继承父级。
                child_parallel = self._resolve_child_parallel_config(original_child)
                original_child._build_candidate_commit_plan(candidate_child, plan, child_parallel)

        success_param_refs = dict(original_params)
        if parallel_overrides is not None:
            success_param_refs.update(parallel_overrides)

        stage_selectors = candidate_state.get("stage_selectors_")
        if isinstance(stage_selectors, dict):
            for child in stage_selectors.values():
                if isinstance(child, BaseFeatureSelector):
                    child.n_jobs = candidate_state.get("n_jobs")
                    child.parallel_backend = candidate_state.get("parallel_backend")
                    child.parallel_config = candidate_state.get("parallel_config")

        plan.append(
            {
                "target": self,
                "candidate_source": candidate,
                "payload": candidate_state,
                "is_selector": True,
                "success_param_refs": success_param_refs,
                "rollback_param_refs": dict(original_params),
                "rollback_param_snapshots": {
                    name: (
                        list(value)
                        if name == "selectors" and isinstance(value, list)
                        else copy.deepcopy(value) if isinstance(value, (list, dict, set)) else value
                    )
                    for name, value in original_params.items()
                },
                "rollback_param_shallow": {
                    name: list(value)
                    for name, value in original_params.items()
                    if name == "selectors" and isinstance(value, list)
                },
                "active_binner": (
                    candidate_params.get("binner") is not None
                    and getattr(candidate, "_binner_instance", None) is candidate_params.get("binner")
                ),
                "rollback_active_binner": (
                    original_params.get("binner") is not None
                    and getattr(self, "_binner_instance", None) is original_params.get("binner")
                ),
            }
        )

    def _apply_candidate_state(self, candidate_state: Dict[str, Any]) -> None:
        """应用一个已验证的候选状态载荷。"""
        # sklearn>=1.8 的 callback 上下文会在调用 estimator 前临时写入该属性，
        # 并在调用返回后负责删除。事务提交不能提前清掉这个框架所有的状态。
        parent_callback_ctx = self.__dict__.get("_parent_callback_ctx")
        has_parent_callback_ctx = "_parent_callback_ctx" in self.__dict__
        self.__dict__.clear()
        self.__dict__.update(candidate_state)
        if has_parent_callback_ctx:
            self.__dict__["_parent_callback_ctx"] = parent_callback_ctx

    @staticmethod
    def _replace_object_state_direct(target: Any, payload: Dict[str, Any]) -> None:
        """直接替换对象状态，仅用于已验证的提交计划。"""
        target.__dict__.clear()
        target.__dict__.update(payload)

    @staticmethod
    def _adopt_external_object_state(original: Any, candidate: Any) -> None:
        """把隔离候选对象的成功状态提交到原外部对象，同时保持对象身份。"""
        if not hasattr(original, "__dict__") or not hasattr(candidate, "__dict__"):
            raise ValidationError("外部对象必须支持状态复制，才能保证事务式拟合")
        original.__dict__.clear()
        original.__dict__.update(candidate.__dict__)

    @staticmethod
    def _selector_from_item(item: Any) -> Any:
        """从 Composite 支持的命名元组或直接元素中取得子筛选器。"""
        if isinstance(item, tuple) and len(item) == 2:
            return item[1]
        return item

    def _adopt_child_selector_states(self, original_items: Any, candidate_items: Any) -> None:
        """递归提交 Composite 候选子筛选器状态。"""
        if len(original_items) != len(candidate_items):
            raise ValidationError("候选子筛选器数量与原配置不一致，无法提交拟合状态")
        for original_item, candidate_item in zip(original_items, candidate_items):
            original = self._selector_from_item(original_item)
            fitted_candidate = self._selector_from_item(candidate_item)
            if isinstance(original, BaseFeatureSelector) and isinstance(fitted_candidate, BaseFeatureSelector):
                original._adopt_fitted_candidate(fitted_candidate)
                self._bind_parallel_config_to_child(original)

    def _bind_parallel_config_to_child(self, child: "BaseFeatureSelector") -> None:
        """按子级优先规则绑定父筛选器的并行配置。"""
        for name, value in self._resolve_child_parallel_config(child).items():
            setattr(child, name, value)

    def _resolve_child_parallel_config(self, child: "BaseFeatureSelector") -> Dict[str, Any]:
        """逐项解析子级有效并行配置，非默认子级值拥有最高优先级。"""
        return {
            "n_jobs": child.n_jobs if child.n_jobs != -1 else self.n_jobs,
            "parallel_backend": (
                child.parallel_backend if child.parallel_backend is not None else self.parallel_backend
            ),
            "parallel_config": (child.parallel_config if child.parallel_config is not None else self.parallel_config),
        }

    def _rebind_committed_parallel_children(self) -> None:
        """修复候选 clone 中自动生成阶段对父级配置副本的引用。

        ``stage_selectors_`` 是父筛选器内部创建的实现细节，不是用户传入的
        Composite 子级，因此其配置始终跟随父级；公开 ``selectors`` 列表
        仍由子级优先继承规则处理。
        """
        stage_selectors = getattr(self, "stage_selectors_", None)
        if isinstance(stage_selectors, dict):
            for child in stage_selectors.values():
                if isinstance(child, BaseFeatureSelector):
                    child.n_jobs = self.n_jobs
                    child.parallel_backend = self.parallel_backend
                    child.parallel_config = self.parallel_config

    def _clear_fitted_state(self) -> None:
        """清理上一次拟合产物，避免重复拟合沿用陈旧报告或得分。"""
        special_names = {
            "_feature_names",
            "_is_fitted",
            "_binner_instance",
            "_drop_reason",
            "dropped",
            "select_columns",
        }
        for name in list(self.__dict__):
            if name.endswith("_") or name in special_names:
                del self.__dict__[name]

    def _fit_once(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "BaseFeatureSelector":
        """拟合筛选器，学习特征重要性。

        支持两种 API 风格：sklearn 风格 ``fit(X, y)`` 和 scorecardpipeline 风格 ``fit(df)`` 。

        **参数**

        :param X: 输入特征矩阵（DataFrame 或 numpy 数组），或包含目标列的完整数据框
        :param y: 目标变量，可选。如果不为 None 则优先使用

        :returns: self

        :raises NotFittedError: 如果在子类实现中需要目标变量但 y 为 None

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> len(selector.selected_features_)
        ...
        """
        # 检查输入并分离特征和目标
        self.target_present_at_fit_ = (
            isinstance(X, pd.DataFrame) and self.target is not None and self.target in X.columns
        )
        self.input_columns_at_fit_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        X_processed, y_processed = self._check_input(X, y)
        if hasattr(self, "report_history"):
            from ._selection_history import initialize_history

            initialize_history(self)
        self.n_samples_in_ = len(X_processed)
        self.input_dtypes_ = X_processed.dtypes.astype(str).to_dict()
        self._selection_input_names_ = list(X_processed.columns)

        # 保存特征名
        self._get_feature_names(X_processed)
        self.n_features_in_ = X_processed.shape[1]

        # 处理include参数（强制保留的特征）
        if self.include is None:
            self.include_ = []
        elif isinstance(self.include, str):
            self.include_ = [self.include]
        elif isinstance(self.include, (list, tuple, np.ndarray)):
            self.include_ = list(self.include)
        else:
            self.include_ = []

        # 处理exclude参数（强制剔除的特征）
        if self.exclude is None:
            self.exclude_ = []
        elif isinstance(self.exclude, str):
            self.exclude_ = [self.exclude]
        elif isinstance(self.exclude, (list, tuple, np.ndarray)):
            self.exclude_ = list(self.exclude)
        else:
            self.exclude_ = []

        # 处理force_drop参数（合并到exclude_中）
        if self.force_drop is not None:
            if isinstance(self.force_drop, str):
                force_drop_list = [self.force_drop]
            elif isinstance(self.force_drop, (list, tuple, np.ndarray)):
                force_drop_list = list(self.force_drop)
            else:
                force_drop_list = []
            # 合并到exclude_（去重并保持用户传入顺序）
            self.exclude_ = list(dict.fromkeys(self.exclude_ + force_drop_list))

        # 强制剔除字段不参与任何计算。普通筛选器的强制保留字段也直接
        # 跳过计算；Corr/VIF/Stepwise 等比较型筛选器可通过钩子声明
        # include 必须作为基准变量继续参与后续计算。
        original_X = X_processed
        selection_columns = self._get_selection_input_columns(original_X)
        self.selection_input_features_ = list(selection_columns)
        if selection_columns == list(original_X.columns):
            # 最常见路径不创建全量列副本；高维数据上一次无意义的 DataFrame
            # 复制就可能额外占用数 GB 内存。
            X_processed = original_X
        else:
            X_processed = original_X.loc[:, selection_columns]

        # 如果配置了分箱器或分箱参数，只对真正需要筛选的字段进行分箱
        self._binner_instance = None
        if X_processed.shape[1] > 0:
            if self._should_apply_binner(y_processed):
                X_processed = self._apply_binner(X_processed, y_processed)

            # 执行子类实现的具体fit逻辑
            self._fit_impl(X_processed, y_processed)
        else:
            self._initialize_empty_selection_result()

        # 创建初始dropped_（只在子类没有设置dropped_时）
        if hasattr(self, "selected_features_") and self.selected_features_ is not None:
            # 如果子类已经设置了详细的dropped_，则保留
            if not hasattr(self, "dropped_") or self.dropped_ is None or len(self.dropped_) == 0:
                dropped_cols = [c for c in X_processed.columns if c not in self.selected_features_]
                if len(dropped_cols) > 0:
                    reason = getattr(self, "_drop_reason", "不满足筛选条件")
                    self.dropped_ = pd.DataFrame({"特征": dropped_cols, "剔除原因": [reason] * len(dropped_cols)})
                    self.removed_features_ = dropped_cols
                else:
                    self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因"])
                    self.removed_features_ = []

        # 确保include的特征被保留
        self._apply_include(original_X)

        # 应用exclude（强制剔除）
        self._apply_exclude(original_X)

        # 所有公开筛选结果都按拟合输入字段顺序输出，include 只改变保留状态，
        # 不应把字段追加到末尾。
        selected_set = set(self.selected_features_)
        self.selected_features_ = [column for column in original_X.columns if column in selected_set]

        # 子类通常会基于实际计算子集刷新特征名；对外仍应暴露本轮原始
        # 输入的 sklearn 元数据，避免 include/exclude 改变 n_features_in_。
        self._get_feature_names(original_X)
        self.n_features_in_ = original_X.shape[1]

        self._is_fitted = True
        return self

    def _included_features_participate_in_selection(self) -> bool:
        """返回强制保留字段是否仍需参与筛选计算。

        单变量阈值类筛选器直接保留 include 字段，不再为其分箱或计算指标。
        需要把 include 当作比较基准的多变量筛选器应覆盖本方法并返回 True。
        """
        return False

    def _get_selection_input_columns(self, X: pd.DataFrame) -> List[str]:
        """按强制字段规则返回真正需要分箱和筛选的输入列。"""
        excluded = set(self.exclude_)
        included = set(self.include_)
        include_participates = self._included_features_participate_in_selection()
        return [
            column
            for column in X.columns
            if column not in excluded and (include_participates or column not in included)
        ]

    def _initialize_empty_selection_result(self) -> None:
        """当所有输入列都已被强制处理时建立最小、完整的拟合结果。"""
        self.selected_features_ = []
        self.scores_ = pd.Series(dtype=float)
        self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因"])
        self.removed_features_ = []

    def _apply_binner(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """应用分箱器对数据进行分箱。

        支持传入已训练或待训练的分箱器实例。如果传入未训练的分箱器，
        将自动在输入数据上进行拟合。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量，可选

        :returns: 分箱后的 DataFrame，列名与输入保持一致

        **参考样例**

        >>> from hscredit.core.selectors.base import BaseFeatureSelector
        >>> from hscredit.core.binning import OptimalBinning
        >>> import pandas as pd
        >>> class DummySelector(BaseFeatureSelector):
        ...     def _fit_impl(self, X, y): pass
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> sel = DummySelector(binner=OptimalBinning(method='uniform', max_n_bins=2))
        >>> X_binned = sel._apply_binner(X, y)
        >>> X_binned.shape
        (100, 3)
        """
        self._binner_instance = self._resolve_binner()
        if self._binner_instance is None:
            return X

        if not self._is_binner_fitted(self._binner_instance):
            fit_method = getattr(self._binner_instance, "fit", None)
            if not callable(fit_method):
                raise ValidationError("未训练的分箱器必须提供 fit 方法")
            if y is not None:
                fit_method(X, y)
            else:
                fit_method(X)

        return self._transform_with_fitted_binner(X)

    def _transform_with_fitted_binner(self, X: pd.DataFrame) -> pd.DataFrame:
        """使用本轮已经拟合的分箱器转换外部同构数据。"""
        if self._binner_instance is None:
            return X

        transform_method = getattr(self._binner_instance, "transform", None)
        apply_method = getattr(self._binner_instance, "apply", None)
        if callable(transform_method):
            try:
                parameters = inspect.signature(transform_method).parameters.values()
                supports_metric = any(
                    parameter.name == "metric" or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                supports_metric = False
            if supports_metric:
                X_binned = transform_method(X, metric="indices")
            else:
                X_binned = transform_method(X)
        elif callable(apply_method):
            X_binned = apply_method(X)
        else:
            raise ValidationError("分箱器必须提供 transform 或 apply 方法")

        return self._normalize_binned_output(X_binned, X)

    def _resolve_binner(self) -> Optional[Any]:
        """按 ``binner > bin_params`` 解析有效分箱器。"""
        if self.binner is not None:
            if isinstance(self.binner, type):
                raise ValidationError("binner 必须传入配置好的分箱器实例，不能传入分箱器类")
            return self.binner

        if self.bin_params is None:
            return None
        if not isinstance(self.bin_params, dict):
            raise ValidationError("bin_params 分箱参数必须是字典")

        from ..binning import OptimalBinning

        bin_params = dict(self.bin_params)
        # 内部分箱器属于筛选器调用树的一部分。只有子级未显式配置时才
        # 继承父级预算；用户在 bin_params 中给出的更具体配置优先。
        bin_params.setdefault("n_jobs", self.n_jobs)
        bin_params.setdefault("parallel_backend", self.parallel_backend)
        bin_params.setdefault(
            "parallel_config",
            dict(self.parallel_config) if self.parallel_config is not None else None,
        )
        return OptimalBinning(**bin_params)

    def _is_binner_fitted(self, binner: Any) -> bool:
        """兼容 HSCredit 与常见 sklearn 风格的分箱器拟合状态。"""
        for name in ("_is_fitted", "is_fitted_", "fitted_"):
            if hasattr(binner, name):
                return bool(getattr(binner, name))

        if not callable(getattr(binner, "fit", None)):
            return callable(getattr(binner, "transform", None)) or callable(getattr(binner, "apply", None))

        try:
            check_is_fitted(binner)
        except (SklearnNotFittedError, TypeError):
            return False
        return True

    def _should_apply_binner(
        self,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> bool:
        """判断当前拟合是否需要执行前置分箱。"""
        return self.binner is not None or self.bin_params is not None

    @staticmethod
    def _normalize_binned_output(X_binned: Any, X: pd.DataFrame) -> pd.DataFrame:
        """将分箱器输出规范为与输入行列对齐的 DataFrame。"""
        if isinstance(X_binned, pd.DataFrame):
            if X_binned.shape != X.shape:
                raise ValidationError(f"分箱结果形状 {X_binned.shape} 与输入形状 {X.shape} 不一致")
            if set(X_binned.columns) != set(X.columns):
                raise ValidationError("分箱结果字段与输入字段不一致")
            result = X_binned.loc[:, X.columns].copy()
            result.index = X.index
            return result

        values = np.asarray(X_binned)
        if values.ndim == 1 and X.shape[1] == 1:
            values = values.reshape(-1, 1)
        if values.shape != X.shape:
            raise ValidationError(f"分箱结果形状 {values.shape} 与输入形状 {X.shape} 不一致")
        return pd.DataFrame(values, columns=X.columns, index=X.index)

    def _apply_include(self, X: pd.DataFrame) -> None:
        """确保 include 的特征被保留。

        将初始化时传入的 include 参数中的特征添加到 selected_features_ 列表中。

        **参数**

        :param X: 输入特征 DataFrame
        """
        if hasattr(self, "selected_features_") and self.selected_features_ is not None:
            # 添加include的特征
            added = False
            for col in self.include_:
                if col in X.columns and col not in self.selected_features_:
                    self.selected_features_.append(col)
                    added = True

            if added:
                dropped_cols = [c for c in X.columns if c not in self.selected_features_]
                if hasattr(self, "dropped_") and self.dropped_ is not None:
                    if len(self.dropped_) > 0 and "特征" in self.dropped_.columns:
                        self.dropped_ = self.dropped_.loc[~self.dropped_["特征"].isin(self.selected_features_)].copy()
                    if len(self.dropped_) == 0 and len(dropped_cols) == 0:
                        self.dropped_ = pd.DataFrame(columns=self.dropped_.columns)
                    self.removed_features_ = (
                        self.dropped_["特征"].tolist()
                        if len(self.dropped_) > 0 and "特征" in self.dropped_.columns
                        else []
                    )
                else:
                    if len(dropped_cols) > 0:
                        reason = getattr(self, "_drop_reason", "不满足筛选条件")
                        self.dropped_ = pd.DataFrame({"特征": dropped_cols, "剔除原因": [reason] * len(dropped_cols)})
                        self.removed_features_ = dropped_cols
                    else:
                        self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因"])
                        self.removed_features_ = []

    @abstractmethod
    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """子类实现的 fit 逻辑。

        子类必须重写此方法，在其中实现具体的特征筛选计算逻辑，
        并设置 self.selected_features_ 和 self.scores_ 属性。

        **参数**

        :param X: 输入特征 DataFrame（已通过 _check_input 处理）
        :param y: 目标变量，可选
        """
        pass

    def _apply_exclude(self, X: pd.DataFrame) -> None:
        """强制剔除指定的特征。

        将初始化时传入的 exclude 和 force_drop 参数中的特征从 selected_features_ 中移除，
        并记录到 forced_dropped_ 和 dropped_ 属性中。

        **参数**

        :param X: 输入特征 DataFrame
        """
        if not hasattr(self, "selected_features_") or self.selected_features_ is None:
            return

        # 记录被强制剔除的特征
        self.forced_dropped_ = []

        # 移除exclude的特征（遍历用户传入的exclude_列表）
        for col in self.exclude_:
            if col in self.selected_features_:
                self.selected_features_.remove(col)
                self.forced_dropped_.append(col)
            elif col in X.columns:
                # 特征原本在X中但不在selected_features_中（已被筛选掉）
                # 仍然记录为强制剔除
                if col not in self.forced_dropped_:
                    self.forced_dropped_.append(col)

        # 更新dropped_报告,添加强制剔除的原因
        if hasattr(self, "dropped_") and self.dropped_ is not None and len(self.dropped_) > 0:
            # 添加强制剔除的特征到dropped_
            for col in self.forced_dropped_:
                # 检查是否已经在dropped_中
                if col not in self.dropped_["特征"].values:
                    new_row = pd.DataFrame({"特征": [col], "剔除原因": ["强制剔除"]})
                    self.dropped_ = pd.concat([self.dropped_, new_row], ignore_index=True)
                else:
                    # 在原原因后附加"[强制剔除]"标记
                    current_reason = self.dropped_.loc[self.dropped_["特征"] == col, "剔除原因"].iloc[0]
                    if "[强制剔除]" not in str(current_reason):
                        self.dropped_.loc[self.dropped_["特征"] == col, "剔除原因"] = f"{current_reason} [强制剔除]"
        elif len(self.forced_dropped_) > 0:
            # 创建新的dropped_记录
            self.dropped_ = pd.DataFrame(
                {"特征": self.forced_dropped_, "剔除原因": ["强制剔除"] * len(self.forced_dropped_)}
            )

        # 更新 removed_features_
        if hasattr(self, "dropped_") and len(self.dropped_) > 0:
            self.removed_features_ = self.dropped_["特征"].tolist()
        else:
            self.removed_features_ = []

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray, List[str]],
    ) -> Union[pd.DataFrame, np.ndarray, List[str]]:
        """根据筛选结果转换数据。

        根据 fit 阶段选中的特征，对输入数据进行筛选。

        **参数**

        :param X: 输入数据，支持 DataFrame、numpy 数组或特征名列表
        :type X: Union[pd.DataFrame, np.ndarray, List[str]]

        :returns: 筛选后的数据，类型与输入一致

            - DataFrame输入：默认选中特征加输入中已有的target列；target_rm=True时只输出特征

            - ndarray输入：返回选中列的ndarray

            - 列表输入：返回筛选后的特征名列表

        :raises NotFittedError: 当筛选器尚未拟合时

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> X_selected = selector.transform(X)
        >>> X_selected.shape[1] < X.shape[1]
        True
        """
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")

        # 如果传入的是列表，返回筛选后的特征列表
        if not isinstance(self.target_rm, (bool, np.bool_)):
            raise ValidationError("target_rm 必须为布尔值")
        if isinstance(X, list) and (len(X) == 0 or all(isinstance(item, str) for item in X)):
            selected = [c for c in X if c in self.selected_features_]
            if not self.target_rm and self.target in X and self.target not in selected:
                selected.append(self.target)
            return selected

        if isinstance(X, pd.DataFrame):
            missing = [column for column in self.feature_names_in_ if column not in X.columns]
            if missing:
                raise ValidationError(f"转换数据缺少拟合字段: {missing}")
            if not X.columns.is_unique:
                raise ValidationError("转换数据字段名不能重复")

            selected = list(self.selected_features_)

            # scorecardpipeline 风格: 如果 X 中包含 target 列，透传到输出
            target_col = getattr(self, "target", None)
            if not self.target_rm and target_col is not None and target_col in X.columns and target_col not in selected:
                return self._format_transform_output(X.loc[:, selected + [target_col]])

            return self._format_transform_output(X.loc[:, selected])

        values = np.asarray(X)
        if values.ndim != 2:
            raise ValidationError("Expected 2D array. Reshape your data（ndarray 输入必须是二维特征矩阵）")
        if values.shape[1] != self.n_features_in_:
            raise ValidationError(
                f"X has {values.shape[1]} features, but {self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input（转换数据特征数与拟合时不一致）"
            )
        return self._format_transform_output(values[:, self.get_support()])

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """拟合并转换数据。

        等价于依次调用 fit(X, y) 和 transform(X)。

        **参数**

        :param X: 输入特征矩阵（DataFrame 或 numpy 数组），或包含目标列的完整数据框
        :type X: Union[pd.DataFrame, np.ndarray]
        :param y: 默认值为 ``None`` 。目标变量，可选
        :type y: Optional[Union[pd.Series, np.ndarray]]

        :returns: 筛选后的特征数据
        """
        return self.fit(X, y).transform(X)

    def get_support_mask(self) -> np.ndarray:
        """获取特征选择掩码。

        返回一个布尔数组，长度为输入特征数，True 表示该特征被选中。

        :returns: 布尔 numpy 数组，True 对应选中特征的下标

        :raises NotFittedError: 当筛选器尚未拟合时

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> mask = selector.get_support_mask()
        >>> mask.sum() == len(selector.selected_features_)
        True
        """
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")

        selected = set(self.selected_features_)
        return np.fromiter(
            (column in selected for column in self._feature_names), dtype=bool, count=self.n_features_in_
        )

    def _get_support_mask(self) -> np.ndarray:
        """实现 sklearn 特征选择器的标准支持掩码接口。"""
        return self.get_support_mask()

    def get_support(self, indices: bool = False) -> np.ndarray:
        """返回选择掩码； ``indices=True`` 时返回选中列位置。

        **参数**

        :param indices: 默认 ``False`` 返回与拟合特征顺序一致的布尔掩码； ``True`` 返回选中特征的整数位置。
        """
        mask = self.get_support_mask()
        return np.flatnonzero(mask) if indices else mask

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """返回转换后的字段名，兼容 sklearn Pipeline。

        **参数**

        :param input_features: 可选输入特征名，默认 ``None`` 使用拟合时记录的名称；传入时须与拟合特征一致。
        """
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")
        if not isinstance(self.target_rm, (bool, np.bool_)):
            raise ValidationError("target_rm 必须为布尔值")
        include_target = bool(getattr(self, "target_present_at_fit_", False))
        if input_features is not None:
            try:
                provided = list(input_features)
                include_target = self.target in provided
                equal = pd.Index([field for field in provided if field != self.target], tupleize_cols=False).equals(
                    pd.Index(self._feature_names, tupleize_cols=False)
                )
            except (TypeError, ValueError):
                equal = False
            if not equal:
                raise ValidationError("input_features 与拟合字段不一致")
        output = list(self.selected_features_)
        if not self.target_rm and include_target and self.target not in output:
            output.append(self.target)
        return np.fromiter(iter(output), dtype=object, count=len(output))

    def report(self, *, strict: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """返回当前筛选器的汇总与具体明细，所有筛选器使用相同中文表头。

        只读取 fit 保存的真实决策，不重新 fit/transform，不根据当前参数重算。
        组合筛选器和评分卡筛选器同时包含根决策与实际子步骤；根节点和子节点
        的计数不能相加作为唯一特征数。不同算法保留各自的指标和实际判定条件。

        :param strict: 默认 True，拒绝未拟合或不完整来源；False 保留可用结果，
            完整性及诊断保存在两张表各自的 attrs 中。
        :return: ``(summary, details)`` 两个独立 DataFrame。summary 包含阶段路径、
            输入/选中/剔除特征数及保留率；details 通过“记录类型”区分特征决策、
            指标明细和迭代事件，包含原始指标、阈值、比较符、筛选结果及原因。
            所有筛选器列顺序和 dtype 一致；无记录的字段留空，不补造历史、不写文件。
        :raises NotFittedError: 严格模式下尚未拟合。
        :raises ValidationError: 参数无效或严格模式下来源不完整。

        **参考样例**

        >>> selector = VarianceSelector(threshold=0.1).fit(data)
        >>> summary, details = selector.report()
        >>> decisions = details.loc[details['记录类型'] == '特征决策']
        """
        if not isinstance(strict, bool):
            raise ValidationError("strict 必须为布尔值")

        return selection_report(self, strict=strict)

    def get_selection_result(self):
        """返回与模型状态隔离的完整报告；读取报告不重新拟合或计算指标。"""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")

        return _selector_report(self, strict=True)

    def get_selection_details(self) -> pd.DataFrame:
        """统一逐特征决策表，含强制操作和未计算指标，所有筛选器列结构一致。"""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")
        return self.get_selection_result().details

    def get_scores_df(self) -> pd.DataFrame:
        """获取特征得分的 DataFrame。

        **返回 DataFrame 列说明**

        - **特征**  (`str`): 特征名称

        - **得分**  (`float`): 特征在筛选器中的得分

        - **状态**  (`str`): 特征状态，取值为 '选中' 或 '剔除'

        按拟合输入顺序排列；不同算法得分方向不同，完整语义见get_selection_details。

        :returns: 包含特征得分的 DataFrame，无得分时返回仅含列名的空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> scores_df = selector.get_scores_df()
        >>> print(scores_df.head())
        """
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")
        snapshot = self.get_selection_result().metadata["阶段快照"]["selector"]
        scores = snapshot["原始得分"]
        selected = set(snapshot["输出字段"])

        records = []
        for feat in snapshot["输入字段"]:
            score = scores.get(feat, np.nan)
            # 转换numpy类型
            if isinstance(score, (np.integer, np.floating)):
                score = float(score)

            status = "选中" if feat in selected else "剔除"
            records.append({"特征": feat, "得分": score, "状态": status})

        df = pd.DataFrame(records, columns=["特征", "得分", "状态"])
        return df

    def get_dropped_df(self) -> pd.DataFrame:
        """获取被剔除特征的 DataFrame。

        **返回 DataFrame 列说明**

        - **特征**  (`str`): 被剔除的特征名称

        - **剔除原因**  (`str`): 特征被剔除的原因描述

        :returns: 包含被剔除特征及原因的 DataFrame，无数据时返回仅含列名的空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        VarianceSelector(...)
        >>> dropped_df = selector.get_dropped_df()
        >>> print(f"共剔除 {len(dropped_df)} 个特征")
        """
        if not getattr(self, "_is_fitted", False):
            return pd.DataFrame(columns=["特征", "剔除原因"])
        snapshot = self.get_selection_result().metadata["阶段快照"]["selector"]
        return pd.DataFrame(snapshot["剔除明细"], columns=snapshot["剔除明细列"])

    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """获取特征名称列表。

        **参数**

        :param X: 输入特征 DataFrame

        :returns: 特征名称列表，同时设置 sklearn 兼容的 feature_names_in_ 属性
        """
        if hasattr(X, "columns"):
            self._feature_names = X.columns.tolist()
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        # sklearn 兼容：设置 feature_names_in_ 属性
        self.feature_names_in_ = np.fromiter(iter(self._feature_names), dtype=object, count=len(self._feature_names))
        return self._feature_names


class CompositeFeatureSelector(BaseFeatureSelector):
    """组合特征筛选器.

    将多个筛选器组合在一起，按顺序执行筛选。后续筛选器基于前面筛选器的结果进行筛选。
    支持通过 include 和 exclude 参数强制保留或剔除特定特征。

    **参数**

    :param bin_params: 默认值为 ``None`` 。可选的 OptimalBinning 构造参数
    :type bin_params: Optional[Dict[str, Any]]
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Optional[Union[int, float]]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]
    :param target_rm: 默认False保留输入中的目标列；仅显式True时移除，不影响拟合时目标隔离
    :type target_rm: bool

    :param selectors: 筛选器列表，按执行顺序排列
    :type selectors: List[BaseFeatureSelector]
    :param strategy: 默认值为 ``'sequential'`` 。组合策略，可选 'sequential' 或 'intersection'

        - 'sequential': 按顺序筛选，每轮剔除不满足条件的特征

        - 'intersection': 取所有筛选器选中特征的交集

    :type strategy: str
    :param include: 默认值为 ``None`` 。强制保留的特征列表，这些特征无论如何都会被保留
    :type include: Optional[List[str]]
    :param exclude: 默认值为 ``None`` 。强制剔除的特征列表，这些特征无论如何都会被剔除
    :type exclude: Optional[List[str]]
    :param target: 目标变量列名，默认为 'target'
    :type target: str
    :param binner: 默认值为 ``None`` 。可选的分箱器
    :type binner: Optional[Any]

    **参考样例**

    ::

        >>> from hscredit.core.selectors import (
        ...     VarianceSelector, CorrSelector, IVSelector
        ... )
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = pd.Series(np.random.randint(0, 2, 200))
        >>> composite = CompositeFeatureSelector([
        ...     VarianceSelector(threshold=0.01),
        ...     CorrSelector(threshold=0.8),
        ...     IVSelector(threshold=0.02),
        ... ])
        >>> composite.fit(X, y)
        CompositeFeatureSelector(...)
        >>> len(composite.selected_features_)
        ...
    """

    def __init__(
        self,
        selectors: List[BaseFeatureSelector],
        strategy: str = "sequential",
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        binner: Optional[Any] = None,
        bin_params: Optional[Dict[str, Any]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        target_rm: bool = False,
    ):
        """初始化组合特征筛选器。

        **参数**

        :param selectors: 筛选器列表，按执行顺序排列
        :param strategy: 组合策略，默认为 'sequential'
        :param target: 目标变量列名，默认为 'target'
        :param include: 强制保留的特征列表
        :param exclude: 强制剔除的特征列表
        :param binner: 可选的分箱器
        :param bin_params: 可选的 OptimalBinning 构造参数
        """
        super().__init__(
            target=target,
            include=include,
            exclude=exclude,
            binner=binner,
            bin_params=bin_params,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            target_rm=target_rm,
        )
        self.selectors = selectors
        self.strategy = strategy

    def _check_input(self, X, y=None):
        if self.strategy not in {"sequential", "intersection"}:
            raise ValidationError("strategy 必须为 sequential 或 intersection")
        if not isinstance(self.selectors, (list, tuple)) or not self.selectors:
            raise ValidationError("selectors 必须是非空筛选器列表")
        for item in self.selectors:
            if isinstance(item, tuple):
                if len(item) != 2 or not isinstance(item[0], str) or not item[0]:
                    raise ValidationError("具名筛选器必须为 (非空名称, 筛选器)")
                child = item[1]
            else:
                child = item
            if not isinstance(child, BaseFeatureSelector):
                raise ValidationError("组合子步骤必须为BaseFeatureSelector实例")
        self.executed_stages_ = []
        self.skipped_stages_ = {}
        return super()._check_input(X, y)

    def _initialize_empty_selection_result(self):
        super()._initialize_empty_selection_result()
        self.executed_stages_ = []
        self.skipped_stages_ = {i: "父级强制操作后无待筛选特征" for i in range(len(self.selectors))}

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """执行组合筛选。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量
        """
        if self.strategy == "sequential":
            self._fit_sequential(X, y)
        else:
            self._fit_intersection(X, y)

    def _fit_sequential(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """顺序筛选策略。

        按列表顺序依次执行每个筛选器，后续筛选器仅在上一轮选中的特征上进行筛选。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量
        """
        # 子筛选器只读输入，无需在第一阶段复制整个高维数据集；后续只在
        # 特征集合真正收缩时创建列子集。
        current_X = X
        all_dropped = []

        for i, item in enumerate(self.selectors):
            # 处理 ('name', selector) 元组格式或直接的 selector 对象
            if isinstance(item, tuple) and len(item) == 2:
                selector_name = item[0]
                selector = item[1]
            else:
                selector_name = item.__class__.__name__
                selector = item

            # 使用当前特征进行筛选；阶段串行执行，每阶段获得完整预算。
            self._configure_child_selector(selector)
            selector.fit(current_X, y)
            self.executed_stages_.append(i)

            # 获取选中特征
            selected = selector.selected_features_

            # 记录被剔除的特征（穿透收集详细指标）
            if hasattr(selector, "dropped_") and len(selector.dropped_) > 0:
                dropped = selector.dropped_.copy(deep=False)
                dropped["筛选轮次"] = i + 1
                dropped["筛选器"] = selector_name
                dropped["筛选器类型"] = selector.__class__.__name__
                all_dropped.append(dropped)

            # 更新当前特征
            if len(selected) > 0:
                if selected != list(current_X.columns):
                    current_X = current_X.loc[:, selected]
            else:
                current_X = current_X.iloc[:, 0:0]
                self.skipped_stages_.update({j: "上游无剩余特征" for j in range(i + 1, len(self.selectors))})
                break

        # 最终选中的特征
        self.selected_features_ = current_X.columns.tolist()
        self.scores_ = None
        self.forced_dropped_ = []  # 初始化forced_dropped_

        # 合并所有剔除记录
        if len(all_dropped) > 0:
            self.dropped_ = pd.concat(all_dropped, ignore_index=True)
            self.removed_features_ = self.dropped_["特征"].tolist()
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "筛选轮次", "筛选器", "筛选器类型"])
            self.removed_features_ = []

    def _fit_intersection(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
        """交集筛选策略。

        所有筛选器独立对全量特征进行筛选，最终取各筛选器选中特征的交集。

        **参数**

        :param X: 输入特征 DataFrame
        :param y: 目标变量
        """
        tasks = []
        for position, item in enumerate(self.selectors):
            named = isinstance(item, tuple) and len(item) == 2
            if named:
                selector_name, selector = item
            else:
                selector_name, selector = item.__class__.__name__, item
            self._configure_child_selector(selector)
            tasks.append((position, selector_name, selector, X, y, named))

        results = self._parallel_execute(
            _fit_intersection_selector,
            tasks,
            default_backend="threading",
            task_labels=[task[1] for task in tasks],
            has_parallel_children=True,
            workload=ParallelWorkload(
                task_count=len(tasks),
                rows=len(X),
                columns=max(1, X.shape[1]),
                data_bytes=int(X.memory_usage(deep=True).sum()),
                cost_per_item=10.0,
                capability="thread_safe",
                has_parallel_children=True,
                operation="组合筛选器交集拟合",
            ),
        )

        selected_sets = []
        all_dropped = []
        fitted_selectors = list(self.selectors)
        for position, selector_name, selector, named, selected, dropped in results:
            self.executed_stages_.append(position)
            fitted_selectors[position] = (selector_name, selector) if named else selector
            selected_sets.append(selected)
            if dropped is not None:
                all_dropped.append(dropped)
        self.selectors = fitted_selectors

        # 取交集，并严格保持原始输入列顺序。
        intersection = set.intersection(*selected_sets) if selected_sets else set()
        self.selected_features_ = [column for column in X.columns if column in intersection]
        self.scores_ = None
        self.forced_dropped_ = []  # 初始化forced_dropped_

        # 构建详细的剔除原因
        removed_features = [c for c in X.columns if c not in self.selected_features_]
        if len(removed_features) > 0:
            # 对于每个被剔除的特征，记录其被哪些筛选器剔除
            drop_reasons = []
            for feature in removed_features:
                rejected_by = []
                for item in self.selectors:
                    if isinstance(item, tuple) and len(item) == 2:
                        sel_name = item[0]
                        sel = item[1]
                    else:
                        sel_name = item.__class__.__name__
                        sel = item
                    if hasattr(sel, "dropped_") and len(sel.dropped_) > 0:
                        if feature in sel.dropped_["特征"].values:
                            rejected_by.append(sel_name)
                if rejected_by:
                    drop_reasons.append(f"被以下筛选器剔除: {', '.join(rejected_by)}")
                else:
                    drop_reasons.append("未被所有筛选器同时选中")

            self.dropped_ = pd.DataFrame({"特征": removed_features, "剔除原因": drop_reasons})
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因"])

        self.removed_features_ = removed_features

        # 保存详细的筛选器结果（用于穿透查询）
        if len(all_dropped) > 0:
            self.detailed_dropped_ = pd.concat(all_dropped, ignore_index=True)
        else:
            self.detailed_dropped_ = pd.DataFrame()

    def _configure_child_selector(self, selector: BaseFeatureSelector) -> None:
        """将组合器的并行配置传递给顺序执行的子筛选器。

        每项配置独立解析：子筛选器的非默认值优先，未配置项才继承父级。
        """
        self._bind_parallel_config_to_child(selector)
