"""特征筛选器基类.

定义特征筛选器的统一接口和通用方法。
所有筛选器都继承此类,确保API的一致性。

报告系统设计:
1. 单个筛选器报告: 每个筛选器实现 get_selection_report(),返回标准化报告
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
>>> report = selector.get_selection_report()
>>> print(f"选中特征数: {report['选中特征数']}")
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Union, List, Dict, Optional, Any, Tuple
from datetime import datetime
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
    """特征筛选报告收集器.

    聚合多个已拟合筛选器的结果并生成汇总报告。
    当前通过 ``add_report`` 手动添加，不会自动挂接 sklearn Pipeline。

    **使用方式**

    ::

        >>> from hscredit.core.selection import (
        ...     SelectionReportCollector,
        ...     VarianceSelector,
        ...     CorrSelector
        ... )
        >>> collector = SelectionReportCollector()
        >>>
        >>> # 方式1: 手动添加筛选器
        >>> selector1 = VarianceSelector(threshold=0.1)
        >>> selector1.fit(X)
        >>> collector.add_report(selector1)
        >>>
        >>> selector2 = CorrSelector(threshold=0.8)
        >>> selector2.fit(X, y)
        >>> collector.add_report(selector2)
        >>>
        >>> # 获取汇总报告
        >>> summary = collector.get_summary()
        >>> print(summary)
        >>>
        >>> # 导出为DataFrame
        >>> df = collector.to_dataframe()
    """

    def __init__(self, name: str = "特征筛选流程", *, source=None, relation="auto", strict=True):
        """初始化报告收集器。

        **参数**

        :param name: 流程名称，用于报告中显示，默认值为 "特征筛选流程"

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> collector = SelectionReportCollector(name="我的筛选流程")
        """
        if not isinstance(name, str):
            if source is not None:
                raise ValidationError("请只通过一个参数提供报告来源")
            source, name = name, "特征筛选流程"
        if relation not in {"auto", "independent", "sequential", "intersection", "union"}:
            raise ValidationError("未知筛选报告关系类型")
        self.name = name
        self.relation = relation
        self.strict = strict
        self._snapshots = []
        self.reports: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        self._feature_origin_count: Optional[int] = None
        if source is not None:
            self.add_report(source)

    def add_report(
        self, selector: "BaseFeatureSelector", stage_name: Optional[str] = None
    ) -> "SelectionReportCollector":
        """添加筛选器报告。

        将已拟合的筛选器结果添加到收集器中，并生成阶段名称。阶段名称默认为"阶段{序号}"。

        **参数**

        :param selector: 已拟合的筛选器对象，必须实现 `get_selection_report()` 方法
        :param stage_name: 阶段名称，如 '粗筛'、'精筛' 等，默认根据已有报告数量自动生成

        :returns: self（支持链式调用）

        :raises ValidationError: 当 selector 未实现 `get_selection_report()` 方法时

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector, stage_name="粗筛")
        SelectionReportCollector(name='特征筛选流程', stages=1)
        """
        from .reporting import collect_selection_report

        snapshot = collect_selection_report(selector, relation=self.relation, strict=self.strict)
        if hasattr(selector, "get_selection_report"):
            report = copy.deepcopy(selector.get_selection_report())
        else:
            summary = snapshot.summary
            roots = summary[summary["父路径"].isna()]
            report = {"筛选器": type(selector).__name__, "输入特征数": 0, "选中特征数": 0}
            if len(roots) == 1:
                row = roots.iloc[0]
                report.update({"输入特征数": row["输入特征数"], "选中特征数": row["选中特征数"]})

        # 添加阶段名称
        if stage_name:
            report["stage_name"] = stage_name
        else:
            report["stage_name"] = f"阶段{len(self.reports) + 1}"

        # 记录第一个筛选器的输入特征数作为原始特征数
        if self._feature_origin_count is None and len(self.reports) == 0:
            self._feature_origin_count = report.get("输入特征数")

        self.reports.append(copy.deepcopy(report))
        self._snapshots.append((report["stage_name"], snapshot.copy()))
        return self

    def add_source(self, source, stage_name=None):
        """添加已拟合单项、Pipeline、列表或完整报告，保存隔离快照。"""
        return self.add_report(source, stage_name=stage_name)

    def get_report(self):
        """返回完整统一报告，列表默认独立比较而非伪造顺序漏斗。"""
        from .reporting import collect_selection_report, SelectionReport

        if not self._snapshots:
            return SelectionReport(metadata={"关系类型": "independent", "完整": False, "诊断": ["无筛选记录"]})
        if len(self._snapshots) == 1:
            return self._snapshots[0][1].copy()
        return collect_selection_report(self._snapshots, relation=self.relation, strict=self.strict)

    def get_details(self):
        return self.get_report().details

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总报告。

        汇总报告包含筛选流程的整体统计信息和各阶段的筛选详情。

        **返回字典键值说明**

        - **流程名称** (`str`): 筛选流程名称
        - **创建时间** (`str`): 报告创建时间，格式为 YYYY-MM-DD HH:MM:SS
        - **筛选轮次** (`int`): 执行的筛选阶段总数
        - **原始特征数** (`int`): 第一阶段输入的特征数量
        - **最终特征数** (`int`): 最后一个阶段输出的特征数量
        - **累计剔除特征数** (`int`): 所有阶段累计剔除的特征总数
        - **特征保留率** (`str`): 特征保留比例，格式为 "XX.XX%"，无记录时为 "N/A"
        - **筛选器列表** (`List[Dict]`): 各阶段的筛选器详情列表，每项包含阶段、筛选器、输入、输出、剔除、阈值

        :returns: 包含汇总统计和阶段详情的字典，无记录时返回包含 "状态" 和 "message" 的字典

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> summary = collector.get_summary()
        >>> print(f"原始特征数: {summary['原始特征数']}, 最终特征数: {summary['最终特征数']}")
        原始特征数: 5, 最终特征数: ...
        """
        if len(self.reports) == 0:
            return {"状态": "无筛选记录", "message": "请先添加筛选器报告"}

        if self._snapshots:
            current = self.get_report()
            stages = current.summary
            roots = stages[stages["父路径"].isna()]
            relation = current.metadata.get("关系类型", "independent")
            single = len(roots) == 1
            sequential = relation == "sequential"
            comparable = single or sequential or relation in {"intersection", "union"}
            first = roots.iloc[0] if len(roots) else None
            last = roots.iloc[-1] if len(roots) else None
            original = None if first is None or pd.isna(first["输入特征数"]) else int(first["输入特征数"])
            final_row = first if single or relation in {"intersection", "union"} else last
            final = (
                None
                if not comparable or final_row is None or pd.isna(final_row["选中特征数"])
                else int(final_row["选中特征数"])
            )
            counts_comparable = original is not None and final is not None and final <= original
            if single and pd.isna(first["剔除特征数"]):
                counts_comparable = False
            return {
                "流程名称": self.name,
                "创建时间": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "关系类型": relation,
                "状态": "完整" if current.metadata.get("完整", True) else "不完整",
                "筛选轮次": len(stages),
                "原始特征数": original if comparable else None,
                "最终特征数": final,
                "累计剔除特征数": original - final if counts_comparable else None,
                "特征保留率": f"{final / original * 100:.2f}%" if original and counts_comparable else "不适用",
                "筛选器列表": [
                    {
                        "阶段": row["阶段名称"],
                        "筛选器": row["筛选器"],
                        "输入": row["输入特征数"],
                        "输出": row["选中特征数"],
                        "剔除": row["剔除特征数"],
                        "执行状态": row["执行状态"],
                    }
                    for _, row in stages.iterrows()
                ],
            }

        # 计算统计信息
        total_selected = self.reports[-1].get("选中特征数", 0) if self.reports else 0
        total_dropped = sum(r.get("输入特征数", 0) - r.get("选中特征数", 0) for r in self.reports)

        summary = {
            "流程名称": self.name,
            "创建时间": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "筛选轮次": len(self.reports),
            "原始特征数": self._feature_origin_count,
            "最终特征数": total_selected,
            "累计剔除特征数": total_dropped,
            "特征保留率": (
                f"{total_selected / self._feature_origin_count * 100:.2f}%" if self._feature_origin_count else "N/A"
            ),
            "筛选器列表": [
                {
                    "阶段": r.get("stage_name", f"阶段{i+1}"),
                    "筛选器": r.get("筛选器", r.get("筛选方法", "Unknown")),
                    "输入": r.get("输入特征数", 0),
                    "输出": r.get("选中特征数", 0),
                    "剔除": r.get("输入特征数", 0) - r.get("选中特征数", 0),
                    "阈值": r.get("阈值", "N/A"),
                }
                for i, r in enumerate(self.reports)
            ],
        }

        return summary

    def get_feature_trace(self) -> pd.DataFrame:
        """获取特征追踪表。

        记录每个特征在每个筛选阶段的状态变化，包括选中、剔除或未处理。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 特征名称
        - **阶段** (`str`): 筛选阶段的名称
        - **筛选器** (`str`): 该阶段使用的筛选器类名
        - **状态** (`str`): 特征在当前阶段的状态，取值为 '选中'、'剔除' 或 '未处理'
        - **得分/原因** (`Any`): 选中特征的得分，或剔除特征的原因

        :returns: 特征追踪 DataFrame，无记录时返回空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> trace = collector.get_feature_trace()
        >>> print(trace.head())
        """
        if self._snapshots:
            return self.get_report().get_feature_trace()
        if len(self.reports) == 0:
            return pd.DataFrame()

        # 收集所有特征，按首次出现顺序稳定输出。
        all_features = []
        seen_features = set()
        for r in self.reports:
            for feature in list(r.get("选中特征", [])) + list(r.get("剔除特征", [])):
                if feature not in seen_features:
                    all_features.append(feature)
                    seen_features.add(feature)

        # 构建追踪表
        trace_data = []

        for i, r in enumerate(self.reports):
            stage = r.get("stage_name", f"阶段{i+1}")
            selected = set(r.get("选中特征", []))
            dropped_list = r.get("剔除特征", [])
            dropped_set = set(dropped_list)
            scores = r.get("特征得分", {})
            dropped_reasons = r.get("剔除原因", [])
            reason_by_feature = dict(zip(dropped_list, dropped_reasons))

            for feat in all_features:
                status = "选中" if feat in selected else ("剔除" if feat in dropped_set else "未处理")

                # 获取得分或剔除原因
                if status == "选中":
                    score_value = scores.get(feat, "N/A")
                elif status == "剔除":
                    score_value = reason_by_feature.get(feat, "N/A")
                else:
                    score_value = "N/A"

                trace_data.append(
                    {
                        "特征": feat,
                        "阶段": stage,
                        "筛选器": r.get("筛选器", "Unknown"),
                        "状态": status,
                        "得分/原因": score_value,
                    }
                )

        return pd.DataFrame(trace_data)

    def get_dropped_summary(self) -> pd.DataFrame:
        """获取被剔除特征的汇总表。

        汇总所有筛选阶段中被剔除的特征及其剔除原因。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 被剔除的特征名称
        - **阶段** (`str`): 该特征被剔除的阶段名称
        - **筛选器** (`str`): 该阶段使用的筛选器类名
        - **剔除原因** (`str`): 特征被剔除的原因描述
        - **得分** (`Any`): 特征在筛选器中的得分，无得分时为 'N/A'

        :returns: 剔除特征汇总 DataFrame，无记录时返回空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> dropped = collector.get_dropped_summary()
        >>> print(f"共剔除 {len(dropped)} 个特征")
        """
        if self._snapshots:
            details = self.get_report().details
            dropped = details.loc[
                details["筛选结果"] == "剔除", ["特征", "阶段路径", "筛选器", "筛选原因", "指标值"]
            ].copy()
            return dropped.rename(columns={"阶段路径": "阶段", "筛选原因": "剔除原因", "指标值": "得分"})
        if len(self.reports) == 0:
            return pd.DataFrame()

        dropped_records = []
        for i, r in enumerate(self.reports):
            dropped_features = r.get("剔除特征", [])
            dropped_reasons = r.get("剔除原因", [])

            for j, feat in enumerate(dropped_features):
                reason = dropped_reasons[j] if j < len(dropped_reasons) else "Unknown"
                dropped_records.append(
                    {
                        "特征": feat,
                        "阶段": r.get("stage_name", f"阶段{i+1}"),
                        "筛选器": r.get("筛选器", "Unknown"),
                        "剔除原因": reason,
                        "得分": r.get("特征得分", {}).get(feat, "N/A"),
                    }
                )

        return pd.DataFrame(dropped_records)

    def to_dataframe(self, kind: str = "summary") -> pd.DataFrame:
        """转换为 DataFrame 格式。

        将收集到的各筛选阶段报告汇总为一个 DataFrame，每行对应一个筛选阶段。

        **返回 DataFrame 列说明**

        - **阶段** (`str`): 筛选阶段的名称
        - **筛选器** (`str`): 该阶段使用的筛选器类名
        - **阈值** (`Any`): 该阶段使用的筛选阈值
        - **输入特征数** (`int`): 该阶段输入的特征数量
        - **选中特征数** (`int`): 该阶段输出的特征数量
        - **剔除特征数** (`int`): 该阶段剔除的特征数量

        :returns: 筛选结果汇总 DataFrame，无记录时返回空 DataFrame

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> df = collector.to_dataframe()
        >>> print(df)
        """
        if kind not in {"summary", "details"}:
            raise ValidationError("kind 必须为 summary 或 details")
        if self._snapshots or kind == "details":
            result = self.get_report()
            return result.summary if kind == "summary" else result.details
        if len(self.reports) == 0:
            return pd.DataFrame()

        rows = []
        for i, r in enumerate(self.reports):
            row = {
                "阶段": r.get("stage_name", f"阶段{i+1}"),
                "筛选器": r.get("筛选器", r.get("筛选方法", "Unknown")),
                "阈值": r.get("阈值", "N/A"),
                "输入特征数": r.get("输入特征数", 0),
                "选中特征数": r.get("选中特征数", 0),
                "剔除特征数": r.get("输入特征数", 0) - r.get("选中特征数", 0),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def to_excel(self, filepath: str, **kwargs):
        """以版本目录和完成清单发布完整Excel/JSON，返回实际产物路径与状态。"""
        return self.get_report().save(filepath, formats=("xlsx", "json"), **kwargs)

    def print_summary(self) -> None:
        """打印汇总报告到控制台。

        以格式化的方式打印筛选流程的整体统计信息和各阶段详情。

        **打印内容**

        - 筛选流程名称和创建时间
        - 筛选轮次、原始/最终特征数、累计剔除数、特征保留率
        - 各阶段筛选器详情（阶段名、筛选器名称、输入/输出/剔除特征数）

        **参考样例**

        >>> from hscredit.core.selectors.base import SelectionReportCollector
        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 100))
        >>> collector = SelectionReportCollector()
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> collector.add_report(selector)
        >>> collector.print_summary()  # doctest: +SKIP
        """
        summary = self.get_summary()

        if summary.get("状态") == "无筛选记录":
            print(summary["状态"])
            print(summary["message"])
            return

        print("=" * 60)
        print(f"特征筛选报告 - {summary['流程名称']}")
        print("=" * 60)
        print(f"创建时间: {summary['创建时间']}")
        print(f"筛选轮次: {summary['筛选轮次']}")
        print(f"原始特征数: {summary['原始特征数']}")
        print(f"最终特征数: {summary['最终特征数']}")
        print(f"累计剔除: {summary['累计剔除特征数']}")
        print(f"特征保留率: {summary['特征保留率']}")
        print()
        print("筛选详情:")
        print("-" * 60)

        def _cjk_ljust(s: str, width: int) -> str:
            import unicodedata

            display_w = sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in s)
            return s + " " * max(0, width - display_w)

        print(f"{_cjk_ljust('阶段', 10)} {_cjk_ljust('筛选器', 20)} {'输入':>6} {'输出':>6} {'剔除':>6}")
        print("-" * 60)
        for item in summary["筛选器列表"]:
            print(
                f"{_cjk_ljust(str(item['阶段']), 10)} {_cjk_ljust(str(item['筛选器']), 20)} {str(item['输入']):>6} {str(item['输出']):>6} {str(item['剔除']):>6}"
            )
        print("=" * 60)

    def __len__(self) -> int:
        """返回已添加的筛选器数量。

        :returns: 已收集的报告数量
        """
        return len(self.reports)

    def __repr__(self) -> str:
        """返回收集器的字符串表示。

        :returns: 形如 ``SelectionReportCollector(name='...', stages=N)`` 的字符串
        """
        return f"SelectionReportCollector(name='{self.name}', stages={len(self.reports)})"


class BaseFeatureSelector(ParallelizableMixin, TransformerMixin, BaseEstimator, ABC):
    """特征筛选器基类.

    所有特征筛选器都继承此类，实现统一的 fit/transform 接口。
    支持中文筛选报告生成。支持可选的分箱器，在筛选前对数据进行分箱处理。

    **参数**

    :param target: 目标变量列名，默认为 'target'。在 scorecardpipeline 风格中用于
        从 DataFrame 中提取目标列；在 sklearn 风格中若 fit 传入了 y 参数则优先使用 y
    :param include: 强制保留的特征列表，这些特征无论如何都会被保留
    :param exclude: 强制剔除的特征列表，这些特征无论如何都会被剔除
    :param binner: 可选的已配置分箱器实例，支持已训练或待训练状态，不接受分箱器类
    :param binning_params: 可选的 ``OptimalBinning`` 构造参数字典。未传入 ``binner`` 时，
        基类使用该字典创建分箱器；同时传入时 ``binner`` 优先
    :param threshold: 筛选阈值，不同筛选器含义不同
    :param n_jobs: 并行工作数，默认为 -1；None 沿用旧串行行为
    :param force_drop: 强制剔除的特征列表，效果与 exclude 合并
    :param parallel_backend: joblib 并行后端，默认为 None
    :param parallel_config: joblib 扩展配置，默认为 None

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
        >>> report = selector.get_selection_report()
        >>> print(f"输入: {report['输入特征数']}, 输出: {report['选中特征数']}")
        输入: 2, 输出: 1
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 目标列是否存在由本次transform输入决定，不能让sklearn按fit时的
        # 固定字段名再次重命名。容器转换由本类使用本次实际输出完成。
        cls._sklearn_auto_wrap_output_keys = set()

    def set_output(self, *, transform=None):
        """配置输出容器；按本次输入决定目标列，不改写拟合字段或报告快照。"""
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
        binning_params: Optional[Dict[str, Any]] = None,
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
        :param binning_params: 可选的 ``OptimalBinning`` 构造参数字典；``binner`` 优先
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
        self.binning_params = binning_params
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

        1. **sklearn 风格**: ``fit(X, y)`` — X 是特征矩阵，y 是目标变量
        2. **scorecardpipeline 风格**: ``fit(df)`` — df 是包含特征和目标列的完整 DataFrame，
           目标列名通过初始化参数 target 指定

        **优先级规则**: 如果 y 不为 None，使用 sklearn 风格；否则检查 X 中是否包含 target 列。

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
        """事务式拟合筛选器，失败时恢复进入本次拟合前的完整状态。"""
        started = perf_counter()
        candidate = clone(self)
        if hasattr(self, "_sklearn_output_config"):
            candidate._sklearn_output_config = copy.deepcopy(self._sklearn_output_config)
        self._prepare_fit_candidate(candidate)
        candidate._fit_once(X, y)
        candidate._finalize_fit()
        candidate.fit_id_ = uuid4().hex
        candidate.fit_duration_seconds_ = perf_counter() - started
        candidate._selection_dropped_columns_ = list(getattr(candidate, "dropped_", pd.DataFrame()).columns)
        candidate._selection_report_dict_ = copy.deepcopy(candidate.get_selection_report())
        from .reporting import capture_selection_report

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

        支持两种 API 风格：sklearn 风格 ``fit(X, y)`` 和 scorecardpipeline 风格 ``fit(df)``。

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
        """按 ``binner > binning_params`` 解析有效分箱器。"""
        if self.binner is not None:
            if isinstance(self.binner, type):
                raise ValidationError("binner 必须传入配置好的分箱器实例，不能传入分箱器类")
            return self.binner

        if self.binning_params is None:
            return None
        if not isinstance(self.binning_params, dict):
            raise ValidationError("binning_params 分箱参数必须是字典")

        from ..binning import OptimalBinning

        binning_params = dict(self.binning_params)
        # 内部分箱器属于筛选器调用树的一部分。只有子级未显式配置时才
        # 继承父级预算；用户在 binning_params 中给出的更具体配置优先。
        binning_params.setdefault("n_jobs", self.n_jobs)
        binning_params.setdefault("parallel_backend", self.parallel_backend)
        binning_params.setdefault(
            "parallel_config",
            dict(self.parallel_config) if self.parallel_config is not None else None,
        )
        return OptimalBinning(**binning_params)

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
        return self.binner is not None or self.binning_params is not None

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
        :param y: 目标变量，可选

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
        """返回选择掩码；``indices=True`` 时返回选中列位置。"""
        mask = self.get_support_mask()
        return np.flatnonzero(mask) if indices else mask

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """返回转换后的字段名，兼容 sklearn Pipeline。"""
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

    def get_selection_report(self) -> Dict[str, Any]:
        """获取中文筛选报告。

        报告包含筛选器的基础信息、统计信息、选中/剔除特征列表及特征得分。

        **返回字典键值说明**

        - **筛选器** (`str`): 筛选器类名
        - **筛选方法** (`str`): 筛选方法名称
        - **时间戳** (`str`): 报告生成时间，格式为 YYYY-MM-DD HH:MM:SS
        - **阈值** (`Any`): 筛选阈值
        - **参数** (`Dict`): 筛选器初始化参数（过滤后的有效参数）
        - **强制操作** (`Dict` | `None`): 强制保留/剔除的特征信息
        - **输入特征数** (`int`): fit 时的输入特征数量
        - **选中特征数** (`int`): 选中的特征数量
        - **剔除特征数** (`int`): 被剔除的特征数量
        - **特征保留率** (`str`): 保留比例，格式为 "XX.XX%"
        - **选中特征** (`List[str]`): 选中特征名称列表
        - **剔除特征** (`List[str]`，可选): 被剔除特征名称列表
        - **剔除原因** (`List[str]`，可选): 各剔除特征对应的原因
        - **剔除详情** (`List[Dict]`，可选): 剔除特征的详细信息
        - **特征得分** (`Dict`): 各特征得分的字典
        - **得分统计** (`Dict`，可选): 得分的统计摘要（最大值、最小值、平均值、中位数）

        :returns: 包含筛选结果的字典，未拟合时返回包含 "状态" 和 "message" 的字典

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
        >>> report = selector.get_selection_report()
        >>> print(f"输入: {report['输入特征数']}, 选中: {report['选中特征数']}")
        输入: 5, 选中: ...
        """
        if hasattr(self, "_selection_report_dict_"):
            return copy.deepcopy(self._selection_report_dict_)
        if not getattr(self, "_is_fitted", False):
            return {"状态": "未拟合", "message": "请先调用fit方法"}

        # 收集参数
        params = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and key not in [
                "n_features_in_",
                "selected_features_",
                "removed_features_",
                "scores_",
                "dropped_",
            ]:
                if isinstance(value, (str, int, float, bool, type(None))):
                    params[key] = value

        # 处理threshold参数（可能名称不同）
        if hasattr(self, "threshold") and self.threshold != 0.0:
            params["threshold"] = self.threshold

        # 添加强制保留/剔除的特征信息
        force_info = {}
        if hasattr(self, "include_") and self.include_:
            force_info["强制保留"] = [feature for feature in self.include_ if feature in self.selected_features_]
        if hasattr(self, "forced_dropped_") and self.forced_dropped_:
            force_info["强制剔除"] = self.forced_dropped_

        # 构建报告
        report = {
            # 基础信息
            "筛选器": self.__class__.__name__,
            "筛选方法": getattr(self, "method_name", self.__class__.__name__),
            "时间戳": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 参数信息
            "阈值": self.threshold,
            "参数": params,
            # 强制保留/剔除信息
            "强制操作": force_info if force_info else None,
            # 统计信息
            "输入特征数": self.n_features_in_,
            "选中特征数": len(self.selected_features_),
            "剔除特征数": self.n_features_in_ - len(self.selected_features_),
            "特征保留率": (
                f"{len(self.selected_features_) / self.n_features_in_ * 100:.2f}%" if self.n_features_in_ > 0 else "0%"
            ),
            # 特征列表
            "选中特征": list(self.selected_features_),
        }

        # 添加dropped信息（DataFrame格式,便于后续分析）
        if hasattr(self, "dropped_") and len(self.dropped_) > 0:
            report["剔除特征"] = self.dropped_["特征"].tolist()
            report["剔除原因"] = self.dropped_["剔除原因"].tolist()
            report["剔除详情"] = self.dropped_.to_dict("records")

        # 添加scores信息
        if hasattr(self, "scores_") and self.scores_ is not None:
            scores_raw = self.scores_
            if isinstance(scores_raw, pd.Series):
                raw_dict = scores_raw.to_dict()
            elif isinstance(scores_raw, np.ndarray):
                feature_names = getattr(self, "feature_names_", None)
                if feature_names is None:
                    feature_names = getattr(self, "selection_input_features_", self._feature_names)
                raw_dict = dict(zip(feature_names, scores_raw))
            elif isinstance(scores_raw, dict):
                raw_dict = scores_raw
            else:
                raw_dict = {}
            scores_dict = {}
            for k, v in raw_dict.items():
                if isinstance(v, (np.integer, np.floating)):
                    scores_dict[k] = float(v)
                else:
                    scores_dict[k] = v
            report["特征得分"] = scores_dict

            # 添加得分统计
            valid_scores = [v for v in scores_dict.values() if isinstance(v, (int, float)) and np.isfinite(v)]
            if valid_scores:
                report["得分统计"] = {
                    "最大值": max(valid_scores),
                    "最小值": min(valid_scores),
                    "平均值": sum(valid_scores) / len(valid_scores),
                    "中位数": float(np.median(valid_scores)),
                }

        return copy.deepcopy(report)

    def get_selection_result(self):
        """返回与模型状态隔离的完整报告；读取报告不重新拟合或计算指标。"""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")
        from .reporting import capture_selection_report

        report = getattr(self, "selection_report_", None)
        if report is None:
            report = capture_selection_report(self)
        return report.copy()

    def get_selection_details(self) -> pd.DataFrame:
        """统一逐特征决策表，含强制操作和未计算指标，所有筛选器列结构一致。"""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")
        report = getattr(self, "selection_report_", None)
        return report.details if report is not None else self.get_selection_result().details

    def get_selection_report_df(self, kind: str = "details") -> pd.DataFrame:
        """获取固定列明细；kind='summary'兼容原有单行汇总。

        适用于快速查看和导出到 Excel 等场景。

        **summary 模式返回列说明**

        - **筛选器** (`str`): 筛选器类名
        - **筛选方法** (`str`): 筛选方法名称
        - **阈值** (`Any`): 筛选阈值
        - **输入特征数** (`int`): 输入特征数量
        - **选中特征数** (`int`): 选中特征数量
        - **剔除特征数** (`int`): 剔除特征数量
        - **保留率** (`str`): 特征保留比例

        :param kind: details返回完整逐特征决策表，summary返回单行统计
        :returns: 与其它筛选器相同结构的明细，或显式选择的单行汇总

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
        >>> report_df = selector.get_selection_report_df()
        >>> print(report_df)
        """
        if kind == "details":
            return self.get_selection_details()
        if kind != "summary":
            raise ValidationError("kind 必须为 details 或 summary")
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError("筛选器尚未拟合，请先调用fit方法")
        report = self.get_selection_report()

        # 提取关键信息
        row = {
            "筛选器": report.get("筛选器", ""),
            "筛选方法": report.get("筛选方法", ""),
            "阈值": report.get("阈值", ""),
            "输入特征数": report.get("输入特征数", 0),
            "选中特征数": report.get("选中特征数", 0),
            "剔除特征数": report.get("剔除特征数", 0),
            "保留率": report.get("特征保留率", ""),
        }

        return pd.DataFrame([row])

    def get_scores_df(self) -> pd.DataFrame:
        """获取特征得分的 DataFrame。

        **返回 DataFrame 列说明**

        - **特征** (`str`): 特征名称
        - **得分** (`float`): 特征在筛选器中的得分
        - **状态** (`str`): 特征状态，取值为 '选中' 或 '剔除'

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
        report = self.get_selection_report()
        scores = report.get("特征得分", {})
        selected = set(report.get("选中特征", []))

        records = []
        for feat in getattr(self, "_selection_input_names_", self._feature_names):
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

        - **特征** (`str`): 被剔除的特征名称
        - **剔除原因** (`str`): 特征被剔除的原因描述

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
        if hasattr(self, "_selection_report_dict_"):
            return pd.DataFrame(
                copy.deepcopy(self._selection_report_dict_.get("剔除详情", [])),
                columns=self._selection_dropped_columns_,
            )
        if hasattr(self, "dropped_"):
            return self.dropped_.copy(deep=True)
        return pd.DataFrame(columns=["特征", "剔除原因"])

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

    :param selectors: 筛选器列表，按执行顺序排列
    :param strategy: 组合策略，可选 'sequential' 或 'intersection'
        - 'sequential': 按顺序筛选，每轮剔除不满足条件的特征
        - 'intersection': 取所有筛选器选中特征的交集
    :param include: 强制保留的特征列表，这些特征无论如何都会被保留
    :param exclude: 强制剔除的特征列表，这些特征无论如何都会被剔除
    :param target: 目标变量列名，默认为 'target'
    :param binner: 可选的分箱器

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
        binning_params: Optional[Dict[str, Any]] = None,
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
        :param binning_params: 可选的 OptimalBinning 构造参数
        """
        super().__init__(
            target=target,
            include=include,
            exclude=exclude,
            binner=binner,
            binning_params=binning_params,
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

    def get_selection_report_df(self, kind: str = "details") -> pd.DataFrame:
        """与单项筛选器相同的固定列明细；完整子步骤使用collect_selection_report。"""
        return super().get_selection_report_df(kind=kind)
