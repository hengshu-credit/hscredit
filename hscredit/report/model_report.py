# -*- coding: utf-8 -*-
"""模型评估报告快速输出.

参考风控建模标准报告模板，提供多 Sheet 结构的模型报告，包括：
- 目录（带超链接）
- 基本信息（项目目标、样本统计、分月分布）
- 模型性能（KS/AUC/PSI、TOP n% LIFT、分月PSI、评分分箱）
- 入模变量重要性 & 分布
- 入模变量有效性分析（逐特征分箱表 + 金额口径 + PSI）
- 模型参数（评分卡详情）
- 模型部署需求
"""

from __future__ import annotations

import logging

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._sample_stats import build_group_distribution_table, build_sample_stats_table

logger = logging.getLogger(__name__)

_SUMMARY_PERCENT_METRICS = {"KS", "AUC", "坏样本率"}


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _ensure_dataframe(X, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    cols = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _ensure_series(y, name: str = "target") -> pd.Series:
    if isinstance(y, pd.Series):
        out = y.copy()
        if out.name is None:
            out.name = name
        return out
    return pd.Series(np.asarray(y), name=name)


def _proba_pos(model, X) -> np.ndarray:
    """获取正类概率."""
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim == 2 and proba.shape[1] >= 2:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            positive = np.flatnonzero(np.asarray(classes) == 1)
            if len(positive) == 1:
                return proba[:, positive[0]]
        return proba[:, 1]
    return proba.reshape(-1)


def _safe_binary_metric(metric, y_true, y_score) -> float:
    """计算二分类指标，单类别样本返回 NaN。"""
    y_arr = np.asarray(y_true)
    if len(y_arr) == 0 or len(np.unique(y_arr[~pd.isna(y_arr)])) < 2:
        return np.nan
    try:
        return float(metric(y_arr, y_score))
    except (TypeError, ValueError):
        return np.nan


def _summary_percent_cols(columns) -> List[Any]:
    """识别 summary 表中需要按百分比显示的列."""
    percent_cols: List[Any] = []
    for col in columns:
        metric = col[0] if isinstance(col, tuple) and col else col
        if metric in _SUMMARY_PERCENT_METRICS:
            percent_cols.append(col)
    return percent_cols


def _score_from_model(model, X) -> np.ndarray:
    """从模型获取评分向量，兼容 ScoreCard / BaseRiskModel / sklearn."""
    # ScoreCard.predict → 评分
    if hasattr(model, "predict"):
        try:
            result = np.asarray(model.predict(X), dtype=float)
            if np.nanmax(np.abs(result)) > 2.0:
                return result
        except Exception:
            pass
    # predict_score（BaseRiskModel 子类）
    if hasattr(model, "predict_score"):
        try:
            return np.asarray(model.predict_score(X), dtype=float)
        except Exception:
            pass
    # 兜底：概率转评分
    proba = _proba_pos(model, X)
    return (1.0 - proba) * 1000.0


def _safe_close_figs():
    """安全关闭 matplotlib 图形以释放内存."""
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass


def _merge_multi_label_bin_tables(tables: List[pd.DataFrame], labels: List[str]) -> pd.DataFrame:
    """将多个单标签分箱表合并为 MultiIndex 列的合并表。

    列结构：分箱详情列保持不变，其余列变为 (标签名, 列名) 的 MultiIndex。
    """
    if not tables:
        return pd.DataFrame()
    if len(tables) == 1:
        return tables[0]

    base = tables[0].copy()
    merge_cols = ["分箱标签", "指标含义", "指标名称", "指标明细"]
    available_merge = [c for c in merge_cols if c in base.columns]
    other_cols = [c for c in base.columns if c not in available_merge]

    # 重排列：合并列在前，明细列在后
    base = base[available_merge + other_cols]

    # 当没有可合并列时（cross-product 陷阱），改用 index 对齐拼接
    if not available_merge:
        result_parts = [base]
        for table, lbl in zip(tables[1:], labels[1:]):
            table_copy = table.copy()
            table_other = [c for c in table_copy.columns if c not in available_merge]
            table_multi_cols = [("分箱详情", c) for c in available_merge] + [(lbl, c) for c in table_other]
            table_copy.columns = pd.MultiIndex.from_tuples(table_multi_cols)
            result_parts.append(table_copy[table_other])
        result = pd.concat(result_parts, axis=1, keys=[labels[0]] + list(labels[1:]), names=["标签"])
        return result

    # 构建 MultiIndex 列
    multi_cols = []
    for col in base.columns:
        if col in available_merge:
            multi_cols.append(("分箱详情", col))
        else:
            multi_cols.append((labels[0], col))
    base.columns = pd.MultiIndex.from_tuples(multi_cols)

    for table, lbl in zip(tables[1:], labels[1:]):
        table_copy = table.copy()
        table_multi_cols = []
        for col in table_copy.columns:
            if col in available_merge:
                table_multi_cols.append(("分箱详情", col))
            else:
                table_multi_cols.append((lbl, col))
        table_copy.columns = pd.MultiIndex.from_tuples(table_multi_cols)
        merge_on = [("分箱详情", c) for c in available_merge]
        base = base.merge(table_copy, on=merge_on)

    return base


def _drop_bin_meta_cols(table: pd.DataFrame) -> pd.DataFrame:
    """删除分箱表中的 ``指标名称`` / ``指标含义`` 列（兼容扁平列与 MultiIndex 列）。"""
    drop_names = {"指标名称", "指标含义"}
    if isinstance(table.columns, pd.MultiIndex):
        keep = [c for c in table.columns if c[-1] not in drop_names]
    else:
        keep = [c for c in table.columns if c not in drop_names]
    return table[keep]


def _next_image_col(ws, start_col: int, width_px: int, default_width: float = 13.0) -> int:
    """根据图片像素宽度与各列实际列宽，计算紧邻图片右侧的下一列号。

    避免使用 ``insert_pic2sheet`` 固定 +8 列的返回值导致图片之间出现多余空列，
    使多张图片首尾紧挨着排列。
    """
    from openpyxl.utils import get_column_letter

    acc = 0.0
    col = start_col
    # 防御性上限，避免列宽异常时陷入死循环
    while acc < width_px and col < start_col + 60:
        dim = ws.column_dimensions.get(get_column_letter(col))
        width = dim.width if (dim is not None and dim.width) else default_width
        acc += width * 7.0 + 5.0
        col += 1
    return col


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------


@dataclass
class ReportDataset:
    name: str
    label: str  # 中文标签: "训练集" / "测试集" / "OOT"
    X: pd.DataFrame
    y: pd.Series
    y_proba: np.ndarray
    score: np.ndarray
    y_dict: Optional[Dict[str, np.ndarray]] = None  # {label_name: y_array}，多标签场景下各标签的独立标签


# ---------------------------------------------------------------------------
# ModelReport
# ---------------------------------------------------------------------------


class ModelReport:
    """面向报表输出的快速模型报告封装.

    参考风控建模标准报告模板，对已训练模型一站式生成多 Sheet 结构的
    Excel / HTML 报告，并提供各分项结果的获取方法（指标、分箱表、特征重要性、
    描述统计、相关性等）。支持任意多个数据集（训练/测试/OOT…）的横向对比，
    以及 ``overdue`` + ``dpds`` 的多逾期标签构建。

    **参数**

    （完整说明见 :meth:`__init__`）

    :param model: 已训练好的模型，需实现 ``predict`` / ``predict_proba``
    :param datasets: 数据集字典或列表（推荐），如 ``{'train': df, 'test': df}``；
        DataFrame 需含目标列或配合 ``overdue``/``dpds`` 构建标签
    :param X_train/y_train/X_test/y_test: 兼容 sklearn 风格的数据传入方式
    :param target: 目标列名（sklearn/scorecardpipeline 风格）
    :param overdue: 逾期天数列名或列表，配合 ``dpds`` 自动构建 0/1 标签
    :param dpds: 逾期定义天数或列表（逾期天数 > dpds 记为坏样本）
    :param feature_names: 特征名称列表，可选

    **属性**

    - model: 传入的已训练模型
    - feature_names: 最终使用的特征名称列表
    - _datasets: 解析后的各数据集（key -> :class:`ReportDataset`）

    **参考样例**

    >>> from hscredit.report import ModelReport
    >>> report = ModelReport(model, datasets={'train': train_df, 'test': test_df})
    >>> report.get_metrics()            # 各数据集指标对比
    >>> report.get_feature_importance(top_n=20)
    >>> report.to_excel('模型报告.xlsx')  # 导出多 Sheet 报告
    """

    _PERCENT_COLS = [
        "样本占比",
        "好样本占比",
        "坏样本占比",
        "坏样本率",
        "LIFT值",
        "坏账改善",
        "累积LIFT值",
        "累积坏账改善",
        "分档KS值",
    ]
    _CONDITION_COLS = ["坏样本率", "LIFT值", "累积LIFT值"]

    def __init__(
        self,
        model,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        feature_names: Optional[List[str]] = None,
        target: Optional[Union[str, Dict]] = None,
        datasets: Optional[Union[List, Dict]] = None,
        overdue: Optional[Union[str, List[str]]] = None,
        dpds: Optional[Union[int, float, List[Union[int, float]]]] = None,
    ):
        """初始化模型报告.

        支持三种调用方式：

        1. datasets API（推荐）：传入数据集字典/列表
           - dict: {'train': DataFrame, 'test': DataFrame, 'oot': DataFrame}
             DataFrame 需包含目标列，或通过 overdue/dpds 自动构建标签
           - list: [DataFrame, DataFrame, ...] 自动命名为训练集、测试集、OOT集...

        2. 兼容 API：传入 X_train/y_train/X_test/y_test
           - sklearn 风格：target='target'
           - overdue/dpds 风格：传入单独的 overdue/dpds 参数

        3. datasets dict（最高优先级）：显式指定各数据集
           覆盖 X_train/y_train/X_test/y_test

        示例::

            # 方式1: datasets dict（DataFrame 直接传入，X 中含目标列）
            report = ModelReport(model, datasets={'train': train_df, 'test': test_df})

            # 方式1: datasets list（自动命名为训练集、测试集）
            report = ModelReport(model, datasets=[train_df, test_df])

            # 方式1: overdue/dpds 自动构建标签（X 中不含目标列）
            report = ModelReport(
                model,
                datasets={'train': df},
                overdue='dpds',     # 逾期天数列名
                dpds=[15, 7, 0],    # 任一 MOB 下 DPD > threshold 则 y=1
            )

            # 方式2: 兼容 sklearn API
            report = ModelReport(model, X_train=X, y_train=y, X_test=X_val, y_test=y_val)

            # 方式2: overdue/dpds 作为独立参数
            report = ModelReport(
                model,
                X_train=df,
                overdue='dpds',    # 逾期列名
                dpds=5,            # DPD > 5 则标记为坏样本
            )

        :param model: 训练好的模型（ScoreCard / XGBoost / LightGBM / sklearn 等）
        :param datasets: 数据集字典/列表（推荐方式）
        :param X_train: 训练集特征（兼容旧 API）
        :param y_train: 训练集标签（兼容旧 API）
        :param X_test: 测试集特征（兼容旧 API）
        :param y_test: 测试集标签（兼容旧 API）
        :param feature_names: 特征名称列表
        :param target: 目标列配置
            - str: 列名，如 'target'
            - dict: {'overdue': col, 'dpds': threshold} 或 {'overdue': col, 'dpds': [15, 7, 0]}
        :param overdue: 逾期列名（str）或多个列名（List[str]），与 dpds 配合自动构建标签
        :param dpds: 逾期天数阈值（int/float）或多个阈值（List），与 overdue 配合使用
        """
        self.model = model
        self._feature_names = feature_names

        # overdue/dpds 优先，构造 target dict
        if overdue is not None and dpds is not None:
            self._target_cfg: Optional[Union[str, Dict]] = {
                "overdue": overdue,
                "dpds": dpds,
            }
        else:
            self._target_cfg = target

        # 构建数据集
        self._datasets: Dict[str, ReportDataset] = {}
        self._datasets_info: Dict[str, str] = {}  # key -> label

        # 多标签指标名列表（用于多标签报告），按构建顺序从第一个数据集获取
        # 当 overdue/dpds 产生多指标时填充，如 ['dpds_m1>15', 'dpds_m1>7', 'dpds_m1>0', ...]
        self._label_names: List[str] = []

        # 确定目标列名
        self._target_name = self._resolve_target_name(target)

        if datasets is not None:
            self._init_from_datasets(datasets)
        else:
            self._init_from_xy(X_train, y_train, X_test, y_test)

        # 从第一个数据集获取特征名，再过滤为模型实际入模特征
        if self._feature_names:
            self.feature_names = list(self._feature_names)
        elif not hasattr(self, "feature_names") or not self.feature_names:
            if self._datasets:
                first_ds = next(iter(self._datasets.values()))
                self.feature_names = list(first_ds.X.columns)
            elif self._feature_names:
                self.feature_names = self._feature_names
            else:
                self.feature_names = []

        # 统一为模型实际入模特征（排除数据集传入的非入模字段如 MOB1、放款金额 等）
        model_required: Optional[List[str]] = None
        if hasattr(self.model, "feature_names_") and self.model.feature_names_ is not None:
            model_required = list(self.model.feature_names_)
        elif hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            model_required = list(self.model.feature_names_in_)

        if model_required:
            # 只保留模型实际入模特征，同时保留原始顺序
            self.feature_names = [f for f in self.feature_names if f in model_required]

        # 缓存
        self._metrics_cache: Optional[pd.DataFrame] = None
        self._importance_cache: Optional[pd.DataFrame] = None
        self._features_describe_cache: Optional[pd.DataFrame] = None

    def _resolve_target_name(self, target) -> str:
        """解析目标配置，返回标签列名.

        overdue/dpds 作为单独参数传入时，target 参数将被忽略，
        标签列名默认为 'target'。
        """
        if isinstance(target, str):
            return target
        if isinstance(target, dict) and "overdue" in target:
            return target.get("label", "target")
        return "target"

    def _build_y(self, X: pd.DataFrame, target_cfg) -> Tuple[pd.Series, Optional[Dict[str, np.ndarray]]]:
        """根据 target 配置从 X 构建 y 标签.

        支持三种配置：
        - None: 从 X 中查找 'target' 列
        - str: 直接取 X[target] 作为标签
        - dict: 联合构建标签
            - 单逾期列:  target={'overdue': col, 'dpds': threshold} 或
                        target={'overdue': col, 'dpds': [t1, t2, ...]}
            - 多逾期列:  target={'overdue': [col1, col2], 'dpds': [t1, t2, ...]}
                          每列 × 每阈值生成指标，任一为真则 label=1

        返回 (y_series, y_dict)：
        - y_series: 聚合标签（任一指标为真则 y=1）
        - y_dict: 多标签场景下各指标的独立标签数组（key 如 'dpds_m1>15'）
                  单标签或非多指标场景下返回 None
        """
        if target_cfg is None:
            for col in ("target", "label", "y", "flag", "overdue"):
                if col in X.columns:
                    return _ensure_series(X[col], name="target"), None
            raise ValueError(
                "未找到目标列（target），请通过 target 参数指定标签列名，"
                "或传入 dict={'overdue': col, 'dpds': threshold} 联合构建"
            )

        if isinstance(target_cfg, str):
            if target_cfg in X.columns:
                return _ensure_series(X[target_cfg], name=target_cfg), None
            raise ValueError(f"目标列 '{target_cfg}' 不存在于数据中")

        if isinstance(target_cfg, dict) and "overdue" in target_cfg:
            overdue_cols = target_cfg["overdue"]
            dpds_vals = target_cfg.get("dpds")
            threshold = target_cfg.get("threshold")
            label_name = target_cfg.get("label", "target")

            # 统一为列表
            if isinstance(overdue_cols, str):
                overdue_cols = [overdue_cols]

            # 支持旧格式 threshold 键，或新格式 dpds 作为阈值
            # 旧格式: {'overdue': col, 'dpds': col, 'threshold': 3}
            # 新格式: {'overdue': col, 'dpds': [15, 7, 0]}
            if threshold is not None:
                # 旧格式：dpds 为列名，threshold 为阈值
                dpds_col = dpds_vals if isinstance(dpds_vals, str) else None
                thresholds = [threshold]
            elif dpds_vals is not None:
                if isinstance(dpds_vals, (int, float)):
                    dpds_vals = [dpds_vals]
                thresholds = dpds_vals
                dpds_col = None
            else:
                # 只有 overdue，无 dpds/threshold：overdue 列值 > 0 → y=1
                thresholds = [0]
                dpds_col = None

            # 验证列名
            for col in overdue_cols:
                if col not in X.columns:
                    raise ValueError(f"逾期列 '{col}' 不存在，请检查列名")

            # 每列 × 每阈值，生成全指标
            indicators = pd.DataFrame(index=X.index)
            for col in overdue_cols:
                for t in thresholds:
                    if dpds_col is not None and dpds_col in X.columns:
                        # dpds 列 > threshold
                        indicators[f"{col}>{t}"] = X[dpds_col] > t
                    else:
                        # col 列 > threshold
                        indicators[f"{col}>{t}"] = X[col] > t

            # 聚合标签（任一指标为真则 y=1）
            y = indicators.any(axis=1).astype(int)
            # 多指标时返回各指标独立标签，供多标签报告使用
            y_dict: Optional[Dict[str, np.ndarray]] = None
            if len(overdue_cols) > 1 or (isinstance(dpds_vals, list) and len(dpds_vals) > 1):
                y_dict = {col: indicators[col].values.astype(np.int8) for col in indicators.columns}
            return _ensure_series(y, name=label_name), y_dict

        raise ValueError(f"target 参数格式错误：{target_cfg}")

    def _init_from_datasets(self, datasets):
        """从 datasets 初始化数据集.

        datasets 支持两种格式：
        - dict: {'train': DataFrame, 'test': DataFrame, ...}
                  DataFrame 直接传入，y 从 X 中通过 target / overdue+dpds 自动构建
        - list: [DataFrame, DataFrame, ...]
                  自动命名为训练集、测试集、OOT集...，y 从 X 中自动构建
        """
        if isinstance(datasets, dict):
            default_labels = {
                "train": "训练集",
                "test": "测试集",
                "oot": "OOT集",
                "val": "验证集",
            }
            for key, value in datasets.items():
                # 区分 (X, y) 元组 和 直接传入 DataFrame 两种格式
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    # 传统元组格式: (X, y)
                    X_raw, y_raw = value[0], value[1]
                    label = default_labels.get(key, key)
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    if y_raw is None:
                        y_s, y_dict = self._build_y(X_df, self._target_cfg)
                        if y_dict and not self._label_names:
                            self._label_names = list(y_dict.keys())
                    else:
                        y_s = _ensure_series(y_raw, name=self._target_name)
                        y_dict = None
                else:
                    # DataFrame 直接传入: X 中含目标列或通过 overdue+dpds 构建标签
                    X_raw = value
                    label = default_labels.get(key, key)
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    y_s, y_dict = self._build_y(X_df, self._target_cfg)
                    if y_dict and not self._label_names:
                        self._label_names = list(y_dict.keys())

                self._add_dataset(key, label, X_df, y_s, y_dict)
                self._datasets_info[key] = label

        elif isinstance(datasets, (list, tuple)):
            default_names = ["train", "test", "oot", "val", "dev"]
            default_labels = ["训练集", "测试集", "OOT集", "验证集", "开发集"]
            for i, value in enumerate(datasets):
                key = default_names[i] if i < len(default_names) else f"dataset_{i}"
                label = default_labels[i] if i < len(default_labels) else f"数据集{i+1}"
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    X_raw, y_raw = value[0], value[1]
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    if y_raw is None:
                        y_s, y_dict = self._build_y(X_df, self._target_cfg)
                        if y_dict and not self._label_names:
                            self._label_names = list(y_dict.keys())
                    else:
                        y_s = _ensure_series(y_raw, name=self._target_name)
                        y_dict = None
                else:
                    X_raw = value
                    X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)
                    y_s, y_dict = self._build_y(X_df, self._target_cfg)
                    if y_dict and not self._label_names:
                        self._label_names = list(y_dict.keys())

                self._add_dataset(key, label, X_df, y_s, y_dict)
                self._datasets_info[key] = label

    def _init_from_xy(self, X_train, y_train, X_test, y_test):
        """从 X/y 参数初始化（兼容旧 API 及 scorecardpipeline 风格）."""
        X_train_df = _ensure_dataframe(X_train, feature_names=self._feature_names)

        # 支持 y_train 为 None 的 scorecardpipeline 风格（从 X 中推导标签）
        if y_train is None:
            y_train_s, y_dict_train = self._build_y(X_train_df, self._target_cfg)
            if y_dict_train and not self._label_names:
                self._label_names = list(y_dict_train.keys())
        else:
            y_train_s = _ensure_series(y_train, name=self._target_name)
            y_dict_train = None

        self._add_dataset("train", "训练集", X_train_df, y_train_s, y_dict_train)
        self._datasets_info["train"] = "训练集"

        if X_test is not None:
            X_test_df = _ensure_dataframe(X_test, feature_names=list(X_train_df.columns))
            if y_test is None:
                y_test_s, y_dict_test = self._build_y(X_test_df, self._target_cfg)
                if y_dict_test and not self._label_names:
                    self._label_names = list(y_dict_test.keys())
            else:
                y_test_s = _ensure_series(y_test, name=self._target_name)
                y_dict_test = None
            self._add_dataset("test", "测试集", X_test_df, y_test_s, y_dict_test)
            self._datasets_info["test"] = "测试集"

    # ---------- 数据集管理 ----------

    def _add_dataset(
        self,
        key: str,
        label: str,
        X: pd.DataFrame,
        y: pd.Series,
        y_dict: Optional[Dict[str, np.ndarray]] = None,
    ):
        # 获取模型实际需要的特征列表，过滤掉额外列，避免预测时报错
        # 优先级：ScoreCard.feature_names_ > sklearn.feature_names_in_ > None
        required_features: Optional[List[str]] = None
        if hasattr(self.model, "feature_names_") and self.model.feature_names_ is not None:
            # ScoreCard 等模型：使用 fit 后确定的入模特征名
            required_features = list(self.model.feature_names_)
        elif hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
            # sklearn 模型
            required_features = list(self.model.feature_names_in_)

        if required_features:
            missing = set(required_features) - set(X.columns)
            if missing:
                raise ValueError(f"数据集缺少以下模型特征: {missing}")
            X_for_pred = X[required_features]
        elif self._feature_names:
            missing = set(self._feature_names) - set(X.columns)
            if missing:
                raise ValueError(f"数据集缺少以下模型特征: {missing}")
            X_for_pred = X[self._feature_names]
        else:
            X_for_pred = X

        if len(X) != len(y):
            raise ValueError(f"特征与标签样本数不一致: X={len(X)}, y={len(y)}")
        self._datasets[key] = ReportDataset(
            name=key,
            label=label,
            X=X,
            y=y,
            y_proba=_proba_pos(self.model, X_for_pred),
            score=_score_from_model(self.model, X_for_pred),
            y_dict=y_dict,
        )

    def add_dataset(self, key: str, label: str, X, y=None, feature_names: Optional[List[str]] = None):
        """添加额外数据集（如 OOT）用于报告.

        :param key: 数据集标识
        :param label: 数据集标签
        :param X: DataFrame（含目标列时 y 可为 None，自动构建标签）
        :param y: 标签列，None 时从 X 中通过 target / overdue+dpds 自动构建
        :param feature_names: 特征名列表
        """
        X = _ensure_dataframe(X, feature_names=feature_names or self.feature_names)
        # y=None 时从 X 中通过 overdue+dpds 自动构建标签（scorecardpipeline 风格）
        if y is None:
            y, y_dict = self._build_y(X, self._target_cfg)
            if y_dict and not self._label_names:
                self._label_names = list(y_dict.keys())
        else:
            y_dict = None
        y = _ensure_series(y, name=self._target_name)
        self._add_dataset(key, label, X, y, y_dict)

    def _is_multi_label(self) -> bool:
        """是否多标签模式."""
        return bool(self._label_names)

    def _get_y(self, dataset_key: str, label: Optional[str] = None) -> np.ndarray:
        """获取指定数据集的 y 数组.

        :param dataset_key: 数据集标识
        :param label: 标签名，None 时返回 combined y
        """
        ds = self._datasets[dataset_key]
        if label and ds.y_dict and label in ds.y_dict:
            return ds.y_dict[label]
        return ds.y.to_numpy()

    def _is_overdue_cfg(self) -> bool:
        """目标配置是否为 overdue + dpds 逾期联合标签模式."""
        return isinstance(self._target_cfg, dict) and "overdue" in self._target_cfg

    def _overdue_dpds(self) -> Tuple[List[str], List[Union[int, float]]]:
        """从 target 配置解析逾期列与逾期天数阈值列表（与 ``_build_y`` 保持一致）。

        返回 (overdue_cols, dpds_thresholds)，可直接透传给 ``feature_bin_stats``。
        """
        cfg = self._target_cfg or {}
        overdue = cfg.get("overdue")
        overdue_cols = [overdue] if isinstance(overdue, str) else list(overdue or [])
        dpds_vals = cfg.get("dpds")
        threshold = cfg.get("threshold")
        if threshold is not None:
            # 旧格式：dpds 为列名，threshold 为阈值，实际阈值列为 dpds 列
            col = dpds_vals if isinstance(dpds_vals, str) else None
            return ([col] if col else overdue_cols), [threshold]
        if dpds_vals is None:
            return overdue_cols, [0]
        if isinstance(dpds_vals, (int, float)):
            dpds_vals = [dpds_vals]
        return overdue_cols, list(dpds_vals)

    def _overdue_label_map(self, separator: str = ">") -> Dict[str, str]:
        """返回内部标签到报告展示标签的映射。"""
        overdue_cols, dpds_vals = self._overdue_dpds()
        display = [f"{col}{separator}{dpd}" for col in overdue_cols for dpd in dpds_vals]
        return dict(zip(self._label_names, display))

    def _normalize_overdue_bin_columns(self, table: pd.DataFrame) -> pd.DataFrame:
        """将 ``feature_bin_stats`` 的原生逾期标签统一为报告标签。"""
        if not isinstance(table.columns, pd.MultiIndex):
            return table
        overdue_cols, dpds_vals = self._overdue_dpds()
        native_labels = [f"{col}_{dpd}+" for col in overdue_cols for dpd in dpds_vals]
        rename_map = dict(zip(native_labels, self._label_names))
        renamed = table.copy()
        renamed.columns = pd.MultiIndex.from_tuples(
            [(rename_map.get(col[0], col[0]), *col[1:]) for col in table.columns],
            names=table.columns.names,
        )
        return renamed

    # ---------- 1. 模型性能指标 ----------

    def get_metrics(self, label: Optional[str] = None) -> pd.DataFrame:
        """KS / AUC / PSI 等核心指标.

        :param label: 多标签模式下指定标签名，None 时使用 combined y
        """
        from ..core.metrics import ks, auc, psi

        ordered_keys = ["train", "test"] + [k for k in self._datasets if k not in ("train", "test")]
        ds_keys = [k for k in ordered_keys if k in self._datasets]
        labels_map = {k: self._datasets[k].label for k in ds_keys}

        if self._is_multi_label() and label:
            # 多标签模式：每列对应一个数据集，多行对应 KS/AUC/样本数/坏样本率
            rows = []
            rows.append(
                {
                    "统计项": "KS",
                    **{
                        labels_map[k]: _safe_binary_metric(ks, self._get_y(k, label), self._datasets[k].y_proba)
                        for k in ds_keys
                    },
                }
            )
            rows.append(
                {
                    "统计项": "AUC",
                    **{
                        labels_map[k]: _safe_binary_metric(auc, self._get_y(k, label), self._datasets[k].y_proba)
                        for k in ds_keys
                    },
                }
            )
            rows.append({"统计项": "样本数", **{labels_map[k]: len(self._get_y(k, label)) for k in ds_keys}})
            rows.append({"统计项": "坏样本率", **{labels_map[k]: float(self._get_y(k, label).mean()) for k in ds_keys}})
            return pd.DataFrame(rows)

        # 单标签模式或 combined y
        rows = []
        rows.append(
            {
                "统计项": "KS",
                **{
                    labels_map[k]: _safe_binary_metric(ks, self._datasets[k].y, self._datasets[k].y_proba)
                    for k in ds_keys
                },
            }
        )
        rows.append(
            {
                "统计项": "AUC",
                **{
                    labels_map[k]: _safe_binary_metric(auc, self._datasets[k].y, self._datasets[k].y_proba)
                    for k in ds_keys
                },
            }
        )
        rows.append({"统计项": "样本数", **{labels_map[k]: len(self._datasets[k].y) for k in ds_keys}})
        rows.append({"统计项": "坏样本率", **{labels_map[k]: float(self._datasets[k].y.mean()) for k in ds_keys}})
        if len(ds_keys) >= 2:
            psi_row: Dict[str, Any] = {"统计项": "PSI", labels_map[ds_keys[0]]: "\\"}
            for k in ds_keys[1:]:
                try:
                    psi_row[labels_map[k]] = psi(self._datasets[ds_keys[0]].score, self._datasets[k].score)
                except Exception:
                    psi_row[labels_map[k]] = np.nan
            rows.append(psi_row)

        return pd.DataFrame(rows)

    # ---------- 2. 评分分箱效果表 ----------

    def get_bin_table(
        self,
        dataset: str = "train",
        method: str = "quantile",
        max_n_bins: int = 10,
        amount_col: Optional[str] = None,
        margins: bool = True,
        label: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """使用 feature_bin_stats 生成评分分箱效果表。

        :param label: 多标签模式下指定单个标签名
        :param labels: 多标签合并模式，传入标签名列表，返回 MultiIndex 列合并表
        """
        from .feature_analyzer import feature_bin_stats

        ds = self._datasets[dataset]
        score_col = "__score__"
        df = ds.X.copy()
        df[score_col] = ds.score

        score_return_cols = [
            "样本总数",
            "好样本数",
            "坏样本数",
            "样本占比",
            "好样本占比",
            "坏样本占比",
            "坏样本率",
            "LIFT值",
            "累积LIFT值",
            "坏账改善",
            "累积坏账改善",
            "分档KS值",
        ]

        # 多标签合并模式：直接通过 feature_bin_stats 的 overdue + dpds 计算多目标合并分箱表，
        # 由其原生输出多级表头（公共列归入「分箱详情」），并去除「指标名称 / 指标含义」列。
        if labels and self._is_multi_label() and self._is_overdue_cfg():
            overdue_cols, dpds_vals = self._overdue_dpds()
            kw: Dict[str, Any] = dict(
                feature=score_col,
                overdue=overdue_cols,
                dpds=dpds_vals,
                method=method,
                desc="模型评分",
                max_n_bins=max_n_bins,
                missing_separate=True,
                margins=margins,
                return_cols=score_return_cols,
            )
            if amount_col and amount_col in df.columns:
                kw["amount"] = amount_col
            table = feature_bin_stats(df, **kw)
            if isinstance(table, tuple):
                table = table[0]
            return _drop_bin_meta_cols(self._normalize_overdue_bin_columns(table))

        # 其他多标签合并模式（非 overdue 配置）：逐标签计算后合并
        if labels and self._is_multi_label():
            tables: List[pd.DataFrame] = []
            for lbl in labels:
                t = self.get_bin_table(
                    dataset=dataset,
                    method=method,
                    max_n_bins=max_n_bins,
                    amount_col=amount_col,
                    margins=margins,
                    label=lbl,
                )
                tables.append(t)
            return _drop_bin_meta_cols(_merge_multi_label_bin_tables(tables, labels))

        target_col = "__target__"
        df[target_col] = self._get_y(dataset, label)

        kw = dict(
            feature=score_col,
            target=target_col,
            method=method,
            desc="模型评分",
            max_n_bins=max_n_bins,
            missing_separate=True,
            margins=margins,
            return_cols=score_return_cols,
        )
        if amount_col and amount_col in df.columns:
            kw["amount"] = amount_col

        table = feature_bin_stats(df, **kw)
        if isinstance(table, tuple):
            table = table[0]
        return _drop_bin_meta_cols(table)

    # ---------- 3. 特征重要性 ----------

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        if self._importance_cache is None:
            from ..core.metrics import ks, iv, psi

            importances = None
            if hasattr(self.model, "get_feature_importances"):
                try:
                    importances = self.model.get_feature_importances()
                except Exception:
                    pass
            if importances is None and hasattr(self.model, "feature_importances_"):
                values = np.asarray(self.model.feature_importances_).reshape(-1)
                if len(values) == len(self.feature_names):
                    importances = pd.Series(values, index=self.feature_names)

            if importances is not None and not isinstance(importances, pd.Series):
                values = np.asarray(importances).reshape(-1)
                if len(values) == len(self.feature_names):
                    importances = pd.Series(values, index=self.feature_names)
                else:
                    importances = None

            if importances is None:
                self._importance_cache = pd.DataFrame(columns=["特征重要性", "IV", "KS", "PSI"])
            else:
                importances = importances.reindex([f for f in importances.index if f in self.feature_names])
                total = importances.abs().sum()
                importance_df = pd.DataFrame(index=importances.index)
                importance_df["特征重要性"] = importances.abs().values / total if total else importances.values

                train_ds = self._datasets["train"]
                y_arr = train_ds.y.to_numpy()

                iv_vals, ks_vals, psi_vals = [], [], []
                for feat in importance_df.index:
                    col = train_ds.X[feat] if feat in train_ds.X.columns else None
                    if col is not None:
                        try:
                            iv_vals.append(iv(y_arr, col))
                        except Exception:
                            iv_vals.append(np.nan)
                        try:
                            ks_vals.append(ks(y_arr, col))
                        except Exception:
                            ks_vals.append(np.nan)
                        if "test" in self._datasets and feat in self._datasets["test"].X.columns:
                            try:
                                psi_vals.append(psi(col, self._datasets["test"].X[feat]))
                            except Exception:
                                psi_vals.append(np.nan)
                        else:
                            psi_vals.append(np.nan)
                    else:
                        iv_vals.append(np.nan)
                        ks_vals.append(np.nan)
                        psi_vals.append(np.nan)

                importance_df["IV"] = iv_vals
                importance_df["KS"] = ks_vals
                importance_df["PSI"] = psi_vals
                self._importance_cache = importance_df.sort_values("特征重要性", ascending=False)

        df = self._importance_cache.copy()
        if top_n is not None:
            df = df.head(top_n)
        return df

    # ---------- 4. 特征描述 ----------

    def get_features_describe(self) -> pd.DataFrame:
        """入模变量重要性及描述性统计."""
        if self._features_describe_cache is not None:
            return self._features_describe_cache.copy()

        importance = self.get_feature_importance()
        features = importance.index.tolist()
        train_X = self._datasets["train"].X[features] if features else self._datasets["train"].X[self.feature_names]
        desc_stats = train_X.describe(percentiles=[0.01, 0.1, 0.5, 0.75, 0.9, 0.99]).T
        desc_stats = desc_stats.rename(
            columns={
                "count": "样本数",
                "mean": "平均值",
                "std": "标准差",
                "min": "最小值",
                "max": "最大值",
                "1%": "1%",
                "10%": "10%",
                "50%": "50%",
                "75%": "75%",
                "90%": "90%",
                "99%": "99%",
            }
        )
        desc_stats["缺失率"] = train_X.isnull().mean()
        desc_stats["字段类型"] = train_X.dtypes.astype(str)
        desc_stats["枚举数"] = train_X.nunique()

        keep_cols = [
            "字段类型",
            "缺失率",
            "枚举数",
            "平均值",
            "标准差",
            "最小值",
            "1%",
            "10%",
            "50%",
            "75%",
            "90%",
            "99%",
            "最大值",
        ]
        keep_cols = [c for c in keep_cols if c in desc_stats.columns]
        result = importance.join(desc_stats[keep_cols], how="left")
        result = result.drop(columns=["样本数"], errors="ignore")
        self._features_describe_cache = result
        return self._features_describe_cache.copy()

    # ---------- 5. 特征相关性 ----------

    def get_features_corr(self) -> pd.DataFrame:
        importance = self.get_feature_importance()
        features = importance.index.tolist()
        if not features:
            features = self.feature_names
        return self._datasets["train"].X[features].corr()

    # ---------- 6. 特征分箱分析 ----------

    def get_feature_bin_table(
        self,
        feature: str,
        dataset: str = "train",
        max_n_bins: int = 10,
        method: str = "quantile",
        margins: bool = True,
        amount_col: Optional[str] = None,
        label: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """单特征分箱效果表，优先使用模型 binner。

        :param label: 多标签模式下指定单个标签名
        :param labels: 多标签合并模式，传入标签名列表，返回 MultiIndex 列合并表
        """
        from .feature_analyzer import feature_bin_stats

        ds = self._datasets[dataset]
        df = ds.X.copy()
        binner = getattr(self.model, "binner", None)

        # 多标签合并模式：通过 feature_bin_stats 的 overdue + dpds 计算多目标合并分箱表，
        # 多级表头由其原生输出（公共列归入「分箱详情」），并去除「指标名称 / 指标含义」列。
        if labels and self._is_multi_label() and self._is_overdue_cfg():
            overdue_cols, dpds_vals = self._overdue_dpds()
            kw: Dict[str, Any] = dict(
                feature=feature,
                overdue=overdue_cols,
                dpds=dpds_vals,
                method=method,
                max_n_bins=max_n_bins,
                margins=margins,
                missing_separate=True,
            )
            if binner is not None:
                kw["binner"] = binner
            if amount_col and amount_col in df.columns:
                kw["amount"] = amount_col
            table = feature_bin_stats(df, **kw)
            if isinstance(table, tuple):
                table = table[0]
            return _drop_bin_meta_cols(self._normalize_overdue_bin_columns(table))

        # 其他多标签合并模式（非 overdue 配置）：逐标签计算后合并
        if labels and self._is_multi_label():
            tables: List[pd.DataFrame] = []
            for lbl in labels:
                t = self.get_feature_bin_table(
                    feature=feature,
                    dataset=dataset,
                    max_n_bins=max_n_bins,
                    method=method,
                    margins=margins,
                    amount_col=amount_col,
                    label=lbl,
                )
                tables.append(t)
            return _drop_bin_meta_cols(_merge_multi_label_bin_tables(tables, labels))

        target_col = "__target__"
        df[target_col] = self._get_y(dataset, label)

        kw = dict(
            feature=feature,
            target=target_col,
            method=method,
            max_n_bins=max_n_bins,
            margins=margins,
            missing_separate=True,
        )
        if binner is not None:
            kw["binner"] = binner

        if amount_col and amount_col in df.columns:
            kw["amount"] = amount_col

        table = feature_bin_stats(df, **kw)
        if isinstance(table, tuple):
            table = table[0]
        return _drop_bin_meta_cols(table)

    # ---------- 8. 图表导出 ----------

    def _get_top_n_lift_table(
        self,
        percentiles: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
        amount_col: Optional[str] = None,
        label: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """构建 TOP n% 尾部区分能力表。

        :param percentiles: TOP n% 的百分位列表
        :param amount_col: 金额字段名（可选），指定时输出金额口径
        :param label: 单标签模式下指定标签名，None 时使用 combined y
        :param labels: 多标签合并模式，返回 MultiIndex 列合并表
        """
        # 多标签合并模式：labels 列表优先，返回 MultiIndex 列结构
        if labels and self._is_multi_label():
            return self._get_top_n_lift_table_multi(
                percentiles=percentiles,
                amount_col=amount_col,
                labels=labels,
            )

        rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            y_arr = self._get_y(ds_key, label)
            n = len(y_arr)
            overall_bad_rate = float(y_arr.mean())

            sorted_idx = np.argsort(-ds.y_proba)
            sorted_y = y_arr[sorted_idx]

            bad_rates: Dict[str, float] = {}
            lifts: Dict[str, float] = {}
            improvements: Dict[str, float] = {}

            for pct in percentiles:
                top_n = max(1, int(n * pct))
                top_bad_rate = float(sorted_y[:top_n].mean())
                lift = top_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0
                improvement = (top_bad_rate - overall_bad_rate) / overall_bad_rate if overall_bad_rate > 0 else 0.0
                key = f"TOP {int(pct * 100)}%"
                bad_rates[key] = top_bad_rate
                lifts[key] = lift
                improvements[key] = improvement

            bad_rates["TOTAL"] = overall_bad_rate
            lifts["TOTAL"] = 1.0
            improvements["TOTAL"] = 0.0

            rows.append({"数据集": tag, "统计项": "坏样本率", **bad_rates})
            rows.append({"数据集": tag, "统计项": "LIFT值", **lifts})
            rows.append({"数据集": tag, "统计项": "坏账改善", **improvements})

            # 金额口径替代订单口径；调用方会将两个结果并排展示。
            if amount_col and amount_col in ds.X.columns:
                amounts = pd.to_numeric(ds.X[amount_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float)
                amounts_sorted = amounts[sorted_idx]
                overall_bad_amount = (
                    float((sorted_y * amounts_sorted).sum() / amounts_sorted.sum())
                    if amounts_sorted.sum() > 0
                    else overall_bad_rate
                )

                amt_bad_rates: Dict[str, float] = {}
                amt_lifts: Dict[str, float] = {}
                amt_improvements: Dict[str, float] = {}

                for pct in percentiles:
                    top_n = max(1, int(n * pct))
                    top_amt = amounts_sorted[:top_n]
                    top_y_sorted = sorted_y[:top_n]
                    top_bad_amt = float((top_y_sorted * top_amt).sum() / top_amt.sum()) if top_amt.sum() > 0 else 0.0
                    lift_amt = top_bad_amt / overall_bad_amount if overall_bad_amount > 0 else 0.0
                    imp_amt = (top_bad_amt - overall_bad_amount) / overall_bad_amount if overall_bad_amount > 0 else 0.0
                    key = f"TOP {int(pct * 100)}%"
                    amt_bad_rates[key] = top_bad_amt
                    amt_lifts[key] = lift_amt
                    amt_improvements[key] = imp_amt

                amt_bad_rates["TOTAL"] = overall_bad_amount
                amt_lifts["TOTAL"] = 1.0
                amt_improvements["TOTAL"] = 0.0

                rows[-3:] = [
                    {"数据集": tag, "统计项": "坏样本率", **amt_bad_rates},
                    {"数据集": tag, "统计项": "LIFT值", **amt_lifts},
                    {"数据集": tag, "统计项": "坏账改善", **amt_improvements},
                ]

        return pd.DataFrame(rows)

    def _get_top_n_lift_table_multi(
        self,
        percentiles: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
        amount_col: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """多标签合并模式的 LIFT 表。

        行: TOP 1% / TOP 3% / TOP 5% / TOP 10% / TOTAL
        列: (数据集, 标签, 指标) 三层 MultiIndex
        """
        if labels is None:
            labels = self._label_names

        # 逐标签复用单标签 TOP n% 表，再沿列方向以「标签」为外层拼接，
        # 形成 行=(数据集, 统计项)、列=(标签, TOP n%) 的二级表头结构。
        per_label: Dict[str, pd.DataFrame] = {}
        for lbl in labels:
            sub = self._get_top_n_lift_table(
                percentiles=percentiles,
                amount_col=amount_col,
                label=lbl,
            )
            per_label[lbl] = sub.set_index(["数据集", "统计项"])

        result_df = pd.concat(per_label, axis=1)
        result_df.columns = pd.MultiIndex.from_tuples(list(result_df.columns), names=["标签", "统计指标"])
        result_df.index = result_df.index.set_names(["数据集", "统计项"])
        return result_df

    def _get_features_summary(self) -> pd.DataFrame:
        """使用 pd.DataFrame.summary() 获取入模变量综合统计."""
        importance = self.get_feature_importance()
        features = importance.index.tolist() if not importance.empty else self.feature_names

        target_col = self._target_name or "target"
        train_df = self._datasets["train"].X[features].copy()
        train_df[target_col] = self._datasets["train"].y.values

        test_df = None
        if "test" in self._datasets:
            test_df = self._datasets["test"].X[features].copy()
            test_df[target_col] = self._datasets["test"].y.values

        try:
            summary_result = train_df.summary(
                features=features,
                y=target_col,
                val_df=test_df,
            )
            return summary_result
        except Exception:
            return self.get_features_describe()

    # 分月评分分布分位数（与特征效率分析保持一致的分位数口径）
    _SCORE_DIST_QUANTILES = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]

    def _get_monthly_metrics(self, date_col: str) -> pd.DataFrame:
        """分月计算 KS/AUC 及评分分布（均值/标准差/极值/分位数）."""
        from ..core.metrics import ks, auc

        q_cols = [f"{int(round(q * 100))}%" for q in self._SCORE_DIST_QUANTILES]

        rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            if date_col not in ds.X.columns:
                continue
            dates = pd.to_datetime(ds.X[date_col])
            months = dates.dt.to_period("M")
            for month in sorted(months.unique()):
                mask = months == month
                y_m = ds.y[mask.values]
                proba_m = ds.y_proba[mask.values]
                score_m = np.asarray(ds.score)[mask.values]
                if len(y_m) == 0:
                    continue
                try:
                    row: Dict[str, Any] = {
                        "数据集": ds.label,
                        "月份": str(month),
                        "样本数": len(y_m),
                        "坏样本率": float(y_m.mean()),
                        "KS": _safe_binary_metric(ks, y_m, proba_m),
                        "AUC": _safe_binary_metric(auc, y_m, proba_m),
                        "均值": float(np.nanmean(score_m)),
                        "标准差": float(np.nanstd(score_m)),
                        "最小值": float(np.nanmin(score_m)),
                        "最大值": float(np.nanmax(score_m)),
                    }
                    q_values = np.nanpercentile(score_m, [q * 100 for q in self._SCORE_DIST_QUANTILES])
                    for q_col, q_val in zip(q_cols, q_values):
                        row[q_col] = float(q_val)
                    rows.append(row)
                except Exception as exc:
                    logger.warning("生成 %s %s 的分月模型效果失败: %s", ds.label, month, exc)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _get_monthly_psi_matrix(self, date_col: str) -> pd.DataFrame:
        """分月 PSI 交叉矩阵."""
        from ..core.metrics import psi

        month_scores: Dict[str, np.ndarray] = {}
        for ds in self._datasets.values():
            if date_col not in ds.X.columns:
                continue
            dates = pd.to_datetime(ds.X[date_col])
            months = dates.dt.to_period("M")
            for month in sorted(months.unique()):
                mask = months == month
                key = str(month)
                if key in month_scores:
                    month_scores[key] = np.concatenate([month_scores[key], ds.score[mask.values]])
                else:
                    month_scores[key] = ds.score[mask.values]

        if len(month_scores) < 2:
            return pd.DataFrame()

        labels = sorted(month_scores.keys())
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        for i, m1 in enumerate(labels):
            for j, m2 in enumerate(labels):
                try:
                    matrix.loc[m1, m2] = psi(month_scores[m1], month_scores[m2])
                except Exception:
                    pass
        return matrix

    # ---------- 8. 图表导出 ----------

    def _export_plots(
        self,
        output_dir: Path,
        n_bins: int = 10,
        bin_method: str = "quantile",
        amount_col: Optional[str] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, pd.DataFrame]]:
        """导出所有图表，返回 (图表路径字典, PSI数据表字典)."""
        from ..core.viz import ks_plot, bin_plot, corr_plot, psi_plot, lift_plot

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, List[str]] = {}
        tables: Dict[str, pd.DataFrame] = {}

        # --- 模型级图表（用于模型性能 Sheet） ---
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            model_figs: List[str] = []

            try:
                bt = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, margins=True)
                bd = bt.iloc[:-1].reset_index(drop=True) if len(bt) > 1 else bt
                p = str(output_dir / f"bin_{ds_key}.png")
                bin_plot(bd, desc="模型评分", ending=f" {tag}", save=p, figsize=(12, 7))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            try:
                p = str(output_dir / f"ks_{ds_key}.png")
                ks_plot(ds.score, ds.y, title=f"{tag} KS曲线", save=p, figsize=(12, 7))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            try:
                p = str(output_dir / f"lift_{ds_key}.png")
                lift_plot(ds.y, ds.y_proba, n_bins=20, title=f"{tag} LIFT曲线", save=p, figsize=(12, 7))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            if model_figs:
                paths[f"model_{ds_key}"] = model_figs

        # --- 特征相关性图 ---
        importance = self.get_feature_importance()
        top_features = importance.index.tolist()
        if len(top_features) >= 2:
            try:
                p = str(output_dir / "feature_corr.png")
                corr_plot(self._datasets["train"].X[top_features], annot=False, save=p)
                _safe_close_figs()
                paths["feature_corr"] = [p]
            except Exception:
                pass

        # --- 逐特征图表（分箱图、分布图、PSI图） ---
        ds_keys = list(self._datasets.keys())
        for feat in top_features or self.feature_names:
            # 分箱图：按 train/test 顺序分组
            bin_figs: List[str] = []
            for ds_key, ds in self._datasets.items():
                try:
                    ft = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, margins=True)
                    fd = ft.iloc[:-1].reset_index(drop=True) if len(ft) > 1 else ft
                    p = str(output_dir / f"bin_{feat}_{ds_key}.png")
                    bin_plot(fd, desc=feat, ending=f" {ds.label}", save=p, figsize=(12, 7))
                    _safe_close_figs()
                    bin_figs.append(p)
                except Exception as exc:
                    logger.warning("生成特征 %s 的 %s 分箱图失败: %s", feat, ds.label, exc)
            if bin_figs:
                paths[f"feat_bin_{feat}"] = bin_figs

            # 特征KS分布图（替换直方图，显示特征对好坏样本的区分能力）
            # 处理缺失值和类别特征
            ks_figs: List[str] = []
            for ds_key, ds in self._datasets.items():
                try:
                    col_raw = ds.X[feat]
                    col = col_raw.dropna()
                    # 检查是否为类别特征或低基数的数值特征
                    is_categorical = col.dtype == "object" or (
                        col.dtype in ["int64", "float64"] and col.nunique() <= 10
                    )
                    if is_categorical:
                        # 类别特征跳过KS图
                        continue
                    y_f = ds.y.loc[col.index]
                    # 确保标签是二分类
                    if y_f.nunique() < 2:
                        continue
                    p = str(output_dir / f"ks_{feat}_{ds_key}.png")
                    ks_plot(col, y_f, title=f"{ds.label} {feat}", save=p, figsize=(12, 7))
                    _safe_close_figs()
                    ks_figs.append(p)
                except Exception:
                    pass
            if ks_figs:
                paths[f"feat_hist_{feat}"] = ks_figs

            # PSI 图（训练集 vs 第一个非训练集），传入 y 以便图与表均包含坏样本率信息
            if len(ds_keys) >= 2:
                try:
                    train_ds = self._datasets[ds_keys[0]]
                    test_ds = self._datasets[ds_keys[1]]
                    train_mask = train_ds.X[feat].notna()
                    test_mask = test_ds.X[feat].notna()
                    train_vals = train_ds.X[feat][train_mask]
                    test_vals = test_ds.X[feat][test_mask]
                    psi_y = np.concatenate(
                        [
                            train_ds.y.to_numpy()[train_mask.to_numpy()],
                            test_ds.y.to_numpy()[test_mask.to_numpy()],
                        ]
                    )
                    p = str(output_dir / f"psi_{feat}.png")
                    psi_result = psi_plot(
                        train_vals,
                        test_vals,
                        y=psi_y,
                        desc=feat,
                        save=p,
                        result=True,
                        plot=True,
                        figsize=(15, 8),
                    )
                    _safe_close_figs()
                    paths[f"feat_psi_{feat}"] = [p]
                    if isinstance(psi_result, pd.DataFrame):
                        tables[f"feat_psi_{feat}"] = psi_result
                except Exception as exc:
                    logger.warning("生成特征 %s 的 PSI 图表失败: %s", feat, exc)

        # --- 评分卡专属图表 ---
        if hasattr(self.model, "lr_model"):
            try:
                from ..core.viz import plot_weights as _pw

                p = str(output_dir / "plot_weights.png")
                _pw(self.model.lr_model, save=p)
                _safe_close_figs()
                paths["model_weights"] = [p]
            except Exception:
                pass

            if len(ds_keys) >= 2:
                try:
                    train_ds = self._datasets[ds_keys[0]]
                    test_ds = self._datasets[ds_keys[1]]
                    score_train = pd.Series(train_ds.score)
                    score_test = pd.Series(test_ds.score)
                    train_mask = score_train.notna()
                    test_mask = score_test.notna()
                    score_train = score_train[train_mask]
                    score_test = score_test[test_mask]
                    score_y = np.concatenate(
                        [
                            train_ds.y.to_numpy()[train_mask.to_numpy()],
                            test_ds.y.to_numpy()[test_mask.to_numpy()],
                        ]
                    )
                    p = str(output_dir / "score_psi.png")
                    score_psi_df = psi_plot(
                        score_train,
                        score_test,
                        y=score_y,
                        desc="模型评分",
                        save=p,
                        result=True,
                        plot=True,
                        figsize=(15, 8),
                    )
                    _safe_close_figs()
                    paths["score_psi"] = [p]
                    if isinstance(score_psi_df, pd.DataFrame):
                        tables["score_psi"] = score_psi_df
                except Exception as exc:
                    logger.warning("生成模型评分 PSI 图表失败: %s", exc)

        return paths, tables

    # ---------- 9. 模型摘要 ----------

    # 模型摘要默认展示的指标（每个指标横向展开到各数据集）
    _SUMMARY_METRICS = ["KS", "AUC", "样本数", "坏样本率"]

    def summary(self) -> pd.DataFrame:
        """模型核心指标摘要表（多层列：统计指标 × 数据集；行：逾期指标）.

        参考 :mod:`hscredit.report.rule_strategy` 中拒绝规则策略表的展示方式：
        列为「统计指标 × 数据集」两层表头，便于同一指标在各数据集上横向对比；
        行为不同逾期指标。``overdue`` + ``dpds`` 多标签模式下每个逾期标签独占一行，
        单标签模式下仅一行（行名为目标列名）。

        :return: 行索引为 ``逾期指标``、列为 ``(统计指标, 数据集)`` 两层表头的 DataFrame

        **参考样例**

        >>> report = ModelReport(model, datasets={'train': tr, 'test': te},
        ...                           overdue=['MOB1'], dpds=[7, 3, 0])
        >>> report.summary()  # 行: MOB1@7 / MOB1@3 / MOB1@0；列: (KS, 训练集) ...
        """
        from ..core.metrics import ks, auc

        ordered_keys = ["train", "test"] + [k for k in self._datasets if k not in ("train", "test")]
        ds_keys = [k for k in ordered_keys if k in self._datasets]
        ds_labels = [self._datasets[k].label for k in ds_keys]

        if self._is_multi_label():
            targets = list(self._label_names)
            display_map = self._overdue_label_map(separator="@") if self._is_overdue_cfg() else {}
            index_labels = [display_map.get(t, t) for t in targets]
        else:
            targets = [None]
            index_labels = [self._target_name or "target"]

        rows: List[Dict[tuple, Any]] = []
        for tgt in targets:
            row: Dict[tuple, Any] = {}
            for ds_key, ds_label in zip(ds_keys, ds_labels):
                y_arr = self._get_y(ds_key, tgt)
                proba = self._datasets[ds_key].y_proba
                row[("KS", ds_label)] = _safe_binary_metric(ks, y_arr, proba)
                row[("AUC", ds_label)] = _safe_binary_metric(auc, y_arr, proba)
                row[("样本数", ds_label)] = len(y_arr)
                row[("坏样本率", ds_label)] = float(y_arr.mean()) if len(y_arr) else np.nan
            rows.append(row)

        columns = pd.MultiIndex.from_tuples(
            [(metric, ds_label) for metric in self._SUMMARY_METRICS for ds_label in ds_labels],
            names=["统计指标", "数据集"],
        )
        index = pd.Index(index_labels, name="逾期指标")
        return pd.DataFrame(rows, index=index, columns=columns)

    # ---------- 10. 控制台输出 ----------

    def print_report(self, n_bins: int = 10, **kwargs) -> None:
        is_multi = self._is_multi_label()
        labels_arg = self._label_names if is_multi else None

        print("=" * 72)
        print("模型评估快速报告")
        print("=" * 72)
        print("\n【模型性能指标】")
        if is_multi:
            # 多标签模式：逐逾期标签展示「统计指标 × 数据集」摘要表
            print(self.summary().to_string())
        else:
            print(self.get_metrics().to_string(index=False))

        importance = self.get_feature_importance(top_n=10)
        if not importance.empty:
            print("\n【Top 10 特征重要性】")
            print(importance.to_string())

        for ds_key, ds in self._datasets.items():
            print(f"\n【{ds.label}评分分箱效果】")
            print(self.get_bin_table(ds_key, max_n_bins=n_bins, labels=labels_arg).to_string(index=False))
        print("\n" + "=" * 72)

    # ---------- 11. to_excel ----------

    def to_excel(
        self,
        filepath: str,
        *,
        n_bins: int = 10,
        bin_method: str = "quantile",
        amount_col: Optional[str] = None,
        date_col: Optional[str] = None,
        date_freq: Optional[str] = None,
        group_col: Optional[str] = None,
        with_plots: bool = True,
        model_name: Optional[str] = None,
        project_desc: Optional[str] = None,
        feature_map: Optional[Dict[str, str]] = None,
        feature_info: Optional[pd.DataFrame] = None,
        data_source: Optional[str] = None,
        loc_cols: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """生成多 Sheet 结构的 Excel 模型报告.

        Sheet 结构：
        - 目录
        - 1-基本信息（项目目标、样本统计、分月/分组分布）
        - 2-模型性能（指标、TOP n%、PSI矩阵、分箱效果）
        - 3-入模变量分析（重要性、相关性、逐特征分箱/KS/PSI）

        :param loc_cols: 定位字段（订单号等），支持 str 或 List[str]，仅用于生产订单测试用例
        """
        from ..excel import ExcelWriter, dataframe2excel

        model_name = model_name or self.model.__class__.__name__
        max_col = 35

        plot_paths: Dict[str, List[str]] = {}
        psi_tables: Dict[str, pd.DataFrame] = {}
        if with_plots:
            plot_dir = Path(filepath).parent / f"{Path(filepath).stem}_assets"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_paths, psi_tables = self._export_plots(
                    plot_dir,
                    n_bins=n_bins,
                    bin_method=bin_method,
                    amount_col=amount_col,
                )

        writer = ExcelWriter()

        # ============================================================
        # 目录 Sheet
        # ============================================================
        contents = pd.DataFrame(
            [
                {"序号": 1, "内容": "1-基本信息", "备注": "项目目标、样本选取、样本坏率分布"},
                {"序号": 2, "内容": "2-模型性能", "备注": "模型效果、区分度、稳定性等内容"},
                {"序号": 3, "内容": "3-入模变量分析", "备注": "模型变量有效性及不同数据集分箱情况"},
                {"序号": 4, "内容": "4-稳定性分析", "备注": "评分分布、PSI、CSI等稳定性分析"},
                {"序号": 5, "内容": "5-模型参数", "备注": "模型选型、参数及评分卡配置"},
                {"序号": 6, "内容": "6-模型部署需求", "备注": "入模变量信息及测试用例"},
            ]
        )

        ws = writer.get_sheet_by_name("目录")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="模型评估报告", style="header_middle", end_space=(2, max_col)
        )
        end_row, _ = dataframe2excel(contents, writer, sheet_name=ws, start_row=end_row + 1, left_cols=["内容", "备注"])

        for i, row in contents.iterrows():
            try:
                target_cell = writer.get_cell_space((2, 2))
                writer.insert_hyperlink2sheet(
                    ws, (end_row - len(contents) + i, 3), hyperlink=f"#'{row['内容']}'!{target_cell}"
                )
            except Exception:
                pass

        _, _ = writer.insert_value2sheet(
            ws, (end_row + 1, 2), value="版本号:", style="middle", end_space=(end_row + 1, 2)
        )
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 1, 3), value="V1.0", style="middle", end_space=(end_row + 1, 4)
        )
        _, _ = writer.insert_value2sheet(ws, (end_row, 2), value="创建日期:", style="middle", end_space=(end_row, 2))
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row, 3), value=date.today().strftime("%Y-%m-%d"), style="middle", end_space=(end_row, 4)
        )
        _, _ = writer.insert_value2sheet(ws, (end_row, 2), value="模型名称:", style="middle", end_space=(end_row, 2))
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row, 3), value=model_name, style="middle", end_space=(end_row, 4)
        )
        writer.adjust_columns_width(ws, start_col=2, end_col=4)

        # ============================================================
        # 1-基本信息 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("1-基本信息")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="一、基本信息", style="header_middle", end_space=(2, max_col)
        )
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 1.1 项目目标
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="1、项目目标", style="header_middle", align={"horizontal": "left"}
        )
        desc_text = project_desc or f"使用 {model_name} 模型进行信用风险评估"
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row, 2),
            value=desc_text,
            style="middle",
            end_space=(end_row, max_col),
            align={"horizontal": "left"},
        )

        # 1.2 数据样本描述
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="2、数据样本描述", style="header_middle", align={"horizontal": "left"}
        )

        label_text = self._target_name or "TARGET"

        # ---- Step 2: 合并所有数据集，计算整体各逾期标签的逾期数据 ----
        # 如果有日期字段，先提取全局日期范围
        global_date_prefix = ""
        if date_col:
            all_dates = []
            for ds in self._datasets.values():
                if ds is not None and date_col in ds.X.columns:
                    all_dates.append(pd.to_datetime(ds.X[date_col]))
            if all_dates:
                all_dates_combined = pd.concat(all_dates, ignore_index=True).dropna()
                if not all_dates_combined.empty:
                    global_date_prefix = (
                        f"{all_dates_combined.min().strftime('%Y-%m-%d')} ~ "
                        f"{all_dates_combined.max().strftime('%Y-%m-%d')}  "
                    )
        sample_interval = global_date_prefix if global_date_prefix else ""

        is_multi = self._is_multi_label()
        ds_keys_list = list(self._datasets.keys())
        dataset_labels = [self._datasets[k].label for k in ds_keys_list]

        # ---- Step 2: 整体样本描述（多标签时逐标签展示坏样本率） ----
        overall_n = sum(len(self._datasets[k].y) for k in ds_keys_list)
        if overall_n > 0:
            if is_multi:
                display_labels = self._overdue_label_map(separator="@")
                label_parts = []
                for lbl in self._label_names:
                    all_y = np.concatenate([self._get_y(k, lbl) for k in ds_keys_list])
                    label_parts.append(f"{display_labels.get(lbl, lbl)}: {round(float(all_y.mean()) * 100, 2)}%")
                overall_desc = f"样本数: {overall_n}, " + ", ".join(label_parts)
            else:
                overall_bad = int(sum(int(self._datasets[k].y.sum()) for k in ds_keys_list))
                overall_bad_rate = overall_bad / overall_n * 100
                overall_desc = f"样本数: {overall_n}, " f"{label_text}: {round(overall_bad_rate, 2)}%"
        else:
            overall_desc = "N/A"

        # ---- Step 3: 固定描述行 ----
        data_source_str = data_source if data_source else "N/A"
        fixed_rows: List[Dict[str, Any]] = [
            {"统计项": "样本区间", "统计内容": sample_interval or "N/A"},
            {"统计项": "整体样本", "统计内容": overall_desc},
            {"统计项": "模型名称", "统计内容": model_name or "N/A"},
            {"统计项": "取样逻辑", "统计内容": project_desc or "N/A"},
            {"统计项": "数据源", "统计内容": data_source_str},
        ]

        # ---- Step 4: 各数据集描述行（与整体样本同一张表，多标签逐标签展示坏样本率） ----
        ds_rows: List[Dict[str, Any]] = []
        for ds_key, ds_label in zip(ds_keys_list, dataset_labels):
            n_samples = len(self._datasets[ds_key].y)
            if is_multi:
                display_labels = self._overdue_label_map(separator="@")
                label_parts = [
                    f"{display_labels.get(lbl, lbl)}: {round(float(self._get_y(ds_key, lbl).mean()) * 100, 2)}%"
                    for lbl in self._label_names
                ]
                content = f"样本数: {n_samples}, " + ", ".join(label_parts)
            else:
                bad_rate = round(float(self._datasets[ds_key].y.mean()) * 100, 2)
                content = f"样本数: {n_samples}, {label_text}: {bad_rate}%"
            ds_rows.append({"统计项": ds_label, "统计内容": content})

        desc_df = pd.DataFrame(fixed_rows + ds_rows)
        end_row, _ = dataframe2excel(
            desc_df, writer, sheet_name=ws, start_row=end_row + 1, left_cols=["统计项", "统计内容"]
        )

        # 1.3 数据样本统计
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="3、数据样本统计", style="header_middle", align={"horizontal": "left"}
        )
        ds_keys_list = list(self._datasets.keys())
        dataset_labels = [self._datasets[k].label for k in ds_keys_list]

        y_maps = [
            {label: self._get_y(ds_key, label) for label in self._label_names}
            if is_multi else {label_text: self._get_y(ds_key)}
            for ds_key in ds_keys_list
        ]
        stat_df, stat_pct_cols = build_sample_stats_table(
            dataset_labels,
            y_maps,
            self._label_names if is_multi else [label_text],
            display_labels=self._overdue_label_map(separator="@") if is_multi else None,
            flat_total_col="样本数",
        )
        if is_multi:
            stat_start_row = end_row + 1
            end_row, _ = dataframe2excel(
                stat_df,
                writer,
                sheet_name=ws,
                start_row=stat_start_row,
                index=True,
                percent_cols=stat_pct_cols,
            )
        else:
            end_row, _ = dataframe2excel(
                stat_df, writer, sheet_name=ws, start_row=end_row + 1, percent_cols=stat_pct_cols
            )

        # 1.4 样本分布情况
        freq_label_map = {"D": "日", "W": "周", "M": "月", "Q": "季度", "Y": "年"}
        if date_col or group_col:
            end_row, _ = writer.insert_value2sheet(
                ws, (end_row + 2, 2), value="4、样本分布情况", style="header_middle", align={"horizontal": "left"}
            )

            def _write_distribution(group_of_ds, title: str):
                """构建并写入一张「数据集 × 数据分组」的分布表（数据集逐段堆叠）。

                :param group_of_ds: 函数 (ds_key, ds) -> 分组值序列(与 ds.y 等长) 或 None
                """
                distribution_dataset_labels: List[str] = []
                distribution_y_maps: List[Dict[str, Any]] = []
                group_values: List[Any] = []
                for ds_key, ds_label in zip(ds_keys_list, dataset_labels):
                    ds = self._datasets[ds_key]
                    gvals = group_of_ds(ds_key, ds)
                    if gvals is None:
                        continue
                    distribution_dataset_labels.append(ds_label)
                    group_values.append(gvals)
                    distribution_y_maps.append(
                        {label: self._get_y(ds_key, label) for label in self._label_names}
                        if is_multi else {label_text: self._get_y(ds_key)}
                    )
                if not distribution_y_maps:
                    return
                dist_df, pct = build_group_distribution_table(
                    distribution_dataset_labels,
                    distribution_y_maps,
                    group_values,
                    self._label_names if is_multi else [label_text],
                    display_labels=self._overdue_label_map(separator="@") if is_multi else None,
                )
                dist_start_row = end_row + 1
                result = dataframe2excel(
                    dist_df,
                    writer,
                    sheet_name=ws,
                    title=title,
                    start_row=end_row + 1,
                    index=not isinstance(dist_df.columns, pd.MultiIndex),
                    percent_cols=pct,
                )
                return result

            # 时间分布
            if date_col:
                freq = date_freq or "M"
                period_col_name = freq_label_map.get(freq, freq)

                def _period_of(ds_key, ds, _freq=freq):
                    if date_col not in ds.X.columns:
                        return None
                    dates = pd.to_datetime(ds.X[date_col])
                    try:
                        return dates.dt.to_period(_freq).astype(str).values
                    except Exception:
                        return dates.dt.to_period("M").astype(str).values

                ret = _write_distribution(_period_of, title=f"{period_col_name}度分布")
                if ret is not None:
                    end_row, _ = ret

            # 分组分布
            if group_col:

                def _group_of(ds_key, ds):
                    if group_col not in ds.X.columns:
                        return None
                    return ds.X[group_col].values

                ret = _write_distribution(_group_of, title=f"{group_col}分布")
                if ret is not None:
                    end_row, _ = ret

        # ============================================================
        # 2-模型性能 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("2-模型性能")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="二、模型性能评估", style="header_middle", end_space=(2, max_col)
        )
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        section_idx = 1

        # 2.1 性能指标
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value=f"{section_idx}、模型性能验证指标",
            style="header_middle",
            align={"horizontal": "left"},
        )
        if is_multi:
            # 多标签模式：列为「标签 × 数据集」二级表头，行为 KS/AUC/样本数/坏样本率。
            import itertools
            from ..core.metrics import ks, auc

            metric_items = ["KS", "AUC", "样本总数", "坏样本率"]
            col_tuples = list(itertools.product(self._label_names, dataset_labels))
            multi_cols = pd.MultiIndex.from_tuples(col_tuples, names=["统计项", "统计指标"])
            rows_list: List[Dict[tuple, Any]] = []
            for metric in metric_items:
                row: Dict[tuple, Any] = {}
                for lbl in self._label_names:
                    for ds_key, ds_label in zip(ds_keys_list, dataset_labels):
                        y_arr = self._get_y(ds_key, lbl)
                        proba = self._datasets[ds_key].y_proba
                        if metric == "样本总数":
                            val = len(y_arr)
                        elif metric == "坏样本率":
                            val = float(y_arr.mean())
                        elif metric == "KS":
                            val = _safe_binary_metric(ks, y_arr, proba)
                        else:
                            val = _safe_binary_metric(auc, y_arr, proba)
                        row[(lbl, ds_label)] = val
                rows_list.append(row)
            metrics_df = pd.DataFrame(rows_list, index=metric_items, columns=multi_cols)
            metrics_df.index.name = "统计指标"
            metrics_start_row = end_row + 1
            end_row, _ = dataframe2excel(
                metrics_df,
                writer,
                sheet_name=ws,
                start_row=metrics_start_row,
                index=True,
                percent_rows=["KS", "AUC", "坏样本率"],
                custom_rows=["样本总数"],
                custom_format="#,##0",
            )
            writer.insert_value2sheet(ws, (metrics_start_row, 2), value="统计项", style="header_middle")
        else:
            metrics = self.get_metrics().replace({"统计项": {"样本数": "样本总数"}}).set_index("统计项")
            end_row, _ = dataframe2excel(
                metrics,
                writer,
                sheet_name=ws,
                start_row=end_row + 1,
                index=True,
                percent_rows=["KS", "AUC", "坏样本率"],
                custom_rows=["样本总数"],
                custom_format="#,##0",
            )
        section_idx += 1

        # 2.2 分月模型效果
        if date_col:
            monthly_metrics = self._get_monthly_metrics(date_col)
            if not monthly_metrics.empty:
                end_row, _ = writer.insert_value2sheet(
                    ws,
                    (end_row + 2, 2),
                    value=f"{section_idx}、分月模型效果",
                    style="header_middle",
                    align={"horizontal": "left"},
                )
                end_row, _ = dataframe2excel(
                    monthly_metrics,
                    writer,
                    sheet_name=ws,
                    start_row=end_row + 1,
                    percent_cols=["坏样本率", "KS", "AUC"],
                )
                section_idx += 1

        # 2.3 模型尾部区分能力
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value=f"{section_idx}、模型尾部区分能力（TOP n%）",
            style="header_middle",
            align={"horizontal": "left"},
        )
        pct_keys = ["TOP 1%", "TOP 3%", "TOP 5%", "TOP 10%", "TOTAL"]
        if is_multi:
            # 多标签合并模式：一次性生成 MultiIndex 列的 LIFT 表
            if amount_col:
                # 订单口径 + 金额口径左右并排
                lift_table = self._get_top_n_lift_table(
                    percentiles=(0.01, 0.03, 0.05, 0.10),
                    labels=self._label_names,
                )
                lift_amt = self._get_top_n_lift_table(
                    percentiles=(0.01, 0.03, 0.05, 0.10),
                    amount_col=amount_col,
                    labels=self._label_names,
                )
                table_start = end_row + 1
                end_row1, end_col1 = dataframe2excel(
                    lift_table,
                    writer,
                    sheet_name=ws,
                    title="订单口径",
                    start_row=table_start,
                    start_col=2,
                    percent_cols=list(lift_table.columns),
                    index=True,
                )
                end_row2, _ = dataframe2excel(
                    lift_amt,
                    writer,
                    sheet_name=ws,
                    title="金额口径",
                    start_row=table_start,
                    start_col=end_col1 + 1,
                    percent_cols=list(lift_amt.columns),
                    index=True,
                )
                end_row = max(end_row1, end_row2)
                writer.insert_value2sheet(ws, (table_start + 2, 2), value="统计指标", style="header_left")
                writer.insert_value2sheet(
                    ws,
                    (table_start + 2, end_col1 + 1),
                    value="统计指标",
                    style="header_left",
                )
                try:
                    from openpyxl.utils import get_column_letter

                    # 金额口径表起始列为 end_col1 + 1，占据 索引层级 + 数据列 共 nlevels+ncols
                    # 列，故其最后一列为 (end_col1 + 1) + nlevels + ncols - 1。
                    filter_end_col = end_col1 + 1 + lift_amt.index.nlevels + len(lift_amt.columns) - 1
                    header_row = table_start + lift_table.columns.nlevels + 1
                    writer.add_auto_filter(
                        ws,
                        f"B{header_row}:{get_column_letter(filter_end_col)}{end_row - 1}",
                    )
                except Exception:
                    pass
            else:
                lift_table = self._get_top_n_lift_table(
                    percentiles=(0.01, 0.03, 0.05, 0.10),
                    labels=self._label_names,
                )
                table_start = end_row + 1
                end_row, _ = dataframe2excel(
                    lift_table,
                    writer,
                    sheet_name=ws,
                    start_row=table_start,
                    percent_cols=list(lift_table.columns),
                    index=True,
                )
                writer.insert_value2sheet(ws, (table_start, 2), value="统计指标", style="header_middle")
                try:
                    from openpyxl.utils import get_column_letter

                    filter_end_col = 2 + lift_table.index.nlevels + len(lift_table.columns) - 1
                    writer.add_auto_filter(
                        ws,
                        f"B{table_start + lift_table.columns.nlevels}:"
                        f"{get_column_letter(filter_end_col)}{end_row - 1}",
                    )
                except Exception:
                    pass
            section_idx += 1
        elif amount_col:
            lift_table = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10), amount_col=None)
            lift_amt = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10), amount_col=amount_col)
            table_start = end_row + 1
            end_row1, end_col1 = dataframe2excel(
                lift_table,
                writer,
                sheet_name=ws,
                title="订单口径",
                start_row=table_start,
                start_col=2,
                percent_cols=pct_keys,
            )
            end_row2, _ = dataframe2excel(
                lift_amt,
                writer,
                sheet_name=ws,
                title="金额口径",
                start_row=table_start,
                start_col=end_col1 + 2,
                percent_cols=pct_keys,
            )
            end_row = max(end_row1, end_row2)
            try:
                n_lift_cols = len(lift_table.columns)
                filter_end_col = end_col1 + 2 + n_lift_cols - 1
                from openpyxl.utils import get_column_letter

                writer.add_auto_filter(ws, f"B{table_start + 2}:{get_column_letter(filter_end_col)}{end_row - 1}")
            except Exception:
                pass
        else:
            lift_table = self._get_top_n_lift_table()
            table_start = end_row + 3
            end_row, _ = dataframe2excel(
                lift_table,
                writer,
                sheet_name=ws,
                start_row=table_start,
                percent_cols=pct_keys,
            )
            try:
                from openpyxl.utils import get_column_letter

                writer.add_auto_filter(
                    ws, f"B{table_start}:{get_column_letter(len(lift_table.columns) + 1)}{end_row - 1}"
                )
            except Exception:
                pass
        if not is_multi:
            section_idx += 1

        # 2.4 分月PSI矩阵
        if date_col:
            psi_matrix = self._get_monthly_psi_matrix(date_col)
            if not psi_matrix.empty:
                end_row, _ = writer.insert_value2sheet(
                    ws,
                    (end_row + 2, 2),
                    value=f"{section_idx}、分月对比PSI",
                    style="header_middle",
                    align={"horizontal": "left"},
                )
                end_row, _ = dataframe2excel(psi_matrix, writer, sheet_name=ws, start_row=end_row + 1, index=True)
                section_idx += 1

        # 2.5 各数据集评分排序性
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{section_idx}、{tag}评分排序性",
                style="header_middle",
                align={"horizontal": "left"},
            )

            figs = plot_paths.get(f"model_{ds_key}", [])

            # 先插入图表（同一行、左右排列，避免 figures 参数导致标题与分箱表之间出现图）
            img_start_row = end_row + 1
            current_col = 2
            max_img_end_row = img_start_row
            for fig in figs:
                try:
                    img_end_row, _ = writer.insert_pic2sheet(ws, fig, (img_start_row, current_col), figsize=(500, 300))
                    current_col = _next_image_col(ws, current_col, 500)
                    max_img_end_row = max(max_img_end_row, img_end_row)
                except Exception:
                    pass
            if figs:
                end_row = max_img_end_row

            if is_multi:
                # 多标签合并模式：所有标签合并为一个 MultiIndex 列的分箱表
                if amount_col:
                    order_table = self.get_bin_table(
                        ds_key,
                        method=bin_method,
                        max_n_bins=n_bins,
                        margins=True,
                        labels=self._label_names,
                    )
                    amount_table = self.get_bin_table(
                        ds_key,
                        method=bin_method,
                        max_n_bins=n_bins,
                        amount_col=amount_col,
                        margins=True,
                        labels=self._label_names,
                    )
                    # 从 MultiIndex 列中提取百分位列
                    pct_cols = [
                        c for c in order_table.columns if (c[1] if isinstance(c, tuple) else c) in self._PERCENT_COLS
                    ]
                    cond_cols = [
                        c for c in order_table.columns if (c[1] if isinstance(c, tuple) else c) in self._CONDITION_COLS
                    ]
                    amt_pct = [
                        c for c in amount_table.columns if (c[1] if isinstance(c, tuple) else c) in self._PERCENT_COLS
                    ]
                    amt_cond = [
                        c for c in amount_table.columns if (c[1] if isinstance(c, tuple) else c) in self._CONDITION_COLS
                    ]
                    order_start_row = end_row + 1
                    order_end_row, order_end_col = dataframe2excel(
                        order_table,
                        writer,
                        sheet_name=ws,
                        title=f"{tag} 订单口径",
                        start_row=order_start_row,
                        start_col=2,
                        percent_cols=pct_cols,
                        condition_cols=cond_cols,
                        condition_color="F76E6C",
                    )
                    amount_end_row, _ = dataframe2excel(
                        amount_table,
                        writer,
                        sheet_name=ws,
                        title=f"{tag} 金额口径",
                        start_row=order_start_row,
                        start_col=order_end_col + 1,
                        percent_cols=amt_pct,
                        condition_cols=amt_cond,
                        condition_color="F76E6C",
                    )
                    end_row = max(order_end_row, amount_end_row)
                else:
                    order_table = self.get_bin_table(
                        ds_key,
                        method=bin_method,
                        max_n_bins=n_bins,
                        margins=True,
                        labels=self._label_names,
                    )
                    pct_cols = [
                        c for c in order_table.columns if (c[1] if isinstance(c, tuple) else c) in self._PERCENT_COLS
                    ]
                    cond_cols = [
                        c for c in order_table.columns if (c[1] if isinstance(c, tuple) else c) in self._CONDITION_COLS
                    ]
                    end_row, _ = dataframe2excel(
                        order_table,
                        writer,
                        sheet_name=ws,
                        title=f"{tag} 评分有效性",
                        start_row=end_row + 1,
                        percent_cols=pct_cols,
                        condition_cols=cond_cols,
                        condition_color="F76E6C",
                    )
            elif amount_col:
                # 订单口径和金额口径左右并排（参考3-入模变量分析的分箱表布局）
                order_table = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, margins=True)
                order_start_row = end_row + 1
                pct_cols = [c for c in self._PERCENT_COLS if c in order_table.columns]
                cond_cols = [c for c in self._CONDITION_COLS if c in order_table.columns]
                order_end_row, order_end_col = dataframe2excel(
                    order_table,
                    writer,
                    sheet_name=ws,
                    title=f"{tag} 订单口径",
                    start_row=order_start_row,
                    start_col=2,
                    percent_cols=pct_cols,
                    condition_cols=cond_cols,
                    condition_color="F76E6C",
                )
                try:
                    amount_table = self.get_bin_table(
                        ds_key, method=bin_method, max_n_bins=n_bins, amount_col=amount_col, margins=True
                    )
                    amt_pct = [c for c in self._PERCENT_COLS if c in amount_table.columns]
                    amt_cond = [c for c in self._CONDITION_COLS if c in amount_table.columns]
                    amount_end_row, _ = dataframe2excel(
                        amount_table,
                        writer,
                        sheet_name=ws,
                        title=f"{tag} 金额口径",
                        start_row=order_start_row,
                        start_col=order_end_col + 1,
                        percent_cols=amt_pct,
                        condition_cols=amt_cond,
                        condition_color="F76E6C",
                    )
                    end_row = max(order_end_row, amount_end_row)
                except Exception:
                    end_row = order_end_row
            else:
                order_table = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, margins=True)
                pct_cols = [c for c in self._PERCENT_COLS if c in order_table.columns]
                cond_cols = [c for c in self._CONDITION_COLS if c in order_table.columns]
                end_row, _ = dataframe2excel(
                    order_table,
                    writer,
                    sheet_name=ws,
                    title=f"{tag} 评分有效性",
                    start_row=end_row + 1,
                    percent_cols=pct_cols,
                    condition_cols=cond_cols,
                    condition_color="F76E6C",
                )
            section_idx += 1

        # ============================================================
        # ============================================================
        # 3-入模变量分析 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("3-入模变量分析")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="三、入模变量分析", style="header_middle", end_space=(2, max_col)
        )
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 3.1 入模变量重要性及分布情况
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value="1、入模变量重要性及分布情况",
            style="header_middle",
            align={"horizontal": "left"},
        )
        features_summary = self._get_features_summary()
        if "特征名" not in features_summary.columns:
            index_name = features_summary.index.name or "index"
            features_summary = features_summary.reset_index().rename(columns={index_name: "特征名"})
        features_summary_start_row = end_row + 1
        end_row, _ = dataframe2excel(
            features_summary,
            writer,
            sheet_name=ws,
            start_row=features_summary_start_row,
            right_cols=[0],
        )
        feature_name_col = 2 + features_summary.columns.get_loc("特征名")
        features_summary_rows = {
            str(feat): features_summary_start_row + features_summary.columns.nlevels + position
            for position, feat in enumerate(features_summary["特征名"])
        }

        # 3.2 相关性
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="2、入模变量相关性", style="header_middle", align={"horizontal": "left"}
        )
        corr_df = self.get_features_corr()
        corr_figs = plot_paths.get("feature_corr", [])
        end_row, _ = dataframe2excel(
            corr_df,
            writer,
            sheet_name=ws,
            start_row=end_row + 1,
            percent_cols=corr_df.columns.tolist(),
            index=True,
            figures=corr_figs,
            right_cols=[0],
        )

        # 3.3 入模变量有效性分析
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="3、入模变量有效性分析", style="header_middle", align={"horizontal": "left"}
        )

        importance = self.get_feature_importance()
        feature_list = importance.index.tolist() if not importance.empty else self.feature_names
        ds_keys_list = list(self._datasets.keys())

        for i, feat in enumerate(feature_list):
            feature_title_row = end_row + 2
            end_row, _ = writer.insert_value2sheet(
                ws,
                (feature_title_row, 2),
                value=f"3.{i + 1}、{feat} 有效性分析",
                style="header_middle",
                align={"horizontal": "left"},
            )

            summary_row = features_summary_rows.get(str(feat))
            if summary_row is not None:
                try:
                    writer.insert_hyperlink2sheet(
                        ws,
                        (summary_row, feature_name_col),
                        hyperlink=f"#'{ws.title}'!B{feature_title_row}",
                    )
                    writer.insert_hyperlink2sheet(
                        ws,
                        (feature_title_row, 2),
                        hyperlink=f"#'{ws.title}'!{writer.get_cell_space((summary_row, feature_name_col))}",
                    )
                except Exception:
                    pass

            # 插入图表（同一行、左右排列，避免 figures 参数导致标题与分箱表之间出现图）
            bin_figs = plot_paths.get(f"feat_bin_{feat}", [])
            hist_figs = plot_paths.get(f"feat_hist_{feat}", [])
            all_figs = bin_figs + hist_figs
            img_start_row = end_row + 1
            current_col = 2
            max_img_end_row = img_start_row
            for fig in all_figs:
                try:
                    img_end_row, _ = writer.insert_pic2sheet(ws, fig, (img_start_row, current_col), figsize=(500, 300))
                    current_col = _next_image_col(ws, current_col, 500)
                    max_img_end_row = max(max_img_end_row, img_end_row)
                except Exception:
                    pass
            if all_figs:
                end_row = max_img_end_row  # 跳过图片占用的所有行，避免重叠

            # 每个数据集分别输出；多标签仅改变表头，不改变数据集粒度。
            for ds_key, ds in self._datasets.items():
                try:
                    labels_arg = self._label_names if is_multi else None
                    ft = self.get_feature_bin_table(
                        feat,
                        ds_key,
                        max_n_bins=n_bins,
                        method=bin_method,
                        margins=True,
                        labels=labels_arg,
                    )
                    ft_pct = (
                        [c for c in ft.columns if c[-1] in self._PERCENT_COLS]
                        if isinstance(ft.columns, pd.MultiIndex)
                        else [c for c in self._PERCENT_COLS if c in ft.columns]
                    )
                    ft_cond = (
                        [c for c in ft.columns if c[-1] in self._CONDITION_COLS]
                        if isinstance(ft.columns, pd.MultiIndex)
                        else [c for c in self._CONDITION_COLS if c in ft.columns]
                    )

                    if amount_col:
                        table_start = end_row + 1
                        order_end_row, order_end_col = dataframe2excel(
                            ft,
                            writer,
                            sheet_name=ws,
                            title=f"{ds.label} 订单口径",
                            start_row=table_start,
                            start_col=2,
                            percent_cols=ft_pct,
                            condition_cols=ft_cond,
                            condition_color="F76E6C",
                        )
                        ft_amt = self.get_feature_bin_table(
                            feat,
                            ds_key,
                            max_n_bins=n_bins,
                            method=bin_method,
                            margins=True,
                            amount_col=amount_col,
                            labels=labels_arg,
                        )
                        amt_pct = (
                            [c for c in ft_amt.columns if c[-1] in self._PERCENT_COLS]
                            if isinstance(ft_amt.columns, pd.MultiIndex)
                            else [c for c in self._PERCENT_COLS if c in ft_amt.columns]
                        )
                        amt_cond = (
                            [c for c in ft_amt.columns if c[-1] in self._CONDITION_COLS]
                            if isinstance(ft_amt.columns, pd.MultiIndex)
                            else [c for c in self._CONDITION_COLS if c in ft_amt.columns]
                        )
                        amount_end_row, _ = dataframe2excel(
                            ft_amt,
                            writer,
                            sheet_name=ws,
                            title=f"{ds.label} 金额口径",
                            start_row=table_start,
                            start_col=order_end_col + 1,
                            percent_cols=amt_pct,
                            condition_cols=amt_cond,
                            condition_color="F76E6C",
                        )
                        end_row = max(order_end_row, amount_end_row)
                    else:
                        end_row, _ = dataframe2excel(
                            ft,
                            writer,
                            sheet_name=ws,
                            title=ds.label,
                            start_row=end_row + 1,
                            percent_cols=ft_pct,
                            condition_cols=ft_cond,
                            condition_color="F76E6C",
                        )
                except Exception as exc:
                    logger.warning("生成特征 %s 的 %s 分箱表失败: %s", feat, ds.label, exc)

            # PSI 图表和数据表
            psi_fig_paths = plot_paths.get(f"feat_psi_{feat}", [])
            psi_df = psi_tables.get(f"feat_psi_{feat}")
            if psi_fig_paths:
                for fig_path in psi_fig_paths:
                    try:
                        end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2), figsize=(500, 300))
                    except Exception:
                        pass
            if isinstance(psi_df, pd.DataFrame) and not psi_df.empty:
                end_row, _ = dataframe2excel(
                    psi_df,
                    writer,
                    sheet_name=ws,
                    title="PSI稳定性分析",
                    start_row=end_row + 1,
                )

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 4-稳定性分析 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("4-稳定性分析")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="四、模型稳定性分析", style="header_middle", end_space=(2, max_col)
        )
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        stab_section = 1

        # 4.1 评分分布统计
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value=f"{stab_section}、评分分布统计",
            style="header_middle",
            align={"horizontal": "left"},
        )
        score_dist_rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            sc = ds.score
            row: Dict[str, Any] = {"数据集": ds.label}
            row["样本数"] = len(sc)
            row["均值"] = float(np.nanmean(sc))
            row["标准差"] = float(np.nanstd(sc))
            row["最小值"] = float(np.nanmin(sc))
            row["25%分位"] = float(np.nanpercentile(sc, 25))
            row["中位数"] = float(np.nanpercentile(sc, 50))
            row["75%分位"] = float(np.nanpercentile(sc, 75))
            row["最大值"] = float(np.nanmax(sc))
            score_dist_rows.append(row)
        score_dist_df = pd.DataFrame(score_dist_rows)
        end_row, _ = dataframe2excel(
            score_dist_df,
            writer,
            sheet_name=ws,
            start_row=end_row + 1,
        )
        stab_section += 1

        # 4.2 评分PSI矩阵（数据集两两对比）
        if len(self._datasets) >= 2:
            from ..core.metrics import psi as _psi

            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{stab_section}、评分PSI对比矩阵",
                style="header_middle",
                align={"horizontal": "left"},
            )
            ds_keys_list = list(self._datasets.keys())
            labels = [self._datasets[k].label for k in ds_keys_list]
            psi_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
            for i, k1 in enumerate(ds_keys_list):
                for j, k2 in enumerate(ds_keys_list):
                    if i == j:
                        psi_matrix.iloc[i, j] = 0.0
                    else:
                        try:
                            psi_matrix.iloc[i, j] = _psi(self._datasets[k1].score, self._datasets[k2].score)
                        except Exception:
                            pass
            end_row, _ = dataframe2excel(psi_matrix, writer, sheet_name=ws, start_row=end_row + 1, index=True)

            # 评分PSI参考阈值说明
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 1, 2),
                value="PSI参考标准：<0.1 稳定 | 0.1~0.25 略变 | >0.25 不稳定",
                style="middle",
                align={"horizontal": "left"},
            )
            stab_section += 1

        # 4.3 评分漂移分析（以训练集为基准）
        if "train" in self._datasets and len(self._datasets) >= 2:
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{stab_section}、评分漂移分析（vs 训练集）",
                style="header_middle",
                align={"horizontal": "left"},
            )
            drift_rows: List[Dict[str, Any]] = []
            base_scores = self._datasets["train"].score
            for ds_key, ds in self._datasets.items():
                if ds_key == "train":
                    continue
                sc = ds.score
                drift = {
                    "数据集": ds.label,
                    "vs": "训练集",
                    "均值偏移": float(np.nanmean(sc) - np.nanmean(base_scores)),
                    "均值偏移%": float((np.nanmean(sc) - np.nanmean(base_scores)) / (np.nanstd(base_scores) + 1e-9)),
                    "中位数偏移": float(np.nanmedian(sc) - np.nanmedian(base_scores)),
                    "好样本(评分>600)占比": float((sc > 600).sum() / len(sc)),
                    "坏样本(评分<500)占比": float((sc < 500).sum() / len(sc)),
                }
                drift_rows.append(drift)
            if drift_rows:
                drift_df = pd.DataFrame(drift_rows)
                pct_cols = [c for c in drift_df.columns if "%" in c or "占比" in c]
                end_row, _ = dataframe2excel(
                    drift_df,
                    writer,
                    sheet_name=ws,
                    start_row=end_row + 1,
                    percent_cols=pct_cols,
                )
            stab_section += 1

        # 4.4 逐特征PSI稳定性表
        if len(self._datasets) >= 2:
            from ..core.metrics import psi as _psi_feat

            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{stab_section}、入模特征PSI稳定性",
                style="header_middle",
                align={"horizontal": "left"},
            )
            importance = self.get_feature_importance()
            feat_list = importance.index.tolist() if not importance.empty else self.feature_names
            psi_rows: List[Dict[str, Any]] = []
            base_ds = self._datasets.get("train") or self._datasets[list(self._datasets.keys())[0]]
            other_ds_keys = [k for k in self._datasets if k != "train"]
            if not other_ds_keys:
                other_ds_keys = [k for k in self._datasets if k != list(self._datasets.keys())[0]]

            for feat in feat_list:
                row: Dict[str, Any] = {"特征": feat}
                has_psi = False
                for dk in other_ds_keys:
                    if dk in self._datasets and feat in self._datasets[dk].X.columns:
                        try:
                            psi_val = _psi_feat(base_ds.X[feat], self._datasets[dk].X[feat])
                            row[f"PSI({self._datasets[dk].label})"] = psi_val
                            has_psi = True
                        except Exception:
                            row[f"PSI({self._datasets[dk].label})"] = np.nan
                if has_psi:
                    psi_rows.append(row)
            if psi_rows:
                psi_feat_df = pd.DataFrame(psi_rows)
                end_row, _ = dataframe2excel(
                    psi_feat_df,
                    writer,
                    sheet_name=ws,
                    start_row=end_row + 1,
                )
            stab_section += 1

        # ============================================================
        # 5-模型参数 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("5-模型参数")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="五、模型选型及参数", style="header_middle", end_space=(2, max_col)
        )
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        param_section = 1

        # 5.1 模型选型
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value=f"{param_section}、模型选型",
            style="header_middle",
            align={"horizontal": "left"},
        )
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row, 2), value=model_name, style="middle", align={"horizontal": "left"}
        )
        param_section += 1

        # 5.2 模型参数
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value=f"{param_section}、模型参数",
            style="header_middle",
            align={"horizontal": "left"},
        )
        params_str = ""
        if hasattr(self.model, "get_params"):
            try:
                params_str = str(self.model.get_params())
            except Exception:
                pass
        if not params_str and hasattr(self.model, "__dict__"):
            params_str = str(
                {k: v for k, v in self.model.__dict__.items() if not k.startswith("_") and not callable(v)}
            )
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row, 2), value=params_str or "N/A", style="middle", align={"horizontal": "left"}
        )
        param_section += 1

        # 5.3 入模特征列表
        end_row, _ = writer.insert_value2sheet(
            ws,
            (end_row + 2, 2),
            value=f"{param_section}、入模特征列表",
            style="header_middle",
            align={"horizontal": "left"},
        )
        features_df = pd.DataFrame({"序号": range(1, len(self.feature_names) + 1), "变量名": self.feature_names})
        if feature_map:
            features_df["变量含义"] = [feature_map.get(f, "") for f in self.feature_names]
        end_row, _ = dataframe2excel(
            features_df, writer, sheet_name=ws, start_row=end_row + 1, left_cols=["变量名", "变量含义"]
        )
        param_section += 1

        # 5.4+ 评分卡专属内容
        # 判断是否为评分卡模型
        is_scorecard = hasattr(self.model, "lr_model") and hasattr(self.model, "scorecard_points")

        if is_scorecard:
            # plot_weights + LR 拟合结果
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{param_section}、逻辑回归拟合结果",
                style="header_middle",
                align={"horizontal": "left"},
            )
            weights_figs = plot_paths.get("model_weights", [])
            if weights_figs:
                for fig_path in weights_figs:
                    try:
                        end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2), figsize=(500, 300))
                    except Exception:
                        pass
            try:
                lr_summary = self.model.lr_model.summary()
                end_row, _ = dataframe2excel(
                    lr_summary, writer, sheet_name=ws, start_row=end_row + 1, title="逻辑回归系数"
                )
            except Exception:
                pass
            param_section += 1

            # 评分卡刻度配置
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{param_section}、评分卡刻度配置",
                style="header_middle",
                align={"horizontal": "left"},
            )
            try:
                scale_df = self.model.scorecard_scale()
                end_row, _ = dataframe2excel(
                    scale_df, writer, sheet_name=ws, start_row=end_row + 1, right_cols=["刻度项"], left_cols=["备注"]
                )
            except Exception:
                pass
            param_section += 1

            # 评分卡
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{param_section}、评分卡分值表",
                style="header_middle",
                align={"horizontal": "left"},
            )
            try:
                sc_points = self.model.scorecard_points(feature_map=feature_map)
                end_row, _ = dataframe2excel(
                    sc_points,
                    writer,
                    sheet_name=ws,
                    start_row=end_row + 1,
                    right_cols=["对应分数", "变量分箱", "变量名称"],
                )
            except Exception:
                pass
            param_section += 1

            # 评分与 Odds 对照
            end_row, _ = writer.insert_value2sheet(
                ws,
                (end_row + 2, 2),
                value=f"{param_section}、评分与Odds对照表",
                style="header_middle",
                align={"horizontal": "left"},
            )
            try:
                odds_ref = self.model.score_odds_reference
                end_row, _ = dataframe2excel(odds_ref, writer, sheet_name=ws, start_row=end_row + 1)
            except Exception:
                pass
            param_section += 1

            # 评分漂移分析
            if len(self._datasets) >= 2:
                end_row, _ = writer.insert_value2sheet(
                    ws,
                    (end_row + 2, 2),
                    value=f"{param_section}、稳定性分析",
                    style="header_middle",
                    align={"horizontal": "left"},
                )
                score_psi_figs = plot_paths.get("score_psi", [])
                if score_psi_figs:
                    for fig_path in score_psi_figs:
                        try:
                            end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2), figsize=(500, 300))
                        except Exception:
                            pass
                score_psi_df = psi_tables.get("score_psi")
                if isinstance(score_psi_df, pd.DataFrame) and not score_psi_df.empty:
                    end_row, _ = dataframe2excel(
                        score_psi_df, writer, sheet_name=ws, start_row=end_row + 1, title="评分PSI"
                    )

        # ============================================================
        # 6-模型部署需求 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("6-模型部署需求")
        end_row, _ = writer.insert_value2sheet(
            ws, (2, 2), value="六、模型部署需求", style="header_middle", end_space=(2, max_col)
        )
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 6.1 入模变量信息
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="1、入模变量信息", style="header_middle", align={"horizontal": "left"}
        )
        if feature_info is not None and isinstance(feature_info, pd.DataFrame) and not feature_info.empty:
            end_row, _ = dataframe2excel(feature_info, writer, sheet_name=ws, start_row=end_row + 1)
        else:
            fi_rows: List[Dict[str, Any]] = []
            for idx, feat in enumerate(self.feature_names):
                fi_rows.append(
                    {
                        "序号": idx + 1,
                        "特征名称": feat,
                        "特征含义": (feature_map or {}).get(feat, ""),
                        "字段类型": str(self._datasets["train"].X[feat].dtype),
                        "缺失值处理": "默认处理",
                    }
                )
            end_row, _ = dataframe2excel(pd.DataFrame(fi_rows), writer, sheet_name=ws, start_row=end_row + 1)

        # 6.2 生产订单测试用例
        end_row, _ = writer.insert_value2sheet(
            ws, (end_row + 2, 2), value="2、生产订单测试用例", style="header_middle", align={"horizontal": "left"}
        )
        try:
            train_ds = self._datasets["train"]
            sample_n = min(5, len(train_ds.X))
            sample_X = train_ds.X[self.feature_names].iloc[:sample_n].copy()

            # 支持定位字段（订单号等）显示在最前方
            if loc_cols:
                if isinstance(loc_cols, str):
                    loc_cols = [loc_cols]
                loc_cols = [c for c in loc_cols if c in train_ds.X.columns]

            test_cases = sample_X.reset_index(drop=True)
            if loc_cols:
                loc_df = train_ds.X[loc_cols].iloc[:sample_n].reset_index(drop=True)
                for i, col in enumerate(loc_cols):
                    test_cases.insert(i, col, loc_df[col])
            test_cases.insert(0, "序号", range(1, sample_n + 1))
            test_cases["模型分数"] = train_ds.score[:sample_n]
            end_row, _ = dataframe2excel(test_cases, writer, sheet_name=ws, start_row=end_row + 1)
        except Exception:
            pass

        # ============================================================
        # 保存
        # ============================================================
        writer.save(filepath)
        return filepath

    # ---------- 12. to_dict ----------

    def to_dict(self) -> Dict[str, Any]:
        labels_arg = self._label_names if self._is_multi_label() else None
        result: Dict[str, Any] = {
            "summary": self.summary().reset_index().to_dict(orient="records"),
            "metrics": self.get_metrics().to_dict(orient="records"),
            "feature_importance": self.get_feature_importance().reset_index().to_dict(orient="records"),
        }
        for ds_key in self._datasets:
            result[f"bin_table_{ds_key}"] = self.get_bin_table(ds_key, labels=labels_arg).to_dict(orient="records")
        return result


# ---------------------------------------------------------------------------
# 快捷函数
# ---------------------------------------------------------------------------


def auto_model_report(
    model,
    datasets: Optional[Union[List, Dict]] = None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    feature_names: Optional[List[str]] = None,
    target: Optional[Union[str, Dict]] = None,
    overdue: Optional[Union[str, List[str]]] = None,
    dpds: Optional[Union[int, float, List[Union[int, float]]]] = None,
    excel_path: Optional[str] = None,
    verbose: bool = True,
    n_bins: int = 10,
    bin_method: str = "quantile",
    amount_col: Optional[str] = None,
    date_col: Optional[str] = None,
    date_freq: Optional[str] = None,
    group_col: Optional[str] = None,
    with_plots: bool = True,
    model_name: Optional[str] = None,
    project_desc: Optional[str] = None,
    feature_map: Optional[Dict[str, str]] = None,
    feature_info: Optional[pd.DataFrame] = None,
    show_lift: bool = True,
    show_importance: bool = True,
    data_source: Optional[str] = None,
    loc_cols: Optional[Union[str, List[str]]] = None,
) -> ModelReport:
    """一键生成模型报告.

    支持三种调用方式：

    1. datasets API（推荐）：传入数据集字典/列表
       - dict: {'train': DataFrame, 'test': DataFrame, 'oot': DataFrame}
         DataFrame 需包含目标列，或通过 overdue/dpds 自动构建标签
       - list: [DataFrame, DataFrame, ...] 自动命名为训练集、测试集、OOT集...

    2. 兼容 API：传入 X_train/y_train/X_test/y_test
       - sklearn 风格：target='target'
       - overdue/dpds 风格：传入单独的 overdue/dpds 参数

    overdue/dpds 用法（自动从 X 构建二分类标签）::

        # 单阈值：MOB 任一期间 DPD > 5 则 y=1
        auto_model_report(model, X_train=df, overdue='dpds', dpds=5)

        # 多阈值：MOB1 DPD>15 或 MOB3 DPD>7 任一触发则 y=1
        auto_model_report(
            model,
            X_train=df,
            overdue=['dpds_m1', 'dpds_m3'],
            dpds=[15, 7, 0],
        )

    示例::

        # 方式1: datasets dict（DataFrame 直接传入，X 中含目标列）
        auto_model_report(model, datasets={'train': train_df, 'test': test_df}, excel_path='report.xlsx')

        # 方式1: datasets list（自动命名）
        auto_model_report(model, datasets=[train_df, test_df], excel_path='report.xlsx')

        # 方式1: overdue/dpds 自动构建标签
        auto_model_report(
            model,
            datasets={'train': df},
            overdue='dpds',
            dpds=[15, 7, 0],
            excel_path='report.xlsx',
        )

        # 方式2: 兼容 sklearn API（分离 X/y）
        auto_model_report(
            model,
            X_train=X, y_train=y,
            X_test=X_val, y_test=y_val,
            excel_path='report.xlsx',
        )

    :param model: 训练好的模型（ScoreCard / XGBoost / LightGBM / sklearn 等）
    :param datasets: 数据集字典/列表，字典键为数据集名称（推荐）
    :param X_train: 训练集特征（兼容旧 API）
    :param y_train: 训练集标签（兼容旧 API）
    :param X_test: 测试集/OOT 特征（兼容旧 API）
    :param y_test: 测试集/OOT 标签（兼容旧 API）
    :param feature_names: 特征名称列表
    :param target: 目标列配置，str 为列名，dict 为 {'overdue': col, 'dpds': threshold}
    :param overdue: 逾期列名（str）或多个列名（List[str]），与 dpds 配合自动构建标签
    :param dpds: 逾期天数阈值（int/float）或多个阈值（List），与 overdue 配合使用
    :param excel_path: Excel 报告输出路径
    :param verbose: 是否打印控制台报告
    :param n_bins: 分箱数
    :param bin_method: 分箱方法
    :param amount_col: 金额字段（用于金额口径分析）
    :param date_col: 日期字段（用于分月分析）
    :param date_freq: 日期频率，支持 'D', 'W', 'M', 'Q' 等（默认自动推断）
    :param group_col: 分组字段（用于分组坏样本率分析）
    :param with_plots: 是否生成图表
    :param model_name: 模型名称
    :param project_desc: 项目描述
    :param feature_map: 特征名称到含义的映射
    :param feature_info: 特征部署信息表
    :param show_lift: 是否在报告中显示 LIFT 曲线
    :param show_importance: 是否在报告中显示特征重要性
    :param data_source: 数据源描述
    :param loc_cols: 定位字段（订单号等），支持 str 或 List[str]，用于生产测试用例列
    :return: ModelReport 实例
    """
    report = ModelReport(
        model=model,
        datasets=datasets,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        target=target,
        overdue=overdue,
        dpds=dpds,
    )

    if verbose:
        report.print_report(n_bins=n_bins)

    if excel_path:
        report.to_excel(
            excel_path,
            n_bins=n_bins,
            bin_method=bin_method,
            amount_col=amount_col,
            date_col=date_col,
            date_freq=date_freq,
            group_col=group_col,
            with_plots=with_plots,
            model_name=model_name,
            project_desc=project_desc,
            feature_map=feature_map,
            feature_info=feature_info,
            data_source=data_source,
            loc_cols=loc_cols,
        )
        if verbose:
            logger.info(f"\nExcel 报告已保存: {excel_path}")

    return report


def compare_models(
    models: Dict[str, object],
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    excel_path: Optional[str] = None,
) -> pd.DataFrame:
    """横向对比多个模型的评估指标.

    对每个模型分别构建 :class:`ModelReport` 并取其 :meth:`~ModelReport.summary`，
    纵向拼接为一张对比表；单个模型构建失败时以 ``错误`` 列记录原因，不影响其余模型。

    :param models: 模型名称到模型对象的映射，如 ``{'XGB': xgb_model, 'LR': lr_model}``
    :param X_train: 训练集特征
    :param y_train: 训练集标签（0/1）
    :param X_test: 测试集特征，可选
    :param y_test: 测试集标签，可选
    :param excel_path: 可选，若提供则将对比表导出到该 Excel 路径
    :return: 含 ``模型名称`` 列的指标对比 ``DataFrame``

    **参考样例**

    >>> from hscredit.report import compare_models
    >>> result = compare_models(
    ...     {'XGBoost': xgb_model, '逻辑回归': lr_model},
    ...     X_train, y_train, X_test, y_test,
    ...     excel_path='模型对比.xlsx',
    ... )
    >>> print(result)
    """
    parts: Dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        try:
            report = ModelReport(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            parts[name] = report.summary()
        except Exception as e:
            # 失败的模型以「错误」列记录原因，不影响其余模型对比
            parts[name] = pd.DataFrame({"错误": [str(e)]}, index=pd.Index(["target"], name="逾期指标"))

    # 以「模型名称」作为最外层行索引纵向拼接，保留 summary 的「统计指标 × 数据集」多层列
    result = pd.concat(parts, names=["模型名称"]) if parts else pd.DataFrame()
    if excel_path and not result.empty:
        from ..excel import dataframe2excel

        dataframe2excel(
            result,
            excel_path,
            index=True,
            percent_cols=_summary_percent_cols(result.columns),
        )
    return result


# 向后兼容别名：旧名称 QuickModelReport 等价于 ModelReport
QuickModelReport = ModelReport

__all__ = ["ModelReport", "QuickModelReport", "auto_model_report", "compare_models"]
