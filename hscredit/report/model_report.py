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

import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


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
        return proba[:, 1]
    return proba.reshape(-1)


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


# ---------------------------------------------------------------------------
# QuickModelReport
# ---------------------------------------------------------------------------

class QuickModelReport:
    """面向报表输出的快速模型报告封装.

    参考风控建模标准报告模板，生成多 Sheet 结构的 Excel / HTML 报告。
    """

    _PERCENT_COLS = [
        "样本占比", "好样本占比", "坏样本占比", "坏样本率",
        "LIFT值", "坏账改善", "累积LIFT值", "累积坏账改善", "分档KS值",
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

        支持两种调用方式：
        1. 兼容 API：传入 X_train/y_train/X_test/y_test（推荐）
           - sklearn 风格：target='target'
           - scorecardpipeline 风格：target='target' 且 X 中包含目标列
           - overdue/dpds 风格：target={'overdue': col, 'dpds': threshold} 或
                               传入单独的 overdue/dpds 参数
        2. 新 API（pipeline 风格）：传入 datasets 参数
           - dict: {'train': (X, y), 'test': (X, y), 'oot': (X, y)}
           - list: [(X, y), (X, y)] 自动命名为训练集、测试集、OOT集...
        3. datasets dict: 显式指定各数据集（最高优先级）
           - 覆盖 X_train/y_train/X_test/y_test

        :param model: 训练好的模型
        :param datasets: 数据集字典/列表，字典键为数据集名称，值为 (X, y) 元组
        :param X_train: 训练集特征（兼容旧 API）
        :param y_train: 训练集标签（兼容旧 API）
        :param X_test: 测试集特征（兼容旧 API）
        :param y_test: 测试集标签（兼容旧 API）
        :param feature_names: 特征名称列表
        :param target: 目标列配置，str 为列名，dict 为 {'overdue': col, 'dpds': col, 'threshold': days}
        :param overdue: 逾期列名列名（str）或多个列名（List[str]），与 dpds 配合使用构建标签
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

        # 确定目标列名
        self._target_name = self._resolve_target_name(target)

        if datasets is not None:
            self._init_from_datasets(datasets)
        else:
            self._init_from_xy(X_train, y_train, X_test, y_test)

        # 从第一个数据集获取特征名
        if not hasattr(self, 'feature_names') or not self.feature_names:
            if self._datasets:
                first_ds = next(iter(self._datasets.values()))
                self.feature_names = list(first_ds.X.columns)
            elif self._feature_names:
                self.feature_names = self._feature_names
            else:
                self.feature_names = []

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

    def _build_y(self, X: pd.DataFrame, target_cfg) -> pd.Series:
        """根据 target 配置从 X 构建 y 标签.

        支持三种配置：
        - None: 从 X 中查找 'target' 列
        - str: 直接取 X[target] 作为标签
        - dict: 联合构建标签
            - 单逾期列:  target={'overdue': col, 'dpds': threshold} 或
                        target={'overdue': col, 'dpds': [t1, t2, ...]}
            - 多逾期列:  target={'overdue': [col1, col2], 'dpds': [t1, t2, ...]}
                          每列 × 每阈值生成指标，任一为真则 label=1

        注：overdue/dpds 也可通过 __init__ 单独参数传入，内部会合并为 dict。
        """
        if target_cfg is None:
            for col in ("target", "label", "y", "flag", "overdue"):
                if col in X.columns:
                    return _ensure_series(X[col], name="target")
            raise ValueError(
                "未找到目标列（target），请通过 target 参数指定标签列名，"
                "或传入 dict={'overdue': col, 'dpds': threshold} 联合构建"
            )

        if isinstance(target_cfg, str):
            if target_cfg in X.columns:
                return _ensure_series(X[target_cfg], name=target_cfg)
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

            # 每列 × 每阈值，生成全指标，任一为真则 y=1
            indicators = pd.DataFrame(index=X.index)
            for col in overdue_cols:
                for t in thresholds:
                    if dpds_col is not None and dpds_col in X.columns:
                        # dpds 列 > threshold
                        indicators[f"{col}>{t}"] = X[dpds_col] > t
                    else:
                        # col 列 > threshold
                        indicators[f"{col}>{t}"] = X[col] > t

            y = indicators.any(axis=1).astype(int)
            return _ensure_series(y, name=label_name)

        raise ValueError(f"target 参数格式错误：{target_cfg}")

    def _init_from_datasets(self, datasets):
        """从 datasets 初始化数据集."""
        if isinstance(datasets, dict):
            # {'train': (X, y), 'test': (X, y), ...}
            default_labels = {
                "train": "训练集", "test": "测试集",
                "oot": "OOT集", "val": "验证集",
            }
            for key, value in datasets.items():
                if not isinstance(value, (tuple, list)) or len(value) < 2:
                    raise ValueError(f"数据集 '{key}' 格式错误，应为 (X, y) 元组")
                X_raw, y_raw = value[0], value[1]
                label = default_labels.get(key, key)
                X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)

                # y 为 None 时，根据 target 配置从 X 推导标签
                if y_raw is None:
                    y_s = self._build_y(X_df, self._target_cfg)
                else:
                    y_s = _ensure_series(y_raw, name=self._target_name)

                self._add_dataset(key, label, X_df, y_s)
                self._datasets_info[key] = label

        elif isinstance(datasets, (list, tuple)):
            # [(X, y), (X, y), ...] - 自动命名
            default_names = ["train", "test", "oot", "val", "dev"]
            default_labels = ["训练集", "测试集", "OOT集", "验证集", "开发集"]
            for i, (X_raw, y_raw) in enumerate(datasets):
                key = default_names[i] if i < len(default_names) else f"dataset_{i}"
                label = default_labels[i] if i < len(default_labels) else f"数据集{i+1}"
                X_df = _ensure_dataframe(X_raw, feature_names=self._feature_names)

                if y_raw is None:
                    y_s = self._build_y(X_df, self._target_cfg)
                else:
                    y_s = _ensure_series(y_raw, name=self._target_name)

                self._add_dataset(key, label, X_df, y_s)
                self._datasets_info[key] = label

    def _init_from_xy(self, X_train, y_train, X_test, y_test):
        """从 X/y 参数初始化（兼容旧 API 及 scorecardpipeline 风格）."""
        X_train_df = _ensure_dataframe(X_train, feature_names=self._feature_names)

        # 支持 y_train 为 None 的 scorecardpipeline 风格（从 X 中推导标签）
        if y_train is None:
            y_train_s = self._build_y(X_train_df, self._target_cfg)
        else:
            y_train_s = _ensure_series(y_train, name=self._target_name)

        self._add_dataset("train", "训练集", X_train_df, y_train_s)
        self._datasets_info["train"] = "训练集"

        if X_test is not None:
            X_test_df = _ensure_dataframe(X_test, feature_names=list(X_train_df.columns))
            if y_test is None:
                y_test_s = self._build_y(X_test_df, self._target_cfg)
            else:
                y_test_s = _ensure_series(y_test, name=self._target_name)
            self._add_dataset("test", "测试集", X_test_df, y_test_s)
            self._datasets_info["test"] = "测试集"

    # ---------- 数据集管理 ----------

    def _add_dataset(self, key: str, label: str, X: pd.DataFrame, y: pd.Series):
        self._datasets[key] = ReportDataset(
            name=key,
            label=label,
            X=X,
            y=y,
            y_proba=_proba_pos(self.model, X),
            score=_score_from_model(self.model, X),
        )

    def add_dataset(self, key: str, label: str, X, y, feature_names: Optional[List[str]] = None):
        """添加额外数据集（如 OOT）用于报告."""
        X = _ensure_dataframe(X, feature_names=feature_names or self.feature_names)
        y = _ensure_series(y, name=self._target_name)
        self._add_dataset(key, label, X, y)

    # ---------- 1. 模型性能指标 ----------

    def get_metrics(self) -> pd.DataFrame:
        """KS / AUC / PSI 等核心指标."""
        if self._metrics_cache is not None:
            return self._metrics_cache.copy()

        from ..core.metrics import ks, auc, psi

        ds_keys = [k for k in ["train", "test"] + [k for k in self._datasets if k not in ("train", "test")] if k in self._datasets]
        labels = {k: self._datasets[k].label for k in ds_keys}

        rows = []
        rows.append({"统计项": "KS", **{labels[k]: ks(self._datasets[k].y, self._datasets[k].y_proba) for k in ds_keys}})
        rows.append({"统计项": "AUC", **{labels[k]: auc(self._datasets[k].y, self._datasets[k].y_proba) for k in ds_keys}})
        rows.append({"统计项": "样本数", **{labels[k]: len(self._datasets[k].y) for k in ds_keys}})
        rows.append({"统计项": "坏样本率", **{labels[k]: float(self._datasets[k].y.mean()) for k in ds_keys}})
        if len(ds_keys) >= 2:
            psi_row: Dict[str, Any] = {"统计项": "PSI", labels[ds_keys[0]]: "\\"}
            for k in ds_keys[1:]:
                try:
                    psi_row[labels[k]] = psi(self._datasets[ds_keys[0]].score, self._datasets[k].score)
                except Exception:
                    psi_row[labels[k]] = np.nan
            rows.append(psi_row)

        self._metrics_cache = pd.DataFrame(rows)
        return self._metrics_cache.copy()

    # ---------- 2. 评分分箱效果表 ----------

    def get_bin_table(
        self,
        dataset: str = "train",
        method: str = "quantile",
        max_n_bins: int = 10,
        amount_col: Optional[str] = None,
        margins: bool = True,
    ) -> pd.DataFrame:
        """使用 feature_bin_stats 生成评分分箱效果表."""
        from .feature_analyzer import feature_bin_stats

        ds = self._datasets[dataset]
        target_col = "__target__"
        score_col = "__score__"
        df = ds.X.copy()
        df[target_col] = ds.y.to_numpy()
        df[score_col] = ds.score

        kw: Dict[str, Any] = dict(
            feature=score_col,
            target=target_col,
            method=method,
            desc="模型评分",
            max_n_bins=max_n_bins,
            missing_separate=True,
            margins=margins,
        )
        if amount_col and amount_col in df.columns:
            kw["amount"] = amount_col

        table = feature_bin_stats(df, **kw)
        if isinstance(table, tuple):
            table = table[0]
        return table

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
                importances = pd.Series(
                    self.model.feature_importances_,
                    index=self.feature_names,
                )

            if importances is None:
                self._importance_cache = pd.DataFrame(columns=["特征重要性", "IV", "KS", "PSI"])
            else:
                importance_df = pd.DataFrame(index=importances.index)
                total = importances.sum()
                importance_df["特征重要性"] = importances.values / total if total else importances.values

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
        desc_stats = desc_stats.rename(columns={
            "count": "样本数", "mean": "平均值", "std": "标准差",
            "min": "最小值", "max": "最大值",
            "1%": "1%", "10%": "10%", "50%": "50%",
            "75%": "75%", "90%": "90%", "99%": "99%",
        })
        desc_stats["缺失率"] = train_X.isnull().mean()
        desc_stats["字段类型"] = train_X.dtypes.astype(str)
        desc_stats["枚举数"] = train_X.nunique()

        keep_cols = ["字段类型", "缺失率", "枚举数", "平均值", "标准差", "最小值", "1%", "10%", "50%", "75%", "90%", "99%", "最大值"]
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
    ) -> pd.DataFrame:
        """单特征分箱效果表，优先使用模型 binner."""
        from .feature_analyzer import feature_bin_stats

        ds = self._datasets[dataset]
        target_col = "__target__"
        df = ds.X.copy()
        df[target_col] = ds.y.to_numpy()

        kw: Dict[str, Any] = dict(
            feature=feature,
            target=target_col,
            method=method,
            max_n_bins=max_n_bins,
            margins=margins,
            missing_separate=True,
        )
        binner = getattr(self.model, "binner", None)
        if binner is not None:
            kw["binner"] = binner

        if amount_col and amount_col in df.columns:
            kw["amount"] = amount_col

        table = feature_bin_stats(df, **kw)
        if isinstance(table, tuple):
            table = table[0]
        return table

    # ---------- 8. 图表导出 ----------

    def _get_top_n_lift_table(
        self,
        percentiles: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10),
        amount_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """构建 TOP n% 尾部区分能力表.

        :param percentiles: TOP n% 的百分位列表
        :param amount_col: 金额字段名（可选），指定时同时输出金额口径
        """
        rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            n = len(ds.y)
            overall_bad_rate = float(ds.y.mean())

            sorted_idx = np.argsort(-ds.y_proba)
            sorted_y = ds.y.iloc[sorted_idx].values

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

            # 金额口径
            if amount_col and amount_col in ds.X.columns:
                amounts = ds.X[amount_col].values
                amounts_sorted = amounts[sorted_idx]
                overall_bad_amount = float(
                    (ds.y.iloc[sorted_idx].values * amounts_sorted).sum()
                    / amounts_sorted.sum()
                ) if amounts_sorted.sum() > 0 else overall_bad_rate

                amt_bad_rates: Dict[str, float] = {}
                amt_lifts: Dict[str, float] = {}
                amt_improvements: Dict[str, float] = {}

                for pct in percentiles:
                    top_n = max(1, int(n * pct))
                    top_amt = amounts_sorted[:top_n]
                    top_y_sorted = sorted_y[:top_n]
                    top_bad_amt = float(
                        (top_y_sorted * top_amt).sum() / top_amt.sum()
                    ) if top_amt.sum() > 0 else 0.0
                    lift_amt = top_bad_amt / overall_bad_amount if overall_bad_amount > 0 else 0.0
                    imp_amt = (top_bad_amt - overall_bad_amount) / overall_bad_amount if overall_bad_amount > 0 else 0.0
                    key = f"TOP {int(pct * 100)}%"
                    amt_bad_rates[key] = top_bad_amt
                    amt_lifts[key] = lift_amt
                    amt_improvements[key] = imp_amt

                amt_bad_rates["TOTAL"] = overall_bad_amount
                amt_lifts["TOTAL"] = 1.0
                amt_improvements["TOTAL"] = 0.0

                rows.append({"数据集": tag, "统计项": "金额坏样本率", **amt_bad_rates})
                rows.append({"数据集": tag, "统计项": "金额LIFT值", **amt_lifts})
                rows.append({"数据集": tag, "统计项": "金额坏账改善", **amt_improvements})

        return pd.DataFrame(rows)

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

    def _get_monthly_metrics(self, date_col: str) -> pd.DataFrame:
        """分月计算 KS/AUC."""
        from ..core.metrics import ks, auc

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
                if len(y_m) < 10 or y_m.nunique() < 2:
                    continue
                try:
                    rows.append({
                        "数据集": ds.label,
                        "月份": str(month),
                        "样本数": len(y_m),
                        "坏样本率": float(y_m.mean()),
                        "KS": ks(y_m, proba_m),
                        "AUC": auc(y_m, proba_m),
                    })
                except Exception:
                    pass
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
                bin_plot(bd, desc="模型评分", ending=f" {tag}", save=p, figsize=(8, 5))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            try:
                p = str(output_dir / f"ks_{ds_key}.png")
                ks_plot(ds.score, ds.y, title=f"{tag} KS曲线", save=p, figsize=(8, 5))
                _safe_close_figs()
                model_figs.append(p)
            except Exception:
                pass

            try:
                p = str(output_dir / f"lift_{ds_key}.png")
                lift_plot(ds.y, ds.y_proba, n_bins=20, title=f"{tag} LIFT曲线", save=p, figsize=(8, 5))
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
        for feat in (top_features or self.feature_names):
            # 分箱图：按 train/test 顺序分组
            bin_figs: List[str] = []
            for ds_key, ds in self._datasets.items():
                try:
                    ft = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, margins=True)
                    fd = ft.iloc[:-1].reset_index(drop=True) if len(ft) > 1 else ft
                    p = str(output_dir / f"bin_{feat}_{ds_key}.png")
                    bin_plot(fd, desc=feat, ending=f" {ds.label}", save=p, figsize=(8, 5))
                    _safe_close_figs()
                    bin_figs.append(p)
                except Exception:
                    pass
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
                    is_categorical = col.dtype == 'object' or (
                        col.dtype in ['int64', 'float64'] and col.nunique() <= 10
                    )
                    if is_categorical:
                        # 类别特征跳过KS图
                        continue
                    y_f = ds.y.loc[col.index]
                    # 确保标签是二分类
                    if y_f.nunique() < 2:
                        continue
                    p = str(output_dir / f"ks_{feat}_{ds_key}.png")
                    ks_plot(col, y_f, title=f"{ds.label} {feat}", save=p, figsize=(8, 5))
                    _safe_close_figs()
                    ks_figs.append(p)
                except Exception:
                    pass
            if ks_figs:
                paths[f"feat_hist_{feat}"] = ks_figs

            # PSI 图（训练集 vs 第一个非训练集）
            if len(ds_keys) >= 2:
                try:
                    train_vals = self._datasets[ds_keys[0]].X[feat].dropna()
                    test_vals = self._datasets[ds_keys[1]].X[feat].dropna()
                    p = str(output_dir / f"psi_{feat}.png")
                    psi_result = psi_plot(train_vals, test_vals, desc=feat, save=p, result=True, plot=True)
                    _safe_close_figs()
                    paths[f"feat_psi_{feat}"] = [p]
                    if isinstance(psi_result, pd.DataFrame):
                        tables[f"feat_psi_{feat}"] = psi_result
                except Exception:
                    pass

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
                    p = str(output_dir / "score_psi.png")
                    score_psi_df = psi_plot(
                        self._datasets[ds_keys[0]].score,
                        self._datasets[ds_keys[1]].score,
                        desc="模型评分", save=p, result=True, plot=True,
                    )
                    _safe_close_figs()
                    paths["score_psi"] = [p]
                    if isinstance(score_psi_df, pd.DataFrame):
                        tables["score_psi"] = score_psi_df
                except Exception:
                    pass

        return paths, tables

    # ---------- 9. 模型摘要 ----------

    def summary(self) -> pd.DataFrame:
        from ..core.metrics import ks, auc

        rows: Dict[str, Any] = {"模型": self.model.__class__.__name__}
        for ds in self._datasets.values():
            prefix = ds.label
            try:
                rows[f"{prefix}_KS"] = ks(ds.y, ds.y_proba)
            except Exception:
                pass
            try:
                rows[f"{prefix}_AUC"] = auc(ds.y, ds.y_proba)
            except Exception:
                pass
            rows[f"{prefix}_样本数"] = len(ds.y)
            rows[f"{prefix}_坏样本率"] = float(ds.y.mean())

        fi = self.get_feature_importance(top_n=5)
        if not fi.empty:
            rows["Top1特征"] = fi.index[0]
            rows["Top1重要性"] = fi.iloc[0]["特征重要性"]

        return pd.DataFrame([rows])

    # ---------- 10. 控制台输出 ----------

    def print_report(self, n_bins: int = 10, **kwargs) -> None:
        print("=" * 72)
        print("模型评估快速报告")
        print("=" * 72)
        print("\n【模型性能指标】")
        print(self.get_metrics().to_string(index=False))

        importance = self.get_feature_importance(top_n=10)
        if not importance.empty:
            print("\n【Top 10 特征重要性】")
            print(importance.to_string())

        for ds_key, ds in self._datasets.items():
            print(f"\n【{ds.label}评分分箱效果】")
            print(self.get_bin_table(ds_key, max_n_bins=n_bins).to_string(index=False))
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
    ) -> str:
        """生成多 Sheet 结构的 Excel 模型报告.

        Sheet 结构：
        - 目录
        - 1-基本信息（项目目标、样本统计、分月/分组分布）
        - 2-模型性能（指标、TOP n%、PSI矩阵、分箱效果）
        - 3-入模变量分析（重要性、相关性、逐特征分箱/KS/PSI）
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
                    plot_dir, n_bins=n_bins, bin_method=bin_method, amount_col=amount_col,
                )

        writer = ExcelWriter()

        # ============================================================
        # 目录 Sheet
        # ============================================================
        contents = pd.DataFrame([
            {"序号": 1, "内容": "1-基本信息", "备注": "项目目标、样本选取、样本坏率分布"},
            {"序号": 2, "内容": "2-模型性能", "备注": "模型效果、区分度、稳定性等内容"},
            {"序号": 3, "内容": "3-入模变量分析", "备注": "模型变量有效性及不同数据集分箱情况"},
            {"序号": 4, "内容": "4-稳定性分析", "备注": "评分分布、PSI、CSI等稳定性分析"},
            {"序号": 5, "内容": "5-模型部署需求", "备注": "入模变量信息及测试用例"},
        ])

        ws = writer.get_sheet_by_name("目录")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="模型评估报告", style="header_middle", end_space=(2, max_col))
        end_row, _ = dataframe2excel(contents, writer, sheet_name=ws, start_row=end_row + 1)

        for i, row in contents.iterrows():
            try:
                target_cell = writer.get_cell_space((2, 2))
                writer.insert_hyperlink2sheet(ws, (end_row - len(contents) + i, 3), hyperlink=f"#'{row['内容']}'!{target_cell}")
            except Exception:
                pass

        _, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="版本号:", style="middle", end_space=(end_row + 1, 2))
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 3), value="V1.0", style="middle", end_space=(end_row + 1, 4))
        _, _ = writer.insert_value2sheet(ws, (end_row + 2, 2), value="创建日期:", style="middle", end_space=(end_row + 2, 2))
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 2, 3), value=date.today().strftime("%Y-%m-%d"), style="middle", end_space=(end_row + 2, 4))
        _, _ = writer.insert_value2sheet(ws, (end_row + 3, 2), value="模型名称:", style="middle", end_space=(end_row + 3, 2))
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 3, 3), value=model_name, style="middle", end_space=(end_row + 3, 4))

        # ============================================================
        # 1-基本信息 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("1-基本信息")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="一、基本信息", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 1.1 项目目标
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="1、项目目标", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        desc_text = project_desc or f"使用 {model_name} 模型进行信用风险评估"
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 2), value=desc_text, style="middle", end_space=(end_row, max_col), align={"horizontal": "left"})

        # 1.2 数据样本统计
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="2、数据样本统计", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        sample_rows: List[Dict[str, Any]] = []
        for ds_key, ds in self._datasets.items():
            sample_rows.append({
                "数据集": ds.label,
                "样本数": len(ds.y),
                "好样本数": int((1 - ds.y).sum()),
                "坏样本数": int(ds.y.sum()),
                "坏样本率": float(ds.y.mean()),
            })
        sample_df = pd.DataFrame(sample_rows)
        end_row, _ = dataframe2excel(sample_df, writer, sheet_name=ws, start_row=end_row + 1, percent_cols=["坏样本率"])

        # 1.3 样本时间/分组分布
        freq_label_map = {"D": "日", "W": "周", "M": "月", "Q": "季度", "Y": "年"}
        if date_col or group_col:
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="3、样本分布情况", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})

            # 时间分布
            if date_col:
                period_labels = {"D": "日期", "W": "周", "M": "月份", "Q": "季度", "Y": "年份"}
                freq = date_freq or "M"
                period_label = period_labels.get(freq, "周期")
                period_col_name = freq_label_map.get(freq, freq)

                for ds_key, ds in self._datasets.items():
                    if date_col in ds.X.columns:
                        dates = pd.to_datetime(ds.X[date_col])
                        try:
                            periods = dates.dt.to_period(freq)
                        except Exception:
                            periods = dates.dt.to_period("M")
                        period_stats = ds.y.groupby(periods).agg(["count", "sum", "mean"]).reset_index()
                        period_stats.columns = [period_label, "样本数", "坏样本数", "坏样本率"]
                        period_stats[period_label] = period_stats[period_label].astype(str)
                        period_stats["坏样本数"] = period_stats["坏样本数"].astype(int)
                        end_row, _ = dataframe2excel(
                            period_stats, writer, sheet_name=ws,
                            title=f"{ds.label} {period_col_name}度分布", start_row=end_row + 1,
                            percent_cols=["坏样本率"],
                        )

            # 分组分布
            if group_col:
                for ds_key, ds in self._datasets.items():
                    if group_col in ds.X.columns:
                        groups = ds.X[group_col]
                        group_stats = pd.DataFrame({
                            "分组": groups,
                            "样本数": 1,
                            "坏样本": ds.y.values,
                        }).groupby("分组").agg({"样本数": "count", "坏样本": "sum"}).reset_index()
                        group_stats["坏样本率"] = group_stats["坏样本"] / group_stats["样本数"]
                        end_row, _ = dataframe2excel(
                            group_stats, writer, sheet_name=ws,
                            title=f"{ds.label} 分组分布", start_row=end_row + 1,
                            percent_cols=["坏样本率"],
                        )

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 2-模型性能 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("2-模型性能")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="二、模型性能评估", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        section_idx = 1

        # 2.1 性能指标
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{section_idx}、模型性能验证指标", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        metrics = self.get_metrics()
        end_row, _ = dataframe2excel(
            metrics, writer, sheet_name=ws, title="模型性能指标",
            start_row=end_row + 1,
            percent_cols=[c for c in metrics.columns if c != "统计项"],
        )
        section_idx += 1

        # 2.2 分月模型效果
        if date_col:
            monthly_metrics = self._get_monthly_metrics(date_col)
            if not monthly_metrics.empty:
                end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{section_idx}、分月模型效果", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
                end_row, _ = dataframe2excel(
                    monthly_metrics, writer, sheet_name=ws, start_row=end_row + 1,
                    percent_cols=["坏样本率", "KS", "AUC"],
                )
                section_idx += 1

        # 2.3 模型尾部区分能力
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{section_idx}、模型尾部区分能力（TOP n%）", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        pct_keys = ["TOP 1%", "TOP 3%", "TOP 5%", "TOP 10%", "TOTAL"]
        if amount_col:
            lift_table = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10), amount_col=None)
            lift_amt = self._get_top_n_lift_table(percentiles=(0.01, 0.03, 0.05, 0.10), amount_col=amount_col)
            table_start = end_row + 1
            end_row1, end_col1 = dataframe2excel(
                lift_table, writer, sheet_name=ws,
                title="订单口径", start_row=table_start, start_col=2,
                percent_cols=pct_keys,
            )
            end_row2, _ = dataframe2excel(
                lift_amt, writer, sheet_name=ws,
                title="金额口径", start_row=table_start, start_col=end_col1 + 2,
                percent_cols=pct_keys,
            )
            end_row = max(end_row1, end_row2)
        else:
            lift_table = self._get_top_n_lift_table()
            end_row, _ = dataframe2excel(
                lift_table, writer, sheet_name=ws, start_row=end_row + 1,
                percent_cols=pct_keys,
            )
        section_idx += 1

        # 2.4 分月PSI矩阵
        if date_col:
            psi_matrix = self._get_monthly_psi_matrix(date_col)
            if not psi_matrix.empty:
                end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{section_idx}、分月对比PSI", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
                end_row, _ = dataframe2excel(psi_matrix, writer, sheet_name=ws, start_row=end_row + 1, index=True)
                section_idx += 1

        # 2.5 各数据集评分排序性
        for ds_key, ds in self._datasets.items():
            tag = ds.label
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{section_idx}、{tag}评分排序性", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})

            figs = plot_paths.get(f"model_{ds_key}", [])
            order_table = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, margins=True)
            pct_cols = [c for c in self._PERCENT_COLS if c in order_table.columns]
            cond_cols = [c for c in self._CONDITION_COLS if c in order_table.columns]

            if amount_col:
                table_start = end_row + 1
                end_row1, end_col1 = dataframe2excel(
                    order_table, writer, sheet_name=ws,
                    title=f"{tag} 订单口径", start_row=table_start, start_col=2,
                    percent_cols=pct_cols, condition_cols=cond_cols, condition_color="F76E6C",
                    figures=figs,
                )
                try:
                    amount_table = self.get_bin_table(ds_key, method=bin_method, max_n_bins=n_bins, amount_col=amount_col, margins=True)
                    amt_pct = [c for c in self._PERCENT_COLS if c in amount_table.columns]
                    amt_cond = [c for c in self._CONDITION_COLS if c in amount_table.columns]
                    end_row2, _ = dataframe2excel(
                        amount_table, writer, sheet_name=ws,
                        title=f"{tag} 金额口径", start_row=table_start, start_col=end_col1 + 2,
                        percent_cols=amt_pct, condition_cols=amt_cond, condition_color="F76E6C",
                    )
                    end_row = max(end_row1, end_row2)
                except Exception:
                    end_row = end_row1
            else:
                end_row, _ = dataframe2excel(
                    order_table, writer, sheet_name=ws,
                    title=f"{tag} 评分有效性", start_row=end_row + 1,
                    percent_cols=pct_cols, condition_cols=cond_cols, condition_color="F76E6C",
                    figures=figs,
                )
            section_idx += 1

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # ============================================================
        # 3-入模变量分析 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("3-入模变量分析")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="三、入模变量分析", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 3.1 入模变量重要性及分布情况
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="1、入模变量重要性及分布情况", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        features_summary = self._get_features_summary()
        end_row, _ = dataframe2excel(
            features_summary, writer, sheet_name=ws,
            start_row=end_row + 1,
            index=True,
        )

        # 3.2 相关性
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="2、入模变量相关性", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        corr_df = self.get_features_corr()
        corr_figs = plot_paths.get("feature_corr", [])
        end_row, _ = dataframe2excel(
            corr_df, writer, sheet_name=ws,
            start_row=end_row + 1,
            percent_cols=corr_df.columns.tolist(),
            index=True,
            figures=corr_figs,
        )

        # 3.3 入模变量有效性分析
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="3、入模变量有效性分析", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})

        importance = self.get_feature_importance()
        feature_list = importance.index.tolist() if not importance.empty else self.feature_names
        ds_keys_list = list(self._datasets.keys())

        for i, feat in enumerate(feature_list):
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"3.{i + 1}、{feat} 有效性分析", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})

            # 插入图表：分箱图(train, test) + 分布图(train, test)
            bin_figs = plot_paths.get(f"feat_bin_{feat}", [])
            hist_figs = plot_paths.get(f"feat_hist_{feat}", [])
            all_figs = bin_figs + hist_figs

            # 各数据集分箱表（订单口径 + 金额口径）
            first_ds = True
            for ds_key, ds in self._datasets.items():
                try:
                    ft = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, margins=True)
                    ft_pct = [c for c in self._PERCENT_COLS if c in ft.columns]
                    ft_cond = [c for c in self._CONDITION_COLS if c in ft.columns]

                    if amount_col:
                        table_start = end_row + 1
                        end_row1, end_col1 = dataframe2excel(
                            ft, writer, sheet_name=ws,
                            title=f"{ds.label} 订单口径", start_row=table_start, start_col=2,
                            percent_cols=ft_pct, condition_cols=ft_cond, condition_color="F76E6C",
                            figures=all_figs if first_ds else [],
                        )
                        try:
                            ft_amt = self.get_feature_bin_table(feat, ds_key, max_n_bins=n_bins, method=bin_method, margins=True, amount_col=amount_col)
                            amt_pct = [c for c in self._PERCENT_COLS if c in ft_amt.columns]
                            amt_cond = [c for c in self._CONDITION_COLS if c in ft_amt.columns]
                            end_row2, _ = dataframe2excel(
                                ft_amt, writer, sheet_name=ws,
                                title=f"{ds.label} 金额口径", start_row=table_start, start_col=end_col1 + 2,
                                percent_cols=amt_pct, condition_cols=amt_cond, condition_color="F76E6C",
                            )
                            end_row = max(end_row1, end_row2)
                        except Exception:
                            end_row = end_row1
                    else:
                        end_row, _ = dataframe2excel(
                            ft, writer, sheet_name=ws,
                            title=f"{ds.label}", start_row=end_row + 1,
                            percent_cols=ft_pct, condition_cols=ft_cond, condition_color="F76E6C",
                            figures=all_figs if first_ds else [],
                        )
                    first_ds = False
                except Exception:
                    pass

            # PSI 图表和数据表
            psi_fig_paths = plot_paths.get(f"feat_psi_{feat}", [])
            psi_df = psi_tables.get(f"feat_psi_{feat}")
            if psi_fig_paths:
                for fig_path in psi_fig_paths:
                    try:
                        end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2))
                    except Exception:
                        pass
            if isinstance(psi_df, pd.DataFrame) and not psi_df.empty:
                end_row, _ = dataframe2excel(
                    psi_df, writer, sheet_name=ws,
                    title="PSI稳定性分析", start_row=end_row + 1,
                )

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 4-稳定性分析 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("4-稳定性分析")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="四、模型稳定性分析", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        stab_section = 1

        # 4.1 评分分布统计
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{stab_section}、评分分布统计", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
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
            score_dist_df, writer, sheet_name=ws, start_row=end_row + 1,
        )
        stab_section += 1

        # 4.2 评分PSI矩阵（数据集两两对比）
        if len(self._datasets) >= 2:
            from ..core.metrics import psi as _psi

            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{stab_section}、评分PSI对比矩阵", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
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
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="PSI参考标准：<0.1 稳定 | 0.1~0.25 略变 | >0.25 不稳定", style="middle", end_space=(end_row, max_col), align={"horizontal": "left"})
            stab_section += 1

        # 4.3 评分漂移分析（以训练集为基准）
        if "train" in self._datasets and len(self._datasets) >= 2:
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{stab_section}、评分漂移分析（vs 训练集）", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
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
                    drift_df, writer, sheet_name=ws, start_row=end_row + 1,
                    percent_cols=pct_cols,
                )
            stab_section += 1

        # 4.4 逐特征PSI稳定性表
        if len(self._datasets) >= 2:
            from ..core.metrics import psi as _psi_feat

            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{stab_section}、入模特征PSI稳定性", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
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
                    psi_feat_df, writer, sheet_name=ws, start_row=end_row + 1,
                )
            stab_section += 1

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 5-模型参数 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("5-模型参数")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="五、模型选型及参数", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        param_section = 1

        # 5.1 模型选型
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、模型选型", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 2), value=model_name, style="middle", end_space=(end_row, max_col), align={"horizontal": "left"})
        param_section += 1

        # 5.2 模型参数
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、模型参数", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        params_str = ""
        if hasattr(self.model, "get_params"):
            try:
                params_str = str(self.model.get_params())
            except Exception:
                pass
        if not params_str and hasattr(self.model, "__dict__"):
            params_str = str({k: v for k, v in self.model.__dict__.items() if not k.startswith("_") and not callable(v)})
        end_row, _ = writer.insert_value2sheet(ws, (end_row, 2), value=params_str or "N/A", style="middle", end_space=(end_row, max_col), align={"horizontal": "left"})
        param_section += 1

        # 5.3 入模特征列表
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、入模特征列表", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        features_df = pd.DataFrame({"序号": range(1, len(self.feature_names) + 1), "变量名": self.feature_names})
        if feature_map:
            features_df["变量含义"] = [feature_map.get(f, "") for f in self.feature_names]
        end_row, _ = dataframe2excel(features_df, writer, sheet_name=ws, start_row=end_row + 1)
        param_section += 1

        # 5.4+ 评分卡专属内容
        is_scorecard = hasattr(self.model, "lr_model") and hasattr(self.model, "scorecard_points")
        if is_scorecard:
            # plot_weights + LR 拟合结果
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、逻辑回归拟合结果", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
            weights_figs = plot_paths.get("model_weights", [])
            if weights_figs:
                for fig_path in weights_figs:
                    try:
                        end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2))
                    except Exception:
                        pass
            try:
                lr_summary = self.model.lr_model.summary()
                end_row, _ = dataframe2excel(lr_summary, writer, sheet_name=ws, start_row=end_row + 1, title="逻辑回归系数")
            except Exception:
                pass
            param_section += 1

            # 评分卡
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、评分卡", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
            try:
                sc_points = self.model.scorecard_points(feature_map=feature_map)
                end_row, _ = dataframe2excel(sc_points, writer, sheet_name=ws, start_row=end_row + 1, title="评分卡分值表")
            except Exception:
                pass
            param_section += 1

            # 评分与 Odds 对照
            end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、评分与Odds对照表", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
            try:
                odds_ref = self.model.score_odds_reference
                end_row, _ = dataframe2excel(odds_ref, writer, sheet_name=ws, start_row=end_row + 1)
            except Exception:
                pass
            param_section += 1

            # 评分漂移分析
            if len(self._datasets) >= 2:
                end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value=f"{param_section}、稳定性分析", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
                score_psi_figs = plot_paths.get("score_psi", [])
                if score_psi_figs:
                    for fig_path in score_psi_figs:
                        try:
                            end_row, _ = writer.insert_pic2sheet(ws, fig_path, (end_row + 1, 2))
                        except Exception:
                            pass
                score_psi_df = psi_tables.get("score_psi")
                if isinstance(score_psi_df, pd.DataFrame) and not score_psi_df.empty:
                    end_row, _ = dataframe2excel(score_psi_df, writer, sheet_name=ws, start_row=end_row + 1, title="评分PSI")

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 6-模型部署需求 Sheet
        # ============================================================
        ws = writer.get_sheet_by_name("6-模型部署需求")
        end_row, _ = writer.insert_value2sheet(ws, (2, 2), value="六、模型部署需求", style="header_middle", end_space=(2, max_col))
        try:
            writer.insert_hyperlink2sheet(ws, (2, 2), hyperlink="#'目录'!B2")
        except Exception:
            pass

        # 6.1 入模变量信息
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="1、入模变量信息", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        if feature_info is not None and isinstance(feature_info, pd.DataFrame) and not feature_info.empty:
            end_row, _ = dataframe2excel(feature_info, writer, sheet_name=ws, start_row=end_row + 1)
        else:
            fi_rows: List[Dict[str, Any]] = []
            for idx, feat in enumerate(self.feature_names):
                fi_rows.append({
                    "序号": idx + 1,
                    "特征名称": feat,
                    "特征含义": (feature_map or {}).get(feat, ""),
                    "字段类型": str(self._datasets["train"].X[feat].dtype),
                    "缺失值处理": "默认处理",
                })
            end_row, _ = dataframe2excel(pd.DataFrame(fi_rows), writer, sheet_name=ws, start_row=end_row + 1)

        # 6.2 生产订单测试用例
        end_row, _ = writer.insert_value2sheet(ws, (end_row + 1, 2), value="2、生产订单测试用例", style="header_middle", end_space=(end_row + 1, max_col), align={"horizontal": "left"})
        try:
            train_ds = self._datasets["train"]
            sample_n = min(5, len(train_ds.X))
            sample_X = train_ds.X[self.feature_names].iloc[:sample_n].copy()
            test_cases = sample_X.reset_index(drop=True)
            test_cases.insert(0, "序号", range(1, sample_n + 1))
            test_cases["模型分数"] = train_ds.score[:sample_n]
            end_row, _ = dataframe2excel(test_cases, writer, sheet_name=ws, start_row=end_row + 1)
        except Exception:
            pass

        try:
            writer.set_freeze_panes(ws, (5, 4))
        except Exception:
            pass

        # ============================================================
        # 保存
        # ============================================================
        writer.save(filepath)
        return filepath

    # ---------- 12. to_dict ----------

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "metrics": self.get_metrics().to_dict(orient="records"),
            "feature_importance": self.get_feature_importance().reset_index().to_dict(orient="records"),
        }
        for ds_key in self._datasets:
            result[f"bin_table_{ds_key}"] = self.get_bin_table(ds_key).to_dict(orient="records")
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
) -> QuickModelReport:
    """一键生成模型报告.

    支持两种调用方式：
    1. 新 API（推荐）：传入 datasets 参数
       - dict: {'train': (X, y), 'test': (X, y), 'oot': (X, y)}
       - list: [(X, y), (X, y), ...] 自动命名为训练集、测试集、OOT集...
    2. 兼容 API：传入 X_train/y_train/X_test/y_test

    overdue/dpds 用法（scorecardpipeline 风格）：
      overdue: 逾期列名列名（str）或多个列名（List[str]）
      dpds: 逾期天数阈值（int/float）或多个阈值（List[int/float]）
      示例：
        auto_model_report(model, X_train=df, overdue='dpds', dpds=[15, 7, 0])
        auto_model_report(model, X_train=df, overdue=['dpds_m1', 'dpds_m3'], dpds=[30, 15, 7, 0])
      等价于传 target={'overdue': ..., 'dpds': ...}

    :param model: 训练好的模型（ScoreCard / BaseRiskModel / sklearn 等）
    :param datasets: 数据集字典/列表，字典键为数据集名称，值为 (X, y) 元组
    :param X_train: 训练集特征（兼容旧 API）
    :param y_train: 训练集标签（兼容旧 API）
    :param X_test: 测试集/OOT 特征（兼容旧 API）
    :param y_test: 测试集/OOT 标签（兼容旧 API）
    :param feature_names: 特征名称列表
    :param target: 目标列配置，str 为列名，dict 为 {'overdue': col, 'dpds': col, 'threshold': days}
    :param overdue: 逾期列名列名（str）或多个列名（List[str]），与 dpds 配合使用构建标签
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
    :return: QuickModelReport 实例
    """
    report = QuickModelReport(
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
        )
        if verbose:
            print(f"\nExcel 报告已保存: {excel_path}")

    return report


def compare_models(
    models: Dict[str, object],
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    excel_path: Optional[str] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for name, model in models.items():
        try:
            report = QuickModelReport(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            summary = report.summary()
            summary.insert(0, "模型名称", name)
            rows.append(summary)
        except Exception as e:
            rows.append(pd.DataFrame([{"模型名称": name, "错误": str(e)}]))

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if excel_path and not result.empty:
        result.to_excel(excel_path, index=False)
    return result


ModelReport = QuickModelReport

__all__ = ["ModelReport", "QuickModelReport", "auto_model_report", "compare_models"]
