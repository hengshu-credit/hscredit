"""规则评估指标模块.

提供丰富的规则评估指标，所有指标计算统一收口到hscredit.core.metrics。
"""

import copy
import logging
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 从统一metrics模块导入指标
from ...core.metrics import (
    ks,
    gini,
    iv,
    lift_table as metrics_lift_table,
)

logger = logging.getLogger(__name__)

from ...utils.parallel import ParallelizableMixin
from ...utils.arguments import bind_column
from ...utils.data_contracts import prepare_xy
from ...core.metrics._semantics import rejection_statistics
from .base import _mining_workload


def _rule_metrics_worker(task):
    """评估一个独立规则并附加稳定的输入序号。"""
    calculator, ordinal, rule, X_train, y_train, X_test, y_test, kwargs = task
    result = calculator.evaluate_rule(
        copy.deepcopy(rule),
        X_train,
        y_train,
        X_test,
        y_test,
        **kwargs,
    )
    result["规则编号"] = ordinal
    result["规则"] = str(rule)
    return result


class RuleMetrics(ParallelizableMixin):
    """规则评估指标计算器.

    提供全面的规则评估指标，支持训练集和测试集的对比分析。
    所有指标计算统一收口到hscredit.core.metrics。

    **参数**

    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]

    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]

    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Optional[Union[int, float]]

    :param target_positive: 正类标签，默认1
    :type target_positive: int

    **参考样例**

    >>> from hscredit.core.rules.mining import RuleMetrics
    >>> metrics = RuleMetrics()
    >>> result = metrics.evaluate_rule(rule, X_train, y_train, X_test, y_test)  # 单规则评估：返回KS/AUC/IV等指标
    >>> results = metrics.evaluate_rules(rules, X_train, y_train, X_test, y_test)  # 批量规则评估：返回规则列表各自的指标
    """

    def __init__(
        self,
        target_positive: int = 1,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        """:param target_positive: 正类标签，默认1"""
        self.target_positive = target_positive
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.parallel_config = parallel_config

    def evaluate_rule(
        self,
        rule,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        amount_train: Optional[pd.Series] = None,
        amount_test: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """评估单个规则.

        :param rule: 规则对象（Rule或MinedRule）
        :param X_train: 训练集特征
        :type X_train: pd.DataFrame
        :param y_train: 训练集标签
        :type y_train: pd.Series
        :param X_test: 默认值为 ``None`` 。测试集特征（可选）
        :type X_test: Optional[pd.DataFrame]
        :param y_test: 默认值为 ``None`` 。测试集标签（可选）
        :type y_test: Optional[pd.Series]
        :param amount_train: 默认值为 ``None`` 。训练集金额（可选），追加金额指标，不替换件数指标；Series 按索引对齐
        :type amount_train: Optional[pd.Series]
        :param amount_test: 默认值为 ``None`` 。测试集金额（可选），仅作用于测试集的金额指标
        :type amount_test: Optional[pd.Series]
        :return: 评估指标字典
        """
        result = {}

        # 训练集评估
        train_metrics = self._calculate_metrics(rule, X_train, y_train, amount_train)
        for k, v in train_metrics.items():
            result[f"训练_{k}"] = v

        # 测试集评估
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(rule, X_test, y_test, amount_test)
            for k, v in test_metrics.items():
                result[f"测试_{k}"] = v

            # 同一命中口径的坏率差（训练减测试）；无命中人群时不虚构差值。
            result["badrate_diff"] = (
                train_metrics["命中坏样本率"] - test_metrics["命中坏样本率"]
                if train_metrics["命中样本数"] > 0 and test_metrics["命中样本数"] > 0
                else np.nan
            )

        return result

    def _calculate_metrics(
        self, rule, X: pd.DataFrame, y: pd.Series, amount: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """计算基础指标.

        规则效果类指标统一通过 Rule.report 计算得到；
        分类指标（accuracy/precision/recall/f1）仍基于命中掩码计算。
        """
        prepared = prepare_xy(X, y, require_y=True, allow_empty_features=True)
        X, y = prepared.X, prepared.y
        labels = pd.unique(y)
        if len(labels) > 2 or (len(labels) == 2 and self.target_positive not in labels):
            raise ValueError("规则评估目标必须为二分类，且包含指定的 target_positive 正类")
        y = y.eq(self.target_positive).astype(int)

        # 应用规则掩码（用于分类指标）
        if hasattr(rule, "evaluate"):
            mask = rule.evaluate(X)
        elif hasattr(rule, "predict"):
            mask = rule.predict(X)
        else:
            raise ValueError("规则对象必须有evaluate或predict方法")

        if isinstance(mask, pd.Series):
            mask = mask.values

        # Rule.report 计算规则效果
        data = X
        report_df = rule.report(datasets=data, target=y.to_numpy())

        hit = (
            report_df[report_df["分箱"] == "命中"].iloc[0].to_dict()
            if "分箱" in report_df.columns and not report_df[report_df["分箱"] == "命中"].empty
            else {}
        )

        # 分类指标
        y_pred = np.zeros_like(y)
        y_pred[mask] = 1

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

        total_bad = y.sum()
        total = len(y)
        hit_bad = float(hit.get("坏样本数", np.sum(y[mask]) if np.any(mask) else 0))
        hit_count = float(hit.get("样本总数", np.sum(mask)))
        badrate_after, badrate_reduction, _ = rejection_statistics(total, total_bad, hit_count, hit_bad)

        metrics = {
            "命中样本数": hit_count,
            "命中样本占比": float(hit.get("样本占比", hit_count / total if total > 0 else 0)),
            "命中坏样本数": hit_bad,
            "命中好样本数": float(hit.get("好样本数", hit_count - hit_bad)),
            "命中坏样本率": float(hit.get("坏样本率", hit_bad / hit_count if hit_count > 0 else 0)),
            "命中LIFT值": float(hit.get("LIFT值", 0)),
            "准确率": accuracy,
            "精确率": precision,
            "召回率": recall,
            "F1值": f1,
            "真正例": int(tp),
            "假正例": int(fp),
            "真负例": int(tn),
            "假负例": int(fn),
            "拦截后坏样本率": float(badrate_after),
            "坏账改善": float(badrate_reduction),
        }

        # 金额指标另行调用统一规则报告，不能与上方件数分母混算。
        if amount is not None:
            amount_data, amount_column = bind_column(data, amount, "amount")
            amount_report = rule.report(datasets=amount_data, target=y.to_numpy(), amount=amount_column, margins=True)
            amount_hit = amount_report.loc[amount_report["分箱"] == "命中"].iloc[0]
            total_amount = float(amount_report.loc[amount_report["分箱"] == "合计", "样本总数"].iloc[0])
            selected_amount = float(amount_hit["样本总数"])
            bad_amount = float(amount_hit["坏样本数"])
            metrics.update(
                {
                    "总金额": total_amount,
                    "命中金额": selected_amount,
                    "命中金额占比": selected_amount / total_amount if total_amount > 0 else 0,
                    "命中损失率": bad_amount / selected_amount if selected_amount > 0 else 0,
                    "命中金额LIFT值": float(amount_hit["LIFT值"]),
                }
            )

        return metrics

    def evaluate_rules(
        self,
        rules: List,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """批量评估规则.

        :param rules: 规则列表
        :type rules: List
        :param X_train: 训练集特征
        :type X_train: pd.DataFrame
        :param y_train: 训练集标签
        :type y_train: pd.Series
        :param X_test: 默认值为 ``None`` 。测试集特征
        :type X_test: Optional[pd.DataFrame]
        :param y_test: 默认值为 ``None`` 。测试集标签
        :type y_test: Optional[pd.Series]
        :return: 评估结果DataFrame
        """
        tasks = [
            (
                self,
                ordinal,
                rule,
                X_train,
                y_train,
                X_test,
                y_test,
                dict(kwargs),
            )
            for ordinal, rule in enumerate(rules)
        ]
        results = self._parallel_execute(
            _rule_metrics_worker,
            tasks,
            task_labels=[f"规则 {ordinal}" for ordinal in range(len(rules))],
            default_backend="threading",
            has_parallel_children=False,
            workload=_mining_workload(
                X_train,
                len(tasks),
                operation="规则指标批量评估",
                cost_per_item=10.0,
            ),
        )
        return pd.DataFrame(results)

    def calculate_ks(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算KS统计量.

        使用统一的KS计算。

        :param y_true: 真实标签
        :type y_true: np.ndarray
        :param y_score: 预测分数
        :type y_score: np.ndarray
        :return: KS值
        """
        return ks(y_true, y_score)

    def calculate_iv(self, feature: pd.Series, target: pd.Series, max_n_bins: int = 10) -> float:
        """计算IV值（信息价值）.

        使用统一的IV计算。

        :param feature: 特征值
        :type feature: pd.Series
        :param target: 目标变量
        :type target: pd.Series
        :param max_n_bins: 默认值为 ``10`` 。分箱数
        :type max_n_bins: int
        :return: IV值
        """
        return iv(target, feature, max_n_bins=max_n_bins)

    def calculate_gini(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        """计算Gini系数.

        使用统一的Gini计算。

        :param y_true: 真实标签
        :type y_true: np.ndarray
        :param y_score: 预测分数
        :type y_score: np.ndarray
        :return: Gini系数
        """
        return gini(y_true, y_score)


def calculate_rule_metrics(
    rule, X: pd.DataFrame, y: pd.Series, X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """便捷函数：计算规则评估指标.

    :param rule: 规则对象
    :param X: 训练集特征
    :type X: pd.DataFrame
    :param y: 训练集标签
    :type y: pd.Series
    :param X_test: 默认值为 ``None`` 。测试集特征
    :type X_test: Optional[pd.DataFrame]
    :param y_test: 默认值为 ``None`` 。测试集标签
    :type y_test: Optional[pd.Series]
    :return: 评估指标字典
    """
    metrics = RuleMetrics()
    return metrics.evaluate_rule(rule, X, y, X_test, y_test)


def calculate_lift_chart(y_true: pd.Series, y_score: pd.Series, n_buckets: int = 10) -> pd.DataFrame:
    """计算Lift图表数据.

    使用统一的lift_table计算。

    :param y_true: 真实标签
    :type y_true: pd.Series
    :param y_score: 预测分数
    :type y_score: pd.Series
    :param n_buckets: 默认值为 ``10`` 。分桶数
    :type n_buckets: int
    :return: Lift数据DataFrame
    """
    return metrics_lift_table(y_true, y_score, n_bins=n_buckets)
