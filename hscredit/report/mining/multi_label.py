"""多标签联合规则挖掘.

支持同时针对多个标签（如 MOB3@30 和 MOB6@30）挖掘规则，
并分析规则在不同标签下的有效性差异。
"""

import copy

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Any
from .base import BaseRuleMiner, _mining_workload
from .single_feature import SingleFeatureRuleMiner


def _multi_label_mining_worker(task):
    """针对单个标签运行完整的单特征候选挖掘。"""
    miner, data, features, label, candidate_min_samples = task
    labels = miner._resolved_labels()
    child_n_jobs = 1 if miner.n_jobs is None or miner.n_jobs in (1, 1.0) else -1
    child = SingleFeatureRuleMiner(
        target=label,
        max_n_bins=miner.n_bins,
        min_lift=1.0,
        min_samples=candidate_min_samples,
        exclude_cols=[lb for lb in labels if lb != label] + (miner.exclude_cols or []),
        n_jobs=child_n_jobs,
        parallel_backend=miner.parallel_backend,
        parallel_config=miner.parallel_config,
    )
    child.fit(data[features + [label]])
    rules = child.get_rules(min_lift=1.0, min_samples=candidate_min_samples)
    ordered_rules = []
    for rule in rules:
        if isinstance(rule, dict):
            expression = rule.get("规则", rule.get("rule", rule.get("expression", "")))
        else:
            expression = getattr(rule, "expr", "")
        if expression:
            ordered_rules.append((expression, rule))
    return label, ordered_rules


class MultiLabelRuleMiner(BaseRuleMiner):
    """多标签规则挖掘器.

    支持同时针对多个标签挖掘规则，并分析规则在不同标签下的有效性差异。

    典型应用场景：
    - 长短期标签都有效的强规则（稳定拒绝规则）
    - 仅短期标签有效（可能是偶发风险，谨慎使用）
    - 仅长期标签有效（长期风险，可做预警规则）
    - 两标签均无效（噪声规则，丢弃）

    :param labels: 标签列名列表，如 ['mob3_30', 'mob6_30']
    :param label_names: 标签中文名列表，如 ['短期标签(MOB3@30)', '长期标签(MOB6@30)']
    :param min_support: 最小支持度（规则覆盖率），默认 0.02
    :param min_lift: 最小 LIFT 值，默认 1.5
    :param max_rules: 每个特征最大规则数，默认 10
    :param n_bins: 数值特征分箱数，默认 10
    :param exclude_cols: 需要排除的列名列表

    **参考样例**

    >>> miner = MultiLabelRuleMiner(
    ...     labels=['mob3_30', 'mob6_30'],  # 同时分析两个标签
    ...     label_names=['短期标签(MOB3@30)', '长期标签(MOB6@30)'],
    ...     min_support=0.02,   # 规则覆盖率>2%
    ...     min_lift=1.5,      # LIFT值>1.5
    ... )
    >>> miner.fit(df, features=['age', 'income', 'credit_score'])
    >>> rules = miner.get_rules(effectiveness='both')  # 获取两标签均有效的规则
    >>> report = miner.get_effectiveness_matrix()  # 获取规则有效性矩阵
    """

    def __setattr__(self, name, value):
        """仅跟踪构造完成后的外部 target 赋值，将其切换为显式模式。"""
        if name == 'target' and self.__dict__.get('_track_target_assignment', False):
            object.__setattr__(self, '_target_is_auto', False)
        object.__setattr__(self, name, value)

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        label_names: Optional[List[str]] = None,
        min_support: float = 0.02,
        min_lift: float = 1.5,
        max_rules: int = 10,
        n_bins: int = 10,
        exclude_cols: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target=labels[0] if labels else 'target',
            exclude_cols=exclude_cols,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self._auto_target_value = self.target
        self._target_is_auto = True
        self._track_target_assignment = True
        self.labels = labels
        self.label_names = label_names
        self.min_support = min_support
        self.min_lift = min_lift
        self.max_rules = max_rules
        self.n_bins = n_bins
        self._rules: List[Dict[str, Any]] = []

    def _resolved_labels(self) -> List[str]:
        """返回内部使用的标签列表，不改写公开构造参数。"""
        return list(self.labels) if self.labels is not None else []

    def _resolved_label_names(self) -> List[str]:
        """返回内部展示名称；未提供时按标签名展示。"""
        labels = self._resolved_labels()
        return list(self.label_names) if self.label_names is not None else labels

    def _set_auto_target(self, target: str) -> None:
        """更新自动派生 target，不触发外部显式赋值标记。"""
        object.__setattr__(self, 'target', target)
        object.__setattr__(self, '_auto_target_value', target)
        object.__setattr__(self, '_target_is_auto', True)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y=None,
        features: Optional[List[str]] = None,
    ) -> 'MultiLabelRuleMiner':
        """拟合多标签规则挖掘器.

        :param X: 输入数据 DataFrame，须包含 labels 指定的标签列
        :param y: 忽略
        :param features: 需要挖掘的特征列表，为 None 时自动选择数值特征
        :return: self
        """
        working = copy.deepcopy(self)
        working._rules = []

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X 必须为 DataFrame，且须包含标签列")

        df = X.copy()
        labels = working._resolved_labels()
        label_names = working._resolved_label_names()
        if getattr(working, '_target_is_auto', True):
            working._set_auto_target(labels[0] if labels else 'target')

        # 验证标签列存在
        missing_labels = [lb for lb in labels if lb not in df.columns]
        if missing_labels:
            raise ValueError(f"标签列缺失: {missing_labels}")

        # 确定特征列
        if features is None:
            exclude = set(labels) | set(working.exclude_cols or [])
            features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        # 对每个标签独立运行单特征规则挖掘
        label_rules = {}  # {label: {rule_expr: rule}}
        all_rule_exprs = []
        seen_rule_exprs = set()

        # 候选规则放宽 LIFT/样本量约束，最终是否有效由各标签下的 min_lift 单独判定
        candidate_min_samples = max(1, int(working.min_support * len(df)))
        tasks = [
            (working, df, list(features), label, candidate_min_samples)
            for label in labels
        ]
        mined_by_label = working._parallel_execute(
            _multi_label_mining_worker,
            tasks,
            task_labels=labels,
            default_backend="threading",
            has_parallel_children=True,
            workload=_mining_workload(
                df,
                len(tasks),
                operation="多标签规则挖掘",
                cost_per_item=20.0,
                has_parallel_children=True,
            ),
        )
        for label, rules in mined_by_label:
            label_rules[label] = {}
            for expression, rule in rules:
                label_rules[label][expression] = rule
                if expression not in seen_rule_exprs:
                    seen_rule_exprs.add(expression)
                    all_rule_exprs.append(expression)

        # 合并规则，为每条规则计算各标签的指标
        merged_rules = []
        for expr in all_rule_exprs:
            mask = df.eval(expr)

            n_match = mask.sum()
            support = n_match / len(df)
            if support < working.min_support:
                continue

            rule_info = {
                '规则': expr,
                '覆盖样本数': int(n_match),
                '覆盖率': round(support * 100, 2),
            }

            effective_labels = []
            for i, label in enumerate(labels):
                lname = label_names[i] if i < len(label_names) else label
                overall_rate = df[label].mean()
                rule_rate = df.loc[mask, label].mean() if n_match > 0 else 0
                lift = rule_rate / overall_rate if overall_rate > 0 else 0

                rule_info[f'{lname}_坏率'] = round(rule_rate * 100, 2)
                rule_info[f'{lname}_LIFT'] = round(lift, 4)
                rule_info[f'{lname}_有效'] = lift >= working.min_lift

                if lift >= working.min_lift:
                    effective_labels.append(lname)

            # 判断规则分类
            if len(effective_labels) == len(labels):
                rule_info['规则类型'] = '强规则（全标签有效）'
                rule_info['建议'] = '稳定拒绝规则'
            elif len(effective_labels) > 0:
                rule_info['规则类型'] = f'部分有效（{",".join(effective_labels)}）'
                rule_info['建议'] = '谨慎使用/预警规则'
            else:
                rule_info['规则类型'] = '无效规则'
                rule_info['建议'] = '放弃'

            merged_rules.append(rule_info)

        # 按第一个标签的 LIFT 降序排序
        if labels:
            first_lift_col = (
                f'{label_names[0]}_LIFT'
                if label_names
                else f'{labels[0]}_LIFT'
            )
            merged_rules.sort(key=lambda r: r.get(first_lift_col, 0), reverse=True)

        working._rules = merged_rules
        working._is_fitted = True
        self._commit_fitted_state(working)
        return self

    def get_rules(
        self,
        effectiveness: str = 'any',
        min_lift_per_label: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """获取筛选后的规则表.

        :param effectiveness: 筛选条件
            - 'both': 所有标签均有效
            - 'any': 任一标签有效（默认）
            - 'short_only': 仅第一个标签有效
            - 'long_only': 仅最后一个标签有效
            - 'all': 不做筛选
        :param min_lift_per_label: 覆盖最小 LIFT 阈值，为 None 时使用 self.min_lift
        :param top_n: 返回前 N 条规则
        :return: 规则 DataFrame
        """
        if not self._is_fitted:
            raise ValueError("请先调用 fit()")

        min_lift = self.min_lift if min_lift_per_label is None else min_lift_per_label
        rules = self._rules.copy()
        labels = self._resolved_labels()
        label_names = self._resolved_label_names()

        lift_columns = [
            f'{label_names[i]}_LIFT' if i < len(label_names) else f'{label}_LIFT'
            for i, label in enumerate(labels)
        ]

        def effective(rule):
            return [rule.get(column, 0) >= min_lift for column in lift_columns]

        if effectiveness == 'both':
            rules = [rule for rule in rules if all(effective(rule))]
        elif effectiveness == 'any':
            rules = [rule for rule in rules if any(effective(rule))]
        elif effectiveness == 'short_only':
            rules = [rule for rule in rules if effective(rule)[0]]
        elif effectiveness == 'long_only':
            rules = [rule for rule in rules if effective(rule)[-1]]

        if top_n:
            rules = rules[:top_n]

        return pd.DataFrame(rules)

    def get_effectiveness_matrix(self) -> pd.DataFrame:
        """规则有效性矩阵：行=规则，列=各标签，格=LIFT值.

        :return: 有效性矩阵 DataFrame
        """
        if not self._is_fitted:
            raise ValueError("请先调用 fit()")

        labels = self._resolved_labels()
        label_names = self._resolved_label_names()
        rows = []
        for rule in self._rules:
            row = {'规则': rule['规则']}
            for i, label in enumerate(labels):
                lname = label_names[i] if i < len(label_names) else label
                row[f'{lname}_LIFT'] = rule.get(f'{lname}_LIFT', 0)
            row['规则类型'] = rule.get('规则类型', '')
            rows.append(row)

        return pd.DataFrame(rows)

    def get_report(self) -> pd.DataFrame:
        """获取完整规则分析报告.

        :return: 含规则分类和业务解读的 DataFrame
        """
        return self.get_rules(effectiveness='all')
