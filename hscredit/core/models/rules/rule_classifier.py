"""规则集分类模型.

提供规则集分类功能，支持规则的组合、嵌套和灵活输出。
支持且/或逻辑组合，支持输出单规则命中结果和规则集最终结果。

统一入口: RulesClassifier

代码风格参考binning模块，遵循sklearn API规范。
"""

from typing import Union, List, Dict, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from ...rules.rule import Rule


def _check_input_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    target: str = 'target'
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """检查并处理输入数据.
    
    支持三种模式:
    1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
    2. scorecardpipeline风格: fit(df) - df包含特征列和目标列
    3. 只传入X: fit(X) - 仅用于验证特征
    
    :param X: 输入数据
    :param y: 目标变量（可选）
    :param target: 目标变量列名（scorecardpipeline风格使用）
    :return: (处理后的X, y)
    """
    # 转换numpy数组为DataFrame
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = pd.DataFrame(X)
    
    # 如果y为None且target列存在于X中，提取target
    if y is None and target in X.columns:
        y = X[target]
        X = X.drop(columns=[target])
    
    return X, y


class LogicOperator(str, Enum):
    """逻辑操作符枚举."""
    AND = "and"
    OR = "or"


@dataclass
class RuleResult:
    """规则评估结果.
    
    :param rule_id: 规则标识
    :param rule_name: 规则名称
    :param expression: 规则表达式
    :param matched: 是否命中
    :param matched_indices: 命中的样本索引
    :param matched_count: 命中样本数
    :param details: 额外详情
    """
    rule_id: str
    rule_name: str
    expression: str
    matched: bool
    matched_indices: np.ndarray = field(repr=False)
    matched_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式."""
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'expression': self.expression,
            'matched': self.matched,
            'matched_count': self.matched_count,
            'details': self.details
        }


class RuleSet:
    """规则集类.
    
    支持规则的层次化组合，可以包含单规则或嵌套规则集。
    支持且/或逻辑操作。
    
    :param name: 规则集名称，默认为"RuleSet"
    :param logic: 逻辑操作符，'and' 表示所有规则都命中才算命中，'or' 表示任一规则命中就算命中
    :param rules: 规则列表，可以是 Rule 或 RuleSet 对象
    :param weight: 规则集权重，用于最终得分计算，默认为1.0
    :param description: 规则集描述
    
    示例:
        >>> from hscredit.core.rules import Rule
        >>> from hscredit.core.models.rule_classifier import RuleSet
        >>> 
        >>> # 创建单规则
        >>> rule1 = Rule("age < 18", name="未成年")
        >>> rule2 = Rule("income > 100000", name="高收入")
        >>> 
        >>> # 创建规则集（且关系）
        >>> rule_set = RuleSet(
        ...     name="高风险用户",
        ...     logic="and",
        ...     rules=[rule1, rule2],
        ...     description="年龄小且收入高"
        ... )
        >>> 
        >>> # 嵌套规则集
        >>> inner_set = RuleSet(name="子规则集", logic="or", rules=[rule1])
        >>> outer_set = RuleSet(name="外层规则集", logic="and", 
        ...                     rules=[inner_set, rule2])
    """
    
    def __init__(
        self,
        name: str = "RuleSet",
        logic: Union[str, LogicOperator] = LogicOperator.AND,
        rules: Optional[List[Union[Rule, 'RuleSet']]] = None,
        weight: float = 1.0,
        description: str = ""
    ):
        self.name = name
        self.logic = LogicOperator(logic.lower()) if isinstance(logic, str) else logic
        self.rules = rules or []
        self.weight = weight
        self.description = description
        self._rule_id = self._generate_id()
        
    def _generate_id(self) -> str:
        """生成规则集唯一标识."""
        import hashlib
        content = f"{self.name}_{self.logic}_{len(self.rules)}_{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def add_rule(self, rule: Union[Rule, 'RuleSet']) -> 'RuleSet':
        """添加规则到规则集.
        
        :param rule: Rule 或 RuleSet 对象
        :return: self，支持链式调用
        """
        self.rules.append(rule)
        return self
    
    def remove_rule(self, index: int) -> 'RuleSet':
        """移除指定索引的规则.
        
        :param index: 规则索引
        :return: self，支持链式调用
        """
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
        return self
    
    def evaluate(
        self, 
        X: pd.DataFrame,
        return_details: bool = True
    ) -> Tuple[np.ndarray, List[RuleResult]]:
        """评估规则集.
        
        :param X: 输入数据
        :param return_details: 是否返回详细结果
        :return: (整体命中结果, 各规则详细结果列表)
        """
        if len(self.rules) == 0:
            n_samples = len(X)
            return np.zeros(n_samples, dtype=bool), []
        
        n_samples = len(X)
        all_results = []
        
        # 收集所有规则的命中结果
        rule_matches = []
        
        for i, rule in enumerate(self.rules):
            if isinstance(rule, RuleSet):
                # 嵌套规则集 - 递归评估
                matches, sub_results = rule.evaluate(X, return_details)
                rule_matches.append(matches)
                
                if return_details:
                    all_results.append(RuleResult(
                        rule_id=rule._rule_id,
                        rule_name=rule.name,
                        expression=f"RuleSet({rule.name}, logic={rule.logic.value})",
                        matched=matches.any(),
                        matched_indices=np.where(matches)[0],
                        matched_count=matches.sum(),
                        details={
                            'type': 'ruleset',
                            'logic': rule.logic.value,
                            'weight': rule.weight,
                            'description': rule.description,
                            'sub_results': sub_results
                        }
                    ))
            elif isinstance(rule, Rule):
                # 单规则
                matches = rule.predict(X).values
                rule_matches.append(matches)
                
                if return_details:
                    all_results.append(RuleResult(
                        rule_id=f"{self._rule_id}_rule_{i}",
                        rule_name=getattr(rule, 'name', None) or rule.expr,
                        expression=rule.expr,
                        matched=matches.any(),
                        matched_indices=np.where(matches)[0],
                        matched_count=matches.sum(),
                        details={
                            'type': 'rule',
                            'weight': getattr(rule, 'weight', 1.0),
                            'description': getattr(rule, 'description', '')
                        }
                    ))
        
        # 根据逻辑操作符合并结果
        if self.logic == LogicOperator.AND:
            # 且逻辑：所有规则都命中才算命中
            if rule_matches:
                final_result = np.all(rule_matches, axis=0)
            else:
                final_result = np.ones(n_samples, dtype=bool)
        else:
            # 或逻辑：任一规则命中就算命中
            if rule_matches:
                final_result = np.any(rule_matches, axis=0)
            else:
                final_result = np.zeros(n_samples, dtype=bool)
        
        return final_result, all_results
    
    def get_all_rules(self, flatten: bool = False) -> List[Union[Rule, 'RuleSet']]:
        """获取所有规则.
        
        :param flatten: 是否扁平化返回（展开嵌套规则集）
        :return: 规则列表
        """
        if not flatten:
            return self.rules.copy()
        
        # 扁平化展开
        flat_rules = []
        for rule in self.rules:
            if isinstance(rule, RuleSet):
                flat_rules.extend(rule.get_all_rules(flatten=True))
            else:
                flat_rules.append(rule)
        return flat_rules
    
    def __repr__(self) -> str:
        return f"RuleSet(name='{self.name}', logic='{self.logic.value}', n_rules={len(self.rules)})"
    
    def __and__(self, other: Union[Rule, 'RuleSet']) -> 'RuleSet':
        """与另一个规则或规则集组合（且逻辑）."""
        return RuleSet(
            name=f"{self.name}_AND_{getattr(other, 'name', 'Other')}",
            logic=LogicOperator.AND,
            rules=[self, other],
            description=f"{self.name} 且 {getattr(other, 'name', str(other))}"
        )
    
    def __or__(self, other: Union[Rule, 'RuleSet']) -> 'RuleSet':
        """与另一个规则或规则集组合（或逻辑）."""
        return RuleSet(
            name=f"{self.name}_OR_{getattr(other, 'name', 'Other')}",
            logic=LogicOperator.OR,
            rules=[self, other],
            description=f"{self.name} 或 {getattr(other, 'name', str(other))}"
        )


class RulesClassifier(BaseEstimator, ClassifierMixin):
    """规则分类器 - 统一入口.
    
    支持传入单规则或规则集（支持嵌套）进行分类预测。
    最外层支持选择且/或逻辑组合，内层规则逻辑由各自的RuleSet定义。
    支持输出单规则命中结果或规则集最终结果，支持输出规则命中原因。
    
    代码风格参考 binning 模块。
    
    :param rules: 规则或规则集列表/单个对象，可以是 Rule、RuleSet 或混合
    :param logic: 最外层逻辑操作符，'and' 或 'or'，默认为 'or'
    :param output_mode: 输出模式
        - 'final': 只输出最终分类结果
        - 'individual': 输出每条规则的命中结果
        - 'both': 同时输出最终结果和单规则结果
        - 'reason': 输出带命中原因的结果
    :param weights: 各规则/规则集的权重列表，用于加权投票，默认为None（等权重）
    :param threshold: 分类阈值，用于概率输出，默认为0.5
    :param target: 目标变量列名，用于scorecardpipeline风格，默认为'target'
    :param verbose: 是否输出详细信息，默认为False
    
    属性:
    - classes_: 类别标签
    - n_features_in_: 输入特征数
    - feature_names_in_: 特征名称
    
    示例:
        >>> from hscredit.core.rules import Rule
        >>> from hscredit.core.models import RulesClassifier, RuleSet
        >>> 
        >>> # 方式1：使用单规则列表
        >>> rules = [
        ...     Rule("age < 18", name="未成年", description="用户未成年"),
        ...     Rule("income > 100000", name="高收入", description="月收入超过10万"),
        ...     Rule("credit_score < 500", name="低信用分", description="信用分低于500")
        ... ]
        >>> clf = RulesClassifier(rules=rules, logic='or', output_mode='both')
        >>> clf.fit(X_train)
        >>> final, individual = clf.predict(X_test)
        >>> 
        >>> # 方式2：使用规则集（支持嵌套）
        >>> high_risk = RuleSet(
        ...     name="高风险",
        ...     logic="and",
        ...     rules=[
        ...         Rule("age < 25", name="年轻"),
        ...         Rule("debt_ratio > 0.6", name="高负债")
        ...     ],
        ...     description="年轻且高负债"
        ... )
        >>> medium_risk = RuleSet(
        ...     name="中风险",
        ...     logic="or",
        ...     rules=[
        ...         Rule("credit_score < 550"),
        ...         Rule("employment_years < 1")
        ...     ]
        ... )
        >>> clf = RulesClassifier(
        ...     rules=[high_risk, medium_risk],  # 外层使用 or 逻辑
        ...     logic='or',
        ...     output_mode='reason'
        ... )
        >>> clf.fit(X_train)
        >>> result, reasons = clf.predict(X_test, return_reason=True)
        >>> 
        >>> # 方式3：混合使用单规则和规则集
        >>> nested_rules = RuleSet(
        ...     name="嵌套规则",
        ...     logic="and",
        ...     rules=[
        ...         Rule("age > 60", name="高龄"),
        ...         RuleSet(  # 内层规则集
        ...             name="健康风险",
        ...             logic="or",
        ...             rules=[
        ...                 Rule("income < 3000"),
        ...                 Rule("debt_ratio > 0.5")
        ...             ]
        ...         )
        ...     ]
        ... )
        >>> clf = RulesClassifier(rules=[nested_rules, Rule("fraud_flag == 1")])
    """
    
    def __init__(
        self,
        rules: Optional[Union[Rule, RuleSet, List[Union[Rule, RuleSet]]]] = None,
        logic: Union[str, LogicOperator] = LogicOperator.OR,
        output_mode: str = 'final',
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
        target: str = 'target',
        verbose: bool = False
    ):
        # 统一处理为列表
        if rules is None:
            self.rules = []
        elif isinstance(rules, (Rule, RuleSet)):
            self.rules = [rules]
        else:
            self.rules = list(rules)
            
        self.logic = LogicOperator(logic.lower()) if isinstance(logic, str) else logic
        self.output_mode = output_mode
        self.weights = weights
        self.threshold = threshold
        self.target = target
        self.verbose = verbose
        
        # 验证参数
        valid_modes = ['final', 'individual', 'both', 'reason']
        if output_mode not in valid_modes:
            raise ValueError(f"不支持的output_mode: {output_mode}，可选: {valid_modes}")
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'RulesClassifier':
        """拟合分类器（学习特征名）.
        
        支持三种模式:
        1. sklearn风格: fit(X, y) - X是特征矩阵，y是目标变量
        2. scorecardpipeline风格: fit(df) - df包含特征列和目标列（由target参数指定）
        3. 仅验证: fit(X) - 仅验证特征，不学习y
        
        :param X: 训练数据
            - sklearn风格: 特征矩阵，shape (n_samples, n_features)
            - scorecardpipeline风格: 完整数据框，包含特征列和目标列
        :param y: 目标变量（可选）
            - sklearn风格: 传入目标变量
            - scorecardpipeline风格: 不传，从X中提取target列
        :param kwargs: 其他参数（保留兼容性）
        :return: 拟合后的分类器
        """
        # 处理输入数据
        X, y = _check_input_data(X, y, self.target)
        
        # 存储特征信息
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        
        # 验证规则中使用的特征都存在
        self._validate_rules(X)
        
        # 设置类别标签
        if y is not None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.array([0, 1])
        
        # 设置权重
        self._setup_weights()
        
        self._is_fitted = True
        
        if self.verbose:
            print(f"RulesClassifier 已拟合")
            print(f"  - 特征数量: {self.n_features_in_}")
            print(f"  - 顶层规则数量: {len(self.rules)}")
            print(f"  - 最外层逻辑: {self.logic.value}")
            print(f"  - 输出模式: {self.output_mode}")
            # 统计总规则数（包括嵌套）
            total_rules = self._count_total_rules()
            print(f"  - 总规则数（含嵌套）: {total_rules}")
        
        return self
    
    def _setup_weights(self) -> None:
        """设置权重."""
        if self.weights is None:
            self.weights_ = [1.0] * len(self.rules)
        else:
            if len(self.weights) != len(self.rules):
                raise ValueError(f"weights长度({len(self.weights)})必须等于rules数量({len(self.rules)})")
            self.weights_ = self.weights
    
    def _auto_init(self, X: pd.DataFrame) -> None:
        """自动初始化（无需fit即可预测）.
        
        :param X: 输入数据
        """
        # 存储特征信息
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)
        
        # 设置默认类别
        self.classes_ = np.array([0, 1])
        
        # 设置权重
        self._setup_weights()
        
        self._is_fitted = True
    
    def _count_total_rules(self) -> int:
        """统计总规则数（包括嵌套）."""
        count = 0
        for rule in self.rules:
            count += self._count_rule_recursive(rule)
        return count
    
    def _count_rule_recursive(self, rule: Union[Rule, RuleSet]) -> int:
        """递归统计规则数."""
        if isinstance(rule, RuleSet):
            total = 0
            for r in rule.rules:
                total += self._count_rule_recursive(r)
            return total
        else:
            return 1
    
    def _validate_rules(self, X: pd.DataFrame) -> None:
        """验证规则的有效性.
        
        :param X: 输入数据
        :raises ValueError: 当规则使用了不存在于数据中的特征时
        """
        available_cols = set(X.columns)
        
        def check_rule_columns(rule: Union[Rule, RuleSet]) -> None:
            if isinstance(rule, RuleSet):
                # 递归检查规则集中的所有规则
                for r in rule.rules:
                    check_rule_columns(r)
            elif isinstance(rule, Rule):
                # 获取规则使用的特征
                from ...rules.rule import get_columns_from_query
                used_cols = set(get_columns_from_query(rule.expr))
                missing = used_cols - available_cols
                if missing:
                    raise ValueError(f"规则 '{rule.expr}' 使用了不存在的特征: {missing}")
        
        for rule in self.rules:
            check_rule_columns(rule)
    
    def get_feature_importances(self, importance_type: str = 'frequency') -> pd.Series:
        """获取特征重要性（基于规则使用频率）.
        
        规则分类器通过统计特征在规则中的使用频率来计算重要性。
        
        :param importance_type: 重要性类型，默认'frequency'
            - 'frequency': 特征在规则中出现的次数
            - 'weighted': 考虑规则权重的加权频率
        :return: 特征重要性Series
        """
        check_is_fitted(self)
        
        if not hasattr(self, 'feature_names_in_') or self.feature_names_in_ is None:
            raise ValueError("未获取特征名称")
        
        # 统计特征使用频率
        feature_counts = {name: 0 for name in self.feature_names_in_}
        
        def count_features(rule: Union[Rule, RuleSet], weight: float = 1.0) -> None:
            if isinstance(rule, RuleSet):
                for r in rule.rules:
                    count_features(r, weight * rule.weight)
            elif isinstance(rule, Rule):
                from ...rules.rule import get_columns_from_query
                used_cols = get_columns_from_query(rule.expr)
                for col in used_cols:
                    if col in feature_counts:
                        if importance_type == 'frequency':
                            feature_counts[col] += 1
                        elif importance_type == 'weighted':
                            feature_counts[col] += weight
        
        for rule in self.rules:
            count_features(rule)
        
        # 创建Series
        importance_series = pd.Series(
            feature_counts,
            name='importance'
        ).sort_values(ascending=False)
        
        self._feature_importances = importance_series
        
        return importance_series
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        return_reason: bool = False
    ) -> Union[np.ndarray, pd.DataFrame, Tuple]:
        """预测.
        
        无需fit即可直接预测。
        
        :param X: 输入数据
        :param return_reason: 是否返回命中原因，当output_mode='reason'时自动为True
        :return: 根据output_mode返回不同格式的结果
            - 'final': np.ndarray，最终分类结果
            - 'individual': pd.DataFrame，每条规则的命中结果
            - 'both': tuple，(最终结果, 单规则结果)
            - 'reason': tuple，(最终结果, 命中原因列表)
        """
        # 转换输入数据
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
        
        # 如果未fit，自动初始化
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            self._auto_init(X)
        
        # 评估所有规则
        final_results = []
        individual_results = {}
        all_rule_results = []
        
        for i, rule in enumerate(self.rules):
            rule_name = getattr(rule, 'name', None) or f"Rule_{i}"
            
            if isinstance(rule, RuleSet):
                # 规则集 - 使用其自身的evaluate方法
                matches, details = rule.evaluate(X, return_details=True)
                final_results.append(matches * self.weights_[i])
                individual_results[rule_name] = matches
                all_rule_results.append({
                    'name': rule_name,
                    'type': 'ruleset',
                    'matches': matches,
                    'details': details,
                    'weight': self.weights_[i]
                })
            elif isinstance(rule, Rule):
                # 单规则
                matches = rule.predict(X).values
                final_results.append(matches * self.weights_[i])
                individual_results[rule_name] = matches
                all_rule_results.append({
                    'name': rule_name,
                    'type': 'rule',
                    'expression': rule.expr,
                    'matches': matches,
                    'weight': self.weights_[i]
                })
        
        # 根据最外层逻辑操作符合并结果
        if len(final_results) == 0:
            n_samples = len(X)
            final_result = np.zeros(n_samples, dtype=int)
        elif self.logic == LogicOperator.AND:
            # 且逻辑：所有规则都命中才算命中
            stacked = np.stack(final_results, axis=0)
            final_result = np.all(stacked > 0, axis=0).astype(int)
        else:
            # 或逻辑：加权求和
            if len(final_results) > 0:
                sum_weights = sum(self.weights_)
                prob = np.sum(final_results, axis=0) / sum_weights if sum_weights > 0 else np.zeros(len(X))
                final_result = (prob >= self.threshold).astype(int)
            else:
                final_result = np.zeros(len(X), dtype=int)
        
        # 根据输出模式返回结果
        if self.output_mode == 'final':
            return final_result
        
        elif self.output_mode == 'individual':
            return pd.DataFrame(individual_results)
        
        elif self.output_mode == 'both':
            return final_result, pd.DataFrame(individual_results)
        
        elif self.output_mode == 'reason':
            reasons = self._generate_reasons(X, all_rule_results, final_result)
            if return_reason:
                return final_result, reasons
            return final_result
        
        return final_result
    
    def _generate_reasons(
        self, 
        X: pd.DataFrame, 
        rule_results: List[Dict],
        final_result: np.ndarray
    ) -> List[List[str]]:
        """生成每条样本的命中原因.
        
        :param X: 输入数据
        :param rule_results: 规则评估结果
        :param final_result: 最终结果
        :return: 每条样本的命中原因列表
        """
        n_samples = len(X)
        sample_reasons = [[] for _ in range(n_samples)]
        
        def collect_reasons(rule_info: Dict, depth: int = 0) -> None:
            """递归收集命中原因."""
            matches = rule_info['matches']
            rule_name = rule_info['name']
            indent = "  " * depth
            
            if rule_info['type'] == 'rule':
                expr = rule_info.get('expression', '')
                for idx in np.where(matches)[0]:
                    sample_reasons[idx].append(f"{indent}命中规则 '{rule_name}': {expr}")
            else:
                # 规则集
                for idx in np.where(matches)[0]:
                    sample_reasons[idx].append(f"{indent}命中规则集 '{rule_name}'")
                
                # 递归收集子规则的原因
                details = rule_info.get('details', [])
                for detail in details:
                    if 'sub_results' in detail.details:
                        for sub in detail.details['sub_results']:
                            # 转换回字典格式
                            sub_dict = {
                                'name': sub.rule_name,
                                'type': sub.details.get('type', 'rule'),
                                'expression': sub.expression,
                                'matches': np.zeros(n_samples, dtype=bool),
                                'details': sub.details
                            }
                            # 填充matches
                            for matched_idx in sub.matched_indices:
                                sub_dict['matches'][matched_idx] = True
                            collect_reasons(sub_dict, depth + 1)
        
        for rule_info in rule_results:
            collect_reasons(rule_info)
        
        return sample_reasons
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率.
        
        无需fit即可直接预测概率。
        
        :param X: 输入数据
        :return: 概率数组，shape (n_samples, 2)
        """
        # 转换输入数据
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            else:
                X = pd.DataFrame(X)
        
        # 如果未fit，自动初始化
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            self._auto_init(X)
        
        # 评估所有规则
        results = []
        for i, rule in enumerate(self.rules):
            if isinstance(rule, RuleSet):
                matches, _ = rule.evaluate(X, return_details=False)
            elif isinstance(rule, Rule):
                matches = rule.predict(X).values
            else:
                continue
            results.append(matches * self.weights_[i])
        
        if len(results) == 0:
            n_samples = len(X)
            proba = np.zeros((n_samples, 2))
            proba[:, 0] = 1.0
            return proba
        
        # 计算加权概率
        sum_weights = sum(self.weights_)
        if sum_weights > 0:
            positive_prob = np.sum(results, axis=0) / sum_weights
        else:
            positive_prob = np.zeros(len(X))
        
        # 限制概率范围
        positive_prob = np.clip(positive_prob, 0, 1)
        
        proba = np.zeros((len(X), 2))
        proba[:, 1] = positive_prob
        proba[:, 0] = 1 - positive_prob
        
        return proba
    
    def get_rule_summary(self) -> pd.DataFrame:
        """获取规则摘要.
        
        :return: 规则摘要DataFrame
        """
        rows = []
        
        def process_rule(rule: Union[Rule, RuleSet], depth: int = 0) -> None:
            indent = "  " * depth
            if isinstance(rule, RuleSet):
                rows.append({
                    '层级': depth,
                    '类型': '规则集',
                    '名称': indent + rule.name,
                    '逻辑': rule.logic.value,
                    '表达式': '-',
                    '权重': rule.weight,
                    '描述': rule.description
                })
                for r in rule.rules:
                    process_rule(r, depth + 1)
            elif isinstance(rule, Rule):
                rows.append({
                    '层级': depth,
                    '类型': '单规则',
                    '名称': indent + (getattr(rule, 'name', None) or '-'),
                    '逻辑': '-',
                    '表达式': rule.expr,
                    '权重': getattr(rule, 'weight', 1.0),
                    '描述': getattr(rule, 'description', '')
                })
        
        for i, rule in enumerate(self.rules):
            process_rule(rule)
        
        return pd.DataFrame(rows)
    
    def add_rule(self, rule: Union[Rule, RuleSet]) -> 'RulesClassifier':
        """添加规则（拟合前使用）.
        
        :param rule: Rule 或 RuleSet 对象
        :return: self，支持链式调用
        """
        self.rules.append(rule)
        return self
    
    def __repr__(self) -> str:
        return (f"RulesClassifier(n_rules={len(self.rules)}, "
                f"logic='{self.logic.value}', output_mode='{self.output_mode}')")


# =============================================================================
# 便捷函数
# =============================================================================

def create_and_ruleset(
    rules: List[Union[Rule, RuleSet]],
    name: str = "AND_RuleSet",
    description: str = ""
) -> RuleSet:
    """创建且逻辑规则集.
    
    :param rules: 规则列表
    :param name: 规则集名称
    :param description: 规则集描述
    :return: RuleSet对象
    """
    return RuleSet(name=name, logic=LogicOperator.AND, rules=rules, description=description)


def create_or_ruleset(
    rules: List[Union[Rule, RuleSet]],
    name: str = "OR_RuleSet",
    description: str = ""
) -> RuleSet:
    """创建或逻辑规则集.
    
    :param rules: 规则列表
    :param name: 规则集名称
    :param description: 规则集描述
    :return: RuleSet对象
    """
    return RuleSet(name=name, logic=LogicOperator.OR, rules=rules, description=description)


def combine_rules(
    *rules: Union[Rule, RuleSet],
    logic: str = 'or',
    name: str = "Combined",
    description: str = ""
) -> RuleSet:
    """组合多个规则为规则集.
    
    :param rules: 多个规则或规则集
    :param logic: 逻辑操作符，'and' 或 'or'
    :param name: 规则集名称
    :param description: 规则集描述
    :return: RuleSet对象
    
    示例:
        >>> from hscredit.core.rules import Rule
        >>> rule1 = Rule("age < 18", name="未成年")
        >>> rule2 = Rule("income > 100000", name="高收入")
        >>> combined = combine_rules(rule1, rule2, logic='and', name="高风险")
    """
    return RuleSet(name=name, logic=logic, rules=list(rules), description=description)


# 保留别名以保持向后兼容
RuleSetClassifier = RulesClassifier
