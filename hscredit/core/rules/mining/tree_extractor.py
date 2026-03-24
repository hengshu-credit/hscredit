"""决策树规则提取模块.

支持从多种树模型中提取规则，包括：
- 决策树 (Decision Tree)
- 随机森林 (Random Forest)
- 卡方决策树 (Chi-square Tree)
- XGBoost/GBDT
- 孤立森林 (Isolation Forest)
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

from .base import BaseRuleMiner, RuleCondition, MinedRule
from ..rule import Rule


class TreeRuleExtractor(BaseRuleMiner):
    """树模型规则提取器.
    
    支持多种树模型的规则提取，包括决策树、随机森林、GBDT、XGBoost和孤立森林。
    
    代码风格参考hscredit的binning模块，fit方法兼容scorecardpipeline风格。
    支持通过**kwargs传入任意sklearn树模型参数。
    
    :param algorithm: 算法类型，'dt', 'rf', 'chi2', 'gbdt', 'xgb', 'isf'
    :param target: 目标变量列名，默认为'target'
    :param exclude_cols: 需要排除的列名列表
    :param max_depth: 树的最大深度，默认5
    :param min_samples_split: 分裂节点最小样本数，默认10
    :param min_samples_leaf: 叶子节点最小样本数，默认5
    :param n_estimators: 森林中树的数量，默认10
    :param max_features: 每棵树考虑的最大特征数，默认'sqrt'
    :param test_size: 测试集比例，默认0.3
    :param random_state: 随机种子，默认42
    :param feature_trends: 特征趋势字典，如{'age': 1}表示正相关
    :param chi2_threshold: 卡方分箱阈值，默认3.841
    :param kwargs: 其他树模型参数，直接传递给底层sklearn模型:
        - DecisionTreeClassifier: criterion, splitter, max_leaf_nodes, etc.
        - RandomForestClassifier: bootstrap, oob_score, class_weight, etc.
        - GradientBoostingClassifier: learning_rate, subsample, loss, etc.
        - IsolationForest: contamination, max_samples, etc.
    
    示例:
        >>> # 决策树规则提取
        >>> extractor = TreeRuleExtractor(algorithm='dt', max_depth=5)
        >>> extractor.fit(df)
        >>> rules = extractor.extract_rules()
        >>> 
        >>> # 随机森林规则提取（传入额外参数）
        >>> extractor = TreeRuleExtractor(
        ...     algorithm='rf',
        ...     n_estimators=50,
        ...     max_depth=10,
        ...     class_weight='balanced',  # sklearn参数
        ...     bootstrap=True,           # sklearn参数
        ...     oob_score=True            # sklearn参数
        ... )
        >>> extractor.fit(X, y)
        >>> rules = extractor.extract_rules()
        >>> 
        >>> # 孤立森林异常规则
        >>> extractor = TreeRuleExtractor(
        ...     algorithm='isf',
        ...     contamination=0.05,  # 异常比例
        ...     max_samples=256
        ... )
        >>> extractor.fit(X)
        >>> anomaly_rules = extractor.extract_rules()
    """
    
    VALID_ALGORITHMS = {'dt', 'rf', 'chi2', 'gbdt', 'xgb', 'isf'}
    
    def __init__(
        self,
        algorithm: str = 'dt',
        target: str = 'target',
        exclude_cols: Optional[List[str]] = None,
        max_depth: int = 5,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_estimators: int = 10,
        max_features: str = 'sqrt',
        test_size: float = 0.3,
        random_state: int = 42,
        feature_trends: Optional[Dict[str, int]] = None,
        chi2_threshold: float = 3.841,
        **kwargs
    ):
        super().__init__(target=target, exclude_cols=exclude_cols)
        
        algorithm = algorithm.lower()
        if algorithm == 'xgb':
            algorithm = 'gbdt'
            warnings.warn("'xgb'算法已弃用，请使用'gbdt'", DeprecationWarning)
        
        if algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"不支持的算法: {algorithm}，可选: {self.VALID_ALGORITHMS}")
        
        self.algorithm = algorithm
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.feature_trends = feature_trends or {}
        self.chi2_threshold = chi2_threshold
        self.model_kwargs = kwargs  # 存储额外的模型参数
        
        self.model_ = None
        self.encoders_ = {}
        self.feature_names_ = []
        self.rules_ = []
        self.is_fitted_ = False
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        **kwargs
    ) -> 'TreeRuleExtractor':
        """拟合提取器.
        
        :param X: 训练数据
        :param y: 目标变量（监督学习需要）
        :param kwargs: 额外参数，可覆盖初始化参数
        :return: self
        """
        # 更新参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.model_kwargs:
                self.model_kwargs[key] = value
        
        X, y = self._check_input_data(X, y)
        
        # 保存特征名
        self.feature_names_ = list(X.columns)
        
        # 编码类别型特征
        X_encoded = self._encode_categorical_features(X)
        
        # 初始化模型
        self.model_ = self._initialize_model()
        
        if self.algorithm == 'isf':
            # 孤立森林不需要y
            self.model_.fit(X_encoded)
            self.X_train_ = X_encoded
            self.X_ = X
        else:
            # 监督学习需要y
            if y is None:
                raise ValueError(f"算法 '{self.algorithm}' 需要目标变量y")
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            self.X_train_ = X_train
            self.X_test_ = X_test
            self.y_train_ = y_train
            self.y_test_ = y_test
            self.X_ = X
            
            # 卡方分箱预处理
            if self.algorithm == 'chi2':
                X_train = self._chi2_preprocess(X_train, y_train)
                X_test = self._chi2_preprocess(X_test, y_test, fit=False)
            
            # 训练模型
            self.model_.fit(X_train, y_train)
            
            # 保存训练后的数据
            self.X_train_ = X_train
            self.X_test_ = X_test
        
        self.is_fitted_ = True
        return self
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """编码类别型特征.
        
        :param X: 输入数据
        :return: 编码后的数据
        """
        X_encoded = X.copy()
        self.encoders_ = {}
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.encoders_[col] = le
        
        return X_encoded.fillna(0)
    
    def _initialize_model(self):
        """初始化模型，支持通过**kwargs传入任意参数."""
        # 构建基础参数字典
        base_params = {
            'random_state': self.random_state
        }
        
        if self.algorithm == 'dt':
            base_params.update({
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
            })
            # 合并额外参数
            base_params.update(self.model_kwargs)
            return DecisionTreeClassifier(**base_params)
        
        elif self.algorithm in ['rf', 'chi2']:
            base_params.update({
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'n_jobs': -1
            })
            # 合并额外参数
            base_params.update(self.model_kwargs)
            return RandomForestClassifier(**base_params)
        
        elif self.algorithm == 'gbdt':
            base_params.update({
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
            })
            # 合并额外参数
            base_params.update(self.model_kwargs)
            return GradientBoostingClassifier(**base_params)
        
        elif self.algorithm == 'isf':
            base_params.update({
                'n_estimators': self.n_estimators,
                'max_samples': min(256, 1000),  # 限制最大样本数
                'contamination': 0.1,
                'n_jobs': -1
            })
            # 合并额外参数（如contamination, max_samples等）
            base_params.update(self.model_kwargs)
            return IsolationForest(**base_params)
    
    def _chi2_preprocess(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fit: bool = True
    ) -> pd.DataFrame:
        """卡方分箱预处理.
        
        :param X: 特征数据
        :param y: 目标变量
        :param fit: 是否拟合
        :return: 分箱后的数据
        """
        from scipy.stats import chi2_contingency
        
        if fit:
            self.chi2_bins_ = {}
        
        X_binned = X.copy()
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            
            if fit:
                # 初始等频分箱
                from sklearn.preprocessing import KBinsDiscretizer
                discretizer = KBinsDiscretizer(
                    n_bins=10,
                    encode='ordinal',
                    strategy='quantile'
                )
                
                try:
                    bins = discretizer.fit_transform(X[[col]].dropna()).flatten()
                    bin_edges = discretizer.bin_edges_[0].copy()
                    
                    # 合并相似分箱
                    while len(bin_edges) > 2:
                        chi2_scores = []
                        
                        for i in range(len(bin_edges) - 2):
                            merged_bins = bins.copy()
                            merged_bins[merged_bins == i + 1] = i
                            
                            try:
                                contingency = pd.crosstab(merged_bins, y)
                                chi2_stat, _, _, _ = chi2_contingency(contingency)
                                chi2_scores.append((i, chi2_stat))
                            except Exception:
                                chi2_scores.append((i, float('inf')))
                        
                        if not chi2_scores:
                            break
                        
                        min_idx, min_chi2 = min(chi2_scores, key=lambda x: x[1])
                        
                        if min_chi2 < self.chi2_threshold:
                            bins[bins > min_idx] -= 1
                            bin_edges = np.delete(bin_edges, min_idx + 1)
                        else:
                            break
                    
                    self.chi2_bins_[col] = bin_edges
                except Exception:
                    continue
            
            # 应用分箱
            if col in getattr(self, 'chi2_bins_', {}):
                X_binned[col] = pd.cut(X[col], bins=self.chi2_bins_[col], labels=False)
        
        return X_binned.fillna(0)
    
    def extract_rules(self) -> List[Dict[str, Any]]:
        """提取规则.
        
        :return: 规则列表
        """
        self._check_fitted()
        
        if self.algorithm == 'dt':
            self.rules_ = self._extract_from_tree(self.model_, tree_id=0)
        
        elif self.algorithm in ['rf', 'chi2']:
            self.rules_ = self._extract_from_forest()
        
        elif self.algorithm == 'gbdt':
            self.rules_ = self._extract_from_gbdt()
        
        elif self.algorithm == 'isf':
            self.rules_ = self._extract_from_isolation_forest()
        
        # 过滤规则
        self.rules_ = self._filter_rules(self.rules_)
        
        # 去重
        self.rules_ = self._deduplicate_rules(self.rules_)
        
        # 计算重要性
        for rule in self.rules_:
            rule['importance'] = self._calculate_rule_importance(rule)
        
        # 排序
        self.rules_.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        return self.rules_
    
    def _extract_from_tree(
        self,
        tree_model,
        tree_id: int = 0
    ) -> List[Dict[str, Any]]:
        """从单棵树提取规则.
        
        :param tree_model: 树模型
        :param tree_id: 树ID
        :return: 规则列表
        """
        tree = tree_model.tree_
        rules = []
        
        def recurse(node_id, conditions):
            """递归遍历树."""
            if tree.feature[node_id] == -2:  # 叶子节点
                # 计算叶子节点的实际坏账率
                badrate = self._calculate_leaf_badrate(conditions)
                
                value = tree.value[node_id][0]
                total = value.sum()
                
                if total == 0:
                    return
                
                predicted_class = 1 if badrate > 0.5 else 0
                
                rule = {
                    'rule_id': len(rules),
                    'conditions': conditions.copy(),
                    'predicted_class': predicted_class,
                    'class_name': 'bad' if predicted_class == 1 else 'good',
                    'class_probability': badrate,
                    'sample_count': int(total),
                    'tree_id': tree_id
                }
                rules.append(rule)
            else:
                # 非叶子节点
                feature = self.feature_names_[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                
                # 左子树 (<=)
                left_conditions = conditions + [{
                    'feature': feature,
                    'threshold': threshold,
                    'operator': '<='
                }]
                recurse(tree.children_left[node_id], left_conditions)
                
                # 右子树 (>)
                right_conditions = conditions + [{
                    'feature': feature,
                    'threshold': threshold,
                    'operator': '>'
                }]
                recurse(tree.children_right[node_id], right_conditions)
        
        recurse(0, [])
        return rules
    
    def _extract_from_forest(self) -> List[Dict[str, Any]]:
        """从随机森林提取规则.
        
        :return: 规则列表
        """
        all_rules = []
        
        for i, tree in enumerate(self.model_.estimators_):
            tree_rules = self._extract_from_tree(tree, tree_id=i)
            all_rules.extend(tree_rules)
        
        return all_rules
    
    def _extract_from_gbdt(self) -> List[Dict[str, Any]]:
        """从GBDT提取规则.
        
        :return: 规则列表
        """
        all_rules = []
        
        for i in range(self.model_.n_estimators_):
            tree = self.model_.estimators_[i, 0]
            tree_rules = self._extract_from_tree(tree, tree_id=i)
            
            # 过滤命中样本过少的规则
            for rule in tree_rules:
                mask = self._apply_conditions(rule['conditions'], self.X_train_)
                hit_count = mask.sum()
                
                if hit_count >= 5:
                    hit_bad = self.y_train_[mask].sum()
                    badrate = hit_bad / hit_count
                    
                    if badrate >= 0.05:
                        rule['sample_count'] = int(hit_count)
                        rule['class_probability'] = badrate
                        all_rules.append(rule)
        
        return all_rules
    
    def _extract_from_isolation_forest(self) -> List[Dict[str, Any]]:
        """从孤立森林提取规则.
        
        :return: 规则列表
        """
        # 计算异常分数
        scores = self.model_.score_samples(self.X_train_)
        threshold = np.percentile(scores, 10)  # 取异常分数最低的10%
        
        anomaly_mask = scores < threshold
        
        rules = []
        
        # 从每棵树提取路径
        for tree_idx, estimator in enumerate(self.model_.estimators_):
            tree = estimator.tree_
            
            def extract_path(node_id, conditions, depth):
                if depth > 3:  # 限制深度
                    return
                
                if tree.feature[node_id] == -2:  # 叶子
                    mask = self._apply_conditions(conditions, self.X_train_)
                    hit_count = mask.sum()
                    
                    if hit_count >= 5:
                        anomaly_count = anomaly_mask[mask].sum()
                        purity = anomaly_count / hit_count
                        
                        if purity >= 0.3:  # 异常纯度要求
                            rule = {
                                'rule_id': len(rules),
                                'conditions': conditions.copy(),
                                'predicted_class': 1,
                                'class_name': 'anomaly',
                                'class_probability': purity,
                                'sample_count': int(hit_count),
                                'tree_id': tree_idx,
                                'anomaly_score': scores[mask].mean()
                            }
                            rules.append(rule)
                else:
                    feature = self.feature_names_[tree.feature[node_id]]
                    thresh = tree.threshold[node_id]
                    
                    # 左子树
                    left_cond = conditions + [{
                        'feature': feature,
                        'threshold': thresh,
                        'operator': '<='
                    }]
                    extract_path(tree.children_left[node_id], left_cond, depth + 1)
                    
                    # 右子树
                    right_cond = conditions + [{
                        'feature': feature,
                        'threshold': thresh,
                        'operator': '>'
                    }]
                    extract_path(tree.children_right[node_id], right_cond, depth + 1)
            
            extract_path(0, [], 0)
        
        return rules
    
    def _apply_conditions(
        self,
        conditions: List[Dict],
        X: pd.DataFrame
    ) -> pd.Series:
        """应用条件到数据.
        
        :param conditions: 条件列表
        :param X: 数据
        :return: 布尔掩码
        """
        mask = pd.Series(True, index=X.index)
        
        for cond in conditions:
            feature = cond['feature']
            threshold = cond['threshold']
            operator = cond['operator']
            
            if operator == '<=':
                mask &= X[feature] <= threshold
            elif operator == '>':
                mask &= X[feature] > threshold
            elif operator == '<':
                mask &= X[feature] < threshold
            elif operator == '>=':
                mask &= X[feature] >= threshold
            elif operator == '==':
                mask &= X[feature] == threshold
        
        return mask
    
    def _calculate_leaf_badrate(self, conditions: List[Dict]) -> float:
        """计算叶子节点的坏账率.
        
        :param conditions: 条件列表
        :return: 坏账率
        """
        if not hasattr(self, 'y_train_'):
            return 0.5
        
        mask = self._apply_conditions(conditions, self.X_train_)
        hit_count = mask.sum()
        
        if hit_count == 0:
            return 0.0
        
        hit_bad = self.y_train_[mask].sum()
        return hit_bad / hit_count
    
    def _filter_rules(self, rules: List[Dict]) -> List[Dict]:
        """根据feature_trends过滤规则.
        
        :param rules: 规则列表
        :return: 过滤后的规则列表
        """
        if not self.feature_trends:
            return rules
        
        filtered = []
        
        for rule in rules:
            valid = True
            
            for cond in rule['conditions']:
                feature = cond['feature']
                operator = cond['operator']
                
                if feature in self.feature_trends:
                    trend = self.feature_trends[feature]
                    
                    # 正相关：只保留>方向的规则
                    if trend == 1 and operator in ['<=', '<']:
                        valid = False
                        break
                    
                    # 负相关：只保留<=方向的规则
                    if trend == -1 and operator in ['>', '>=']:
                        valid = False
                        break
            
            if valid:
                filtered.append(rule)
        
        return filtered
    
    def _deduplicate_rules(
        self,
        rules: List[Dict],
        similarity_threshold: float = 0.9
    ) -> List[Dict]:
        """规则去重.
        
        :param rules: 规则列表
        :param similarity_threshold: 相似度阈值
        :return: 去重后的规则列表
        """
        if not rules:
            return []
        
        unique_rules = []
        
        for rule in rules:
            # 生成规则签名
            conditions = rule['conditions']
            signature = '|'.join([
                f"{c['feature']}{c['operator']}{c['threshold']:.4f}"
                for c in sorted(conditions, key=lambda x: x['feature'])
            ])
            
            # 检查是否已存在相似规则
            is_duplicate = False
            for existing in unique_rules:
                existing_sig = '|'.join([
                    f"{c['feature']}{c['operator']}{c['threshold']:.4f}"
                    for c in sorted(existing['conditions'], key=lambda x: x['feature'])
                ])
                
                # 计算相似度
                if signature == existing_sig:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_rules.append(rule)
        
        return unique_rules
    
    def _calculate_rule_importance(self, rule: Dict) -> float:
        """计算规则重要性.
        
        :param rule: 规则字典
        :return: 重要性分数
        """
        sample_count = rule.get('sample_count', 0)
        class_prob = rule.get('class_probability', 0)
        
        # 坏样本权重更高
        weight = 2.0 if rule.get('class_name') in ['bad', 'anomaly'] else 1.0
        
        return sample_count * class_prob * weight
    
    def get_rules(
        self,
        top_n: int = 100,
        min_samples: int = 10,
        min_confidence: float = 0.0
    ) -> List[MinedRule]:
        """获取挖掘的规则对象.
        
        :param top_n: 返回规则数量
        :param min_samples: 最小样本数
        :param min_confidence: 最小置信度
        :return: MinedRule列表
        """
        if not self.rules_:
            self.extract_rules()
        
        mined_rules = []
        
        for rule in self.rules_[:top_n]:
            if rule['sample_count'] < min_samples:
                continue
            if rule['class_probability'] < min_confidence:
                continue
            
            conditions = [
                RuleCondition(
                    c['feature'],
                    c['threshold'],
                    c['operator']
                )
                for c in rule['conditions']
            ]
            
            mined_rule = MinedRule(
                conditions=conditions,
                metric_score=rule.get('importance', 0),
                description=self._rule_to_string(rule),
                metadata=rule
            )
            mined_rules.append(mined_rule)
        
        return mined_rules
    
    def get_rule_objects(
        self,
        top_n: int = 100,
        min_samples: int = 10
    ) -> List[Rule]:
        """获取Rule对象列表.
        
        :param top_n: 规则数量
        :param min_samples: 最小样本数
        :return: Rule对象列表
        """
        mined_rules = self.get_rules(top_n, min_samples)
        return [r.to_rule_object() for r in mined_rules]
    
    def get_rules_dataframe(self, top_n: int = 100) -> pd.DataFrame:
        """获取规则DataFrame.
        
        :param top_n: 规则数量
        :return: 规则DataFrame
        """
        if not self.rules_:
            self.extract_rules()
        
        data = []
        
        for rule in self.rules_[:top_n]:
            rule_str = self._rule_to_string(rule)
            
            data.append({
                'rule_id': rule['rule_id'],
                'rule': rule_str,
                'predicted_class': rule['predicted_class'],
                'class_name': rule['class_name'],
                'class_probability': rule['class_probability'],
                'sample_count': rule['sample_count'],
                'importance': rule.get('importance', 0),
                'tree_id': rule.get('tree_id', 0)
            })
        
        return pd.DataFrame(data)
    
    def _rule_to_string(self, rule: Dict) -> str:
        """将规则转换为字符串.
        
        :param rule: 规则字典
        :return: 规则字符串
        """
        conditions = rule['conditions']
        
        if not conditions:
            return "True"
        
        parts = []
        for c in conditions:
            parts.append(f"{c['feature']} {c['operator']} {c['threshold']:.4f}")
        
        return " AND ".join(parts)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性.
        
        :return: 特征重要性DataFrame
        """
        if not hasattr(self.model_, 'feature_importances_'):
            raise ValueError(f"算法 '{self.algorithm}' 不支持特征重要性")
        
        importance = self.model_.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def _check_fitted(self):
        """检查是否已拟合."""
        if not self.is_fitted_:
            raise RuntimeError("请先调用fit()方法")
