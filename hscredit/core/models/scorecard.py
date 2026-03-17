# -*- coding: utf-8 -*-
"""
评分卡模型.

将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .logistic_regression import LogisticRegression


class ScoreCard(BaseEstimator, TransformerMixin):
    """评分卡模型.

    将逻辑回归模型转换为评分卡，支持评分卡输出、保存和导出等功能。

    **参数**

    :param pdo: odds 每增加 rate 倍时减少的分值，默认 60
    :param rate: 倍率，默认 2
    :param base_odds: 基础 odds，默认 35
        - 通常根据业务经验设置的基础比率（违约概率/正常概率）
        - 估算方法：（1-样本坏客户占比）/坏客户占比
        - 例如：35:1 => 0.972 => 坏样本率 2.8%
    :param base_score: 基础 odds 对应的分数，默认 750
    :param lr_model: 预训练的逻辑回归模型，可选
    :param combiner: 特征分箱器，可选
    :param transer: WOE 转换器，可选
    :param calculate_stats: 是否计算统计信息，默认 True

    **属性**

    :ivar factor: 补偿值 B，计算方式：pdo / ln(rate)
    :ivar offset: 刻度 A，计算方式：base_score - B * ln(base_odds)
    :ivar rules_: 评分卡规则字典，包含每个特征的分箱和对应分数
    :ivar base_effect_: 每个特征的基础效应分数

    **参考样例**

    基本使用::

        >>> from hscredit.core.models import ScoreCard
        >>> from hscredit.core.binning import DecisionTreeBinner
        >>> from hscredit.core.encoders import WOEEncoder
        >>> 
        >>> # 方式1：从零开始训练
        >>> scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
        >>> scorecard.fit(X_woe, y)
        >>> 
        >>> # 方式2：使用预训练的 LR 模型
        >>> lr = LogisticRegression(calculate_stats=True)
        >>> lr.fit(X_woe, y)
        >>> scorecard = ScoreCard(lr_model=lr)
        >>> 
        >>> # 预测评分
        >>> scores = scorecard.predict(X)
        >>> 
        >>> # 输出评分卡配置
        >>> scale_df = scorecard.scorecard_scale()
        >>> print(scale_df)
        >>> 
        >>> # 输出评分卡
        >>> points_df = scorecard.scorecard_points()
        >>> print(points_df)

    参考:
        - toad.ScoreCard
        - scorecardpipeline.ScoreCard
        - optbinning.Scorecard
    """

    def __init__(
        self,
        pdo: float = 60,
        rate: float = 2,
        base_odds: float = 35,
        base_score: float = 750,
        lr_model: Optional[LogisticRegression] = None,
        combiner: Optional[Any] = None,
        transer: Optional[Any] = None,
        calculate_stats: bool = True,
        **kwargs
    ):
        self.pdo = pdo
        self.rate = rate
        self.base_odds = base_odds
        self.base_score = base_score
        self.lr_model = lr_model
        self.combiner = combiner
        self.transer = transer
        self.calculate_stats = calculate_stats
        
        # 计算评分转换参数
        self.factor = pdo / np.log(rate)
        self.offset = base_score - self.factor * np.log(base_odds)
        
        # 初始化属性
        self.rules_ = {}
        self.base_effect_ = None
        self._feature_names = None

    @property
    def coef_(self) -> np.ndarray:
        """获取逻辑回归系数."""
        check_is_fitted(self)
        return self.lr_model_.coef_[0]

    @property
    def intercept_(self) -> float:
        """获取逻辑回归截距."""
        check_is_fitted(self)
        return self.lr_model_.intercept_[0]

    @property
    def n_features_(self) -> int:
        """获取非零系数特征数量."""
        return (self.coef_ != 0).sum()

    @property
    def feature_names_(self) -> list:
        """获取特征名列表."""
        if self._feature_names is None:
            self._feature_names = list(self.rules_.keys())
        return self._feature_names

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sample_weight: Optional[np.ndarray] = None
    ) -> 'ScoreCard':
        """训练评分卡模型.

        :param X: 训练数据（WOE 转换后）
        :param y: 目标变量
        :param sample_weight: 样本权重
        :return: self
        """
        # 转换为 DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self._feature_names = X.columns.tolist()

        # 如果没有预训练的 LR 模型，则训练一个新的
        if self.lr_model is None:
            self.lr_model_ = LogisticRegression(
                calculate_stats=self.calculate_stats,
                max_iter=1000
            )
        else:
            self.lr_model_ = self.lr_model

        # 训练 LR 模型
        if not hasattr(self.lr_model_, 'coef_'):
            self.lr_model_.fit(X, y, sample_weight=sample_weight)

        # 生成评分卡规则
        self._generate_rules(X)

        # 计算基础效应（用于解释评分）
        sub_scores = self._woe_to_score(X)
        self.base_effect_ = pd.Series(
            np.median(sub_scores, axis=0),
            index=self.feature_names_
        )

        return self

    def _generate_rules(self, X: pd.DataFrame):
        """生成评分卡规则.

        :param X: 训练数据
        """
        self.rules_ = {}
        
        for i, col in enumerate(self.feature_names_):
            coef = self.coef_[i]
            
            # 获取该特征的 WOE 值
            if self.transer is not None and col in self.transer:
                # 从 WOE 转换器获取
                woe_values = self.transer[col]['woe']
                bins = self.transer[col]['bins']
            else:
                # 从训练数据推断
                unique_woe = X[col].dropna().unique()
                woe_values = sorted(unique_woe)
                bins = None

            # 计算每个 WOE 对应的分数
            scores = []
            for woe in woe_values:
                score = self._woe_to_point(woe, coef)
                scores.append(score)

            self.rules_[col] = {
                'bins': bins if bins is not None else woe_values,
                'woe': woe_values,
                'scores': np.array(scores),
                'coef': coef
            }

    def _woe_to_point(self, woe: float, coef: float) -> float:
        """将 WOE 值转换为分数.

        :param woe: WOE 值
        :param coef: 逻辑回归系数
        :return: 分数
        """
        # Score = A - B * (intercept + coef * woe)
        # 但这里只计算特征贡献部分：-B * coef * woe
        return -self.factor * coef * woe

    def _woe_to_score(self, X: pd.DataFrame) -> np.ndarray:
        """将 WOE 数据转换为分数矩阵.

        :param X: WOE 数据
        :return: 分数矩阵
        """
        scores = np.zeros((X.shape[0], len(self.feature_names_)))
        
        for i, col in enumerate(self.feature_names_):
            if col in X.columns:
                coef = self.coef_[i]
                scores[:, i] = -self.factor * coef * X[col].values
        
        return scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测评分.

        :param X: 原始数据或 WOE 转换后的数据
        :return: 评分数组
        """
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 如果有 combiner 和 transer，则先转换
        if self.combiner is not None and self.transer is not None:
            # TODO: 实现原始数据到 WOE 的转换
            raise NotImplementedError(
                "Combiner and Transer transformation not yet implemented. "
                "Please provide WOE-transformed data."
            )

        # 确保列顺序一致
        X = X[self.feature_names_]

        # 计算每个特征的分数
        sub_scores = self._woe_to_score(X)
        
        # 总分 = 截距分数 + 各特征分数之和
        intercept_score = self.offset - self.factor * self.intercept_
        total_score = intercept_score + sub_scores.sum(axis=1)

        return total_score

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率（使用底层 LR 模型）.

        :param X: 数据
        :return: 概率数组，shape (n_samples, 2)
        """
        check_is_fitted(self)
        return self.lr_model_.predict_proba(X)

    def scorecard_scale(self) -> pd.DataFrame:
        """输出评分卡基础配置.

        :return: 包含基础配置的 DataFrame
        """
        check_is_fitted(self)

        scale_df = pd.DataFrame([
            {
                "刻度项": "base_odds",
                "刻度值": self.base_odds,
                "备注": "根据业务经验设置的基础比率（违约概率/正常概率），估算方法：（1-样本坏客户占比）/坏客户占比"
            },
            {
                "刻度项": "base_score",
                "刻度值": self.base_score,
                "备注": "基础ODDS对应的分数"
            },
            {
                "刻度项": "rate",
                "刻度值": self.rate,
                "备注": "设置分数的倍率"
            },
            {
                "刻度项": "pdo",
                "刻度值": self.pdo,
                "备注": "表示分数增长PDO时，ODDS值增长到RATE倍"
            },
            {
                "刻度项": "B",
                "刻度值": self.factor,
                "备注": "补偿值，计算方式：pdo / ln(rate)"
            },
            {
                "刻度项": "A",
                "刻度值": self.offset,
                "备注": "刻度，计算方式：base_score - B * ln(base_odds)"
            },
        ])

        return scale_df

    def scorecard_points(
        self,
        feature_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """输出评分卡分箱信息及其对应的分数.

        :param feature_map: 特征描述字典，格式为 {特征名: 描述}
        :return: 评分卡 DataFrame
        """
        check_is_fitted(self)

        if feature_map is None:
            feature_map = {}

        rows = []
        for col in self.feature_names_:
            rule = self.rules_[col]
            
            # 格式化分箱标签
            bins = rule['bins']
            scores = rule['scores']
            
            if isinstance(bins[0], (list, np.ndarray)):
                # 类别特征
                for bin_vals, score in zip(bins, scores):
                    bin_label = ', '.join([str(v) for v in bin_vals])
                    rows.append({
                        '变量名称': col,
                        '变量含义': feature_map.get(col, ''),
                        '变量分箱': bin_label,
                        '对应分数': round(score, 2)
                    })
            else:
                # 数值特征
                for i, score in enumerate(scores):
                    if i == 0:
                        bin_label = f'[-inf, {bins[0]})'
                    elif i == len(scores) - 1:
                        if pd.isna(bins[-1]):
                            bin_label = '缺失值'
                        else:
                            bin_label = f'[{bins[-2]}, +inf)'
                    else:
                        bin_label = f'[{bins[i-1]}, {bins[i]})'
                    
                    rows.append({
                        '变量名称': col,
                        '变量含义': feature_map.get(col, ''),
                        '变量分箱': bin_label,
                        '对应分数': round(score, 2)
                    })

        return pd.DataFrame(rows)

    def score_to_bad_rate_table(
        self,
        scores: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
        method: str = 'quantile'
    ) -> pd.DataFrame:
        """输出评分区间对应坏率和odds的对照表.

        :param scores: 评分数组
        :param y: 真实标签
        :param n_bins: 分箱数量，默认 10
        :param method: 分箱方法，'quantile' 或 'uniform'
        :return: 对照表 DataFrame
        """
        # 创建 DataFrame
        df = pd.DataFrame({'score': scores, 'y': y})
        
        # 分箱
        if method == 'quantile':
            df['score_bin'] = pd.qcut(df['score'], q=n_bins, duplicates='drop')
        else:
            df['score_bin'] = pd.cut(df['score'], bins=n_bins)
        
        # 统计
        stats = df.groupby('score_bin').agg({
            'y': ['count', 'sum', 'mean']
        }).reset_index()
        
        stats.columns = ['评分区间', '样本数', '坏样本数', '坏样本率']
        
        # 计算 odds
        stats['好样本数'] = stats['样本数'] - stats['坏样本数']
        stats['Odds'] = stats['好样本数'] / stats['坏样本数'].replace(0, np.nan)
        stats['Odds'] = stats['Odds'].replace(np.nan, np.inf)
        
        # 计算 KS
        stats['累计好样本占比'] = stats['好样本数'].cumsum() / stats['好样本数'].sum()
        stats['累计坏样本占比'] = stats['坏样本数'].cumsum() / stats['坏样本数'].sum()
        stats['KS'] = abs(stats['累计坏样本占比'] - stats['累计好样本占比'])
        
        # 格式化
        stats['坏样本率'] = stats['坏样本率'].apply(lambda x: f'{x:.2%}')
        stats['Odds'] = stats['Odds'].apply(lambda x: f'{x:.2f}' if x != np.inf else 'inf')
        stats['KS'] = stats['KS'].apply(lambda x: f'{x:.4f}')
        
        return stats

    def save_pickle(
        self,
        file: str,
        engine: str = 'joblib'
    ) -> None:
        """保存模型为 pickle/joblib/dill 文件.

        :param file: 文件路径
        :param engine: 保存引擎，'pickle'/'joblib'/'dill'，默认 'joblib'
        """
        # 创建目录
        file_dir = os.path.dirname(file)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        if engine == 'pickle':
            import pickle
            with open(file, 'wb') as f:
                pickle.dump(self, f)
        elif engine == 'joblib':
            import joblib
            joblib.dump(self, file)
        elif engine == 'dill':
            try:
                import dill
                with open(file, 'wb') as f:
                    dill.dump(self, f)
            except ImportError:
                raise ImportError("使用 dill 需要安装: pip install dill")
        else:
            raise ValueError(f"engine 参数必须是 'pickle'/'joblib'/'dill' 之一，当前为: {engine}")

        print(f"模型已保存至: {file}")

    @classmethod
    def load_pickle(cls, file: str, engine: str = 'joblib') -> 'ScoreCard':
        """从文件加载模型.

        :param file: 文件路径
        :param engine: 加载引擎，'pickle'/'joblib'/'dill'
        :return: ScoreCard 实例
        """
        if engine == 'pickle':
            import pickle
            with open(file, 'rb') as f:
                return pickle.load(f)
        elif engine == 'joblib':
            import joblib
            return joblib.load(file)
        elif engine == 'dill':
            try:
                import dill
                with open(file, 'rb') as f:
                    return dill.load(f)
            except ImportError:
                raise ImportError("使用 dill 需要安装: pip install dill")
        else:
            raise ValueError(f"engine 参数必须是 'pickle'/'joblib'/'dill' 之一，当前为: {engine}")

    def export_pmml(
        self,
        pmml_file: str = 'scorecard.pmml',
        debug: bool = False
    ) -> Optional[Any]:
        """导出评分卡模型为 PMML 文件.

        需要 JDK 1.8+ 和 sklearn2pmml 库。

        :param pmml_file: PMML 文件路径
        :param debug: 是否开启调试模式
        :return: 如果 debug=True，返回 pipeline 对象
        """
        try:
            from sklearn_pandas import DataFrameMapper
            from sklearn.linear_model import LinearRegression
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
            from sklearn2pmml.preprocessing import LookupTransformer, ExpressionTransformer
        except ImportError as e:
            raise ImportError(
                "导出 PMML 需要安装以下依赖：\n"
                "  pip install sklearn-pandas sklearn2pmml\n"
                "同时需要 JDK 1.8+ 环境"
            ) from e

        check_is_fitted(self)

        # 构建 DataFrameMapper
        mapper = []
        samples = {}

        for var, rule in self.rules_.items():
            bins = rule['bins']
            scores = rule['scores']

            if isinstance(bins[0], (np.ndarray, list)):
                # 类别特征：使用 LookupTransformer
                mapping = {}
                default_value = 0.0
                
                for bin_vals, score in zip(bins, scores):
                    for bin_val in bin_vals:
                        if pd.isna(bin_val) or bin_val == 'nan':
                            default_value = float(score)
                        else:
                            mapping[str(bin_val)] = float(score)

                mapper.append((
                    [var],
                    LookupTransformer(mapping=mapping, default_value=default_value)
                ))
                samples[var] = [list(mapping.keys())[i] for i in np.random.randint(0, len(mapping), 20)]
            else:
                # 数值特征：使用 ExpressionTransformer
                expression_string = self._build_expression(bins, scores)
                mapper.append(([var], ExpressionTransformer(expression_string)))
                samples[var] = np.random.random(20) * 100

        scorecard_mapper = DataFrameMapper(mapper, df_out=True)

        # 构建 PMML Pipeline
        pipeline = PMMLPipeline([
            ('preprocessing', scorecard_mapper),
            ('scorecard', LinearRegression(fit_intercept=False)),
        ])

        # 拟合虚拟数据
        sample_df = pd.DataFrame(samples)
        sample_y = pd.Series(np.random.randint(0, 2, 20), name='score')
        pipeline.fit(sample_df, sample_y)
        pipeline.named_steps['scorecard'].coef_ = np.ones(len(mapper))

        # 导出 PMML
        try:
            sklearn2pmml(pipeline, pmml_file, with_repr=True, debug=debug)
            print(f"PMML 文件已导出至: {pmml_file}")
        except Exception as e:
            import traceback
            print(f"导出 PMML 失败: {e}")
            traceback.print_exc()
            if debug:
                return pipeline
            raise

        if debug:
            return pipeline

    def _build_expression(self, bins: np.ndarray, scores: np.ndarray) -> str:
        """构建数值特征的表达式字符串.

        :param bins: 分箱边界
        :param scores: 对应分数
        :return: 表达式字符串
        """
        expression = ""
        has_empty = len(bins) > 0 and pd.isna(bins[-1])

        if has_empty:
            score_empty = scores[-1]
            bin_scores = scores[:-1]
            bin_vars = bins[:-1]
            expression = f"{score_empty} if pandas.isnull(X[0]) "
        else:
            bin_scores = scores
            bin_vars = bins

        total_bins = len(bin_scores)
        end_string = ""

        for i in range(total_bins):
            if i == 0:
                _expression = f"{bin_scores[i]}"
            elif i == total_bins - 1:
                _expression += f" if X[0] < {bin_vars[i-1]} else {bin_scores[i]}"
            else:
                _expression += f" if X[0] < {bin_vars[i-1]} else ({bin_scores[i]} "
                end_string += ")"

        _expression += end_string

        if has_empty:
            expression += f"else ({_expression})" if _expression.count('else') > 0 else _expression
        else:
            expression = _expression

        return expression

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性（基于系数绝对值）.

        :return: 特征重要性 DataFrame
        """
        check_is_fitted(self)

        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'coef': self.coef_,
            'importance': np.abs(self.coef_)
        }).sort_values('importance', ascending=False)

        return importance_df

    def get_reason(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        keep: int = 3
    ) -> pd.DataFrame:
        """获取评分的主要原因（Top K 影响特征）.

        :param X: 数据
        :param keep: 保留前 K 个原因
        :return: 原因 DataFrame
        """
        check_is_fitted(self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 计算各特征分数
        sub_scores = self._woe_to_score(X[self.feature_names_])
        
        # 计算与基础效应的差异
        effect_diff = sub_scores - self.base_effect_.values
        
        # 找出 Top K 原因
        reasons_list = []
        for i in range(len(X)):
            row_diff = effect_diff[i]
            # 按绝对值排序
            top_indices = np.argsort(np.abs(row_diff))[::-1][:keep]
            
            reasons = []
            for idx in top_indices:
                feature = self.feature_names_[idx]
                diff = row_diff[idx]
                direction = "降低" if diff < 0 else "提升"
                reasons.append(f"{feature}({direction}{abs(diff):.1f}分)")
            
            reasons_list.append('; '.join(reasons))

        return pd.DataFrame({'reason': reasons_list})
