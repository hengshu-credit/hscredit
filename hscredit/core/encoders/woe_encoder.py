"""WOE (Weight of Evidence) 编码器.

提供直接计算WOE的编码功能，不依赖分箱模块。
"""

from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd

from .base import BaseEncoder


class WOEEncoder(BaseEncoder):
    """WOE (证据权重) 编码器.

    直接对类别特征计算WOE值，不依赖分箱功能。

    WOE计算公式：
    WOE = ln(P(好样本|类别) / P(坏样本|类别)) = ln(好样本占比/坏样本占比)

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有类别型列
    :param regularization: 正则化参数，防止除零，默认为1.0
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - mapping_: WOE编码映射字典，格式为 {col: {category: woe_value}}
    - iv_: 各特征的IV值，格式为 {col: iv_value}

    **参考样例**

    基本使用::

        >>> from hscredit.core.encoders import WOEEncoder
        >>> encoder = WOEEncoder(cols=['category', 'score'])
        >>> X_encoded = encoder.fit_transform(X, y)
        >>> print(encoder.iv_)

    获取IV摘要::

        >>> summary = encoder.summary()
        >>> print(summary)

    参考:
        https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        regularization: float = 1.0,
        handle_unknown: str = 'value',
        handle_missing: str = 'value',
        drop_invariant: bool = False,
        return_df: bool = True,
        target: Optional[str] = None,
    ):
        """初始化WOE编码器。

        :param cols: 需要编码的列名列表
        :param regularization: 正则化参数，防止除零，默认为1.0
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
        )
        self.regularization = regularization

        self.iv_: Dict[str, float] = {}

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        WOEEncoder支持数值型和类别型列，因此返回所有列（除了目标列）。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合WOE编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量，二分类 (0/1)
        :raises ValueError: 当y为空或目标变量不是二元时抛出
        """
        if y is None:
            raise ValueError("WOEEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y).astype(int)

        unique = y.unique()
        if len(unique) != 2:
            raise ValueError(f"目标变量必须是二元的，当前有{len(unique)}个唯一值")
        if not set(unique).issubset({0, 1}):
            raise ValueError("目标变量必须是0和1")

        total_good = (y == 0).sum()
        total_bad = (y == 1).sum()

        for col in self.cols_:
            woe_map, iv = self._fit_categorical(X[col], y, total_good, total_bad)
            self.mapping_[col] = woe_map
            self.iv_[col] = iv

    def _fit_categorical(
        self, x: pd.Series, y: pd.Series, total_good: int, total_bad: int
    ) -> tuple:
        """拟合类别特征的WOE。

        :param x: 特征列
        :param y: 目标变量
        :param total_good: 好样本总数
        :param total_bad: 坏样本总数
        :return: WOE映射和IV值的元组 (woe_map, iv)
        """
        woe_map = {}

        for category in x.unique():
            if pd.isna(category):
                continue

            mask = x == category
            good_count = (y[mask] == 0).sum()
            bad_count = (y[mask] == 1).sum()

            woe = self._compute_woe(good_count, bad_count, total_good, total_bad)
            woe_map[category] = woe

        if self.handle_missing == 'value':
            woe_map[np.nan] = 0.0
        elif self.handle_missing == 'return_nan':
            woe_map[np.nan] = np.nan

        if self.handle_unknown == 'value':
            woe_map['__UNKNOWN__'] = 0.0
        elif self.handle_unknown == 'return_nan':
            woe_map['__UNKNOWN__'] = np.nan

        iv = self._compute_iv_categorical(x, y, total_good, total_bad)

        return woe_map, iv

    def _compute_woe(
        self, good_count: int, bad_count: int, total_good: int, total_bad: int
    ) -> float:
        """计算WOE值（带正则化）。

        WOE = ln(坏样本占比 / 好样本占比)
        与 toad、scorecardpipeline 及 hscredit 分箱模块保持一致。
        坏样本集中的箱 WOE > 0，好样本集中的箱 WOE < 0，
        LR 系数为正，便于理解。

        :param good_count: 好样本数量
        :param bad_count: 坏样本数量
        :param total_good: 好样本总数
        :param total_bad: 坏样本总数
        :return: WOE值
        """
        good_rate = (good_count + self.regularization) / (total_good + 2 * self.regularization)
        bad_rate = (bad_count + self.regularization) / (total_bad + 2 * self.regularization)

        woe = np.log(bad_rate / good_rate)
        return woe

    def _compute_iv_categorical(
        self, x: pd.Series, y: pd.Series, total_good: int, total_bad: int
    ) -> float:
        """计算类别特征的IV。

        :param x: 特征列
        :param y: 目标变量
        :param total_good: 好样本总数
        :param total_bad: 坏样本总数
        :return: IV值
        """
        iv = 0.0
        for category in x.dropna().unique():
            mask = x == category
            good_count = (y[mask] == 0).sum()
            bad_count = (y[mask] == 1).sum()

            good_dist = (good_count + self.regularization) / (total_good + 2 * self.regularization)
            bad_dist = (bad_count + self.regularization) / (total_bad + 2 * self.regularization)

            iv += (bad_dist - good_dist) * np.log(bad_dist / good_dist)

        return iv

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据为WOE编码。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选）
        :return: 编码后的数据
        """
        for col in self.cols_:
            if col not in self.mapping_:
                continue

            woe_map = self.mapping_[col]
            X[col] = X[col].map(woe_map)

            if self.handle_unknown == 'value':
                X[col] = X[col].fillna(0.0)
            elif self.handle_unknown == 'error' and X[col].isna().any():
                raise ValueError(f"列'{col}'包含未知类别")

        return X

    def get_iv(self) -> Dict[str, float]:
        """获取各特征的IV值。

        :return: 特征名到IV值的映射字典
        """
        return self.iv_

    def summary(self) -> pd.DataFrame:
        """获取WOE编码摘要。

        :return: 包含各特征IV值和预测能力的摘要表
        """
        if not self.iv_:
            return pd.DataFrame()

        summary = []
        for col, iv in self.iv_.items():
            if iv < 0.02:
                power = '无预测力'
            elif iv < 0.1:
                power = '弱预测力'
            elif iv < 0.3:
                power = '中等预测力'
            elif iv < 0.5:
                power = '强预测力'
            else:
                power = '超强预测力(需检查)'

            summary.append({
                '特征': col,
                'IV值': round(iv, 4),
                '预测能力': power,
            })

        return pd.DataFrame(summary).sort_values('IV值', ascending=False)

    def export(self, to_json: Optional[str] = None) -> Dict[str, Dict]:
        """导出WOE编码规则，兼容 toad/scorecardpipeline 格式.

        导出格式与 toad.WOETransformer.export() 和 scorecardpipeline.WOETransformer.export() 保持一致。

        :param to_json: 可选，JSON 文件保存路径。如果提供，将规则保存到该文件
        :return: WOE编码规则字典，格式为 {feature: {value: woe_value, ...}, ...}

        **示例**

        >>> encoder = WOEEncoder(cols=['category', 'city'])
        >>> encoder.fit(X, y)
        >>> 
        >>> # 导出为字典
        >>> rules = encoder.export()
        >>> # 返回格式: {'category': {'A': 0.5, 'B': -0.3, ...}, 'city': {...}}
        >>> 
        >>> # 导出并保存到 JSON 文件
        >>> rules = encoder.export(to_json='woe_rules.json')
        
        **与 toad/scorecardpipeline 的兼容性**

        导出的规则可以直接被 toad 和 scorecardpipeline 加载:
        
        >>> # toad 加载
        >>> import toad
        >>> transformer = toad.transform.WOETransformer()
        >>> transformer.load(rules)
        >>> 
        >>> # scorecardpipeline 加载
        >>> from scorecardpipeline import WOETransformer
        >>> transformer = WOETransformer()
        >>> transformer.load(rules)
        """
        import json
        
        if not hasattr(self, 'mapping_') or not self.mapping_:
            raise ValueError("WOEEncoder 尚未拟合，请先调用 fit 方法")
        
        # 构建与 toad 兼容的格式: {feature: {value: woe_value}}
        rules = {}
        for col, woe_map in self.mapping_.items():
            # 将 WOE 映射转换为可 JSON 序列化的格式
            col_rules = {}
            for value, woe in woe_map.items():
                # 处理特殊值
                if value == '__UNKNOWN__':
                    continue  # toad 不保存 __UNKNOWN__
                if pd.isna(value):
                    col_rules['nan'] = woe  # toad 使用字符串 'nan'
                else:
                    col_rules[str(value)] = float(woe)
            rules[col] = col_rules
        
        if to_json is not None:
            # 确保目录存在
            import os
            dir_path = os.path.dirname(to_json)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(to_json, 'w', encoding='utf-8') as f:
                json.dump(rules, f, ensure_ascii=False, indent=2)
        
        return rules

    def load(self, from_json: Union[str, Dict], update: bool = False) -> 'WOEEncoder':
        """加载WOE编码规则，兼容 toad/scorecardpipeline 格式.

        从字典或 JSON 文件加载WOE编码规则，支持 toad 和 scorecardpipeline 导出的格式。

        :param from_json: WOE规则字典或 JSON 文件路径
            - 字典: {'category': {'A': 0.5, 'B': -0.3}}
            - 文件路径: 'woe_rules.json'
        :param update: 是否更新现有规则（而非替换），默认为 False
        :return: self，支持链式调用

        **示例**

        >>> encoder = WOEEncoder()
        >>> 
        >>> # 从字典加载
        >>> rules = {'category': {'A': 0.5, 'B': -0.3}}
        >>> encoder.load(rules)
        >>> 
        >>> # 从 JSON 文件加载
        >>> encoder.load('woe_rules.json')
        >>> 
        >>> # 更新现有规则
        >>> encoder.load({'new_feature': {'X': 0.2}}, update=True)
        
        **与 toad/scorecardpipeline 的兼容性**

        可以直接加载 toad 和 scorecardpipeline 导出的规则:
        
        >>> # toad 导出
        >>> import toad
        >>> toad_transformer = toad.transform.WOETransformer()
        >>> toad_transformer.fit(df, y)
        >>> rules = toad_transformer.export()
        >>> 
        >>> # hscredit 加载
        >>> from hscredit.core.encoders import WOEEncoder
        >>> encoder = WOEEncoder()
        >>> encoder.load(rules)
        """
        import json
        
        if isinstance(from_json, str):
            # 从文件加载
            with open(from_json, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        else:
            # 直接使用字典
            rules = from_json
        
        if not update:
            self.mapping_ = {}
            self.cols_ = []
        
        # 加载规则
        for col, col_rules in rules.items():
            if col not in self.mapping_:
                self.mapping_[col] = {}
            if col not in self.cols_:
                self.cols_.append(col)
            
            for value, woe in col_rules.items():
                # 处理 toad 的特殊值
                if value == 'nan':
                    self.mapping_[col][np.nan] = woe
                else:
                    # 尝试转换为原始类型
                    try:
                        # 尝试作为数字解析
                        if '.' in str(value):
                            key = float(value)
                        else:
                            key = int(value)
                    except (ValueError, TypeError):
                        key = value
                    self.mapping_[col][key] = woe
            
            # 添加未知值处理
            if self.handle_unknown == 'value':
                self.mapping_[col]['__UNKNOWN__'] = 0.0
            elif self.handle_unknown == 'return_nan':
                self.mapping_[col]['__UNKNOWN__'] = np.nan
        
        self._fitted = True
        return self
