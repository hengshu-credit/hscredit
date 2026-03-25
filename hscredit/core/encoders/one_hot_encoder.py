"""One-Hot Encoder (独热编码器).

将类别特征转换为独热编码形式，支持数值型和类别型数据。
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .base import BaseEncoder


class OneHotEncoder(BaseEncoder):
    """独热编码器.

    将每个类别转换为一个二进制列，适用于类别数量不多的特征。
    支持数值型和类别型数据。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则编码所有列
    :param drop: 是否删除某一列以避免多重共线性，默认为None
        - None: 保留所有列
        - 'first': 删除第一列
        - 'if_binary': 二值特征时删除一列
    :param handle_unknown: 处理未知类别的方式，默认为'ignore'
        - 'error': 抛出错误
        - 'ignore': 忽略（所有编码列为0）
    :param handle_missing: 处理缺失值的方式，默认为'value'
        - 'value': 单独编码为'missing'列
        - 'error': 抛出错误
    :param use_cat_names: 是否使用类别值作为列名后缀，默认为True
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - categories_: 各列的类别列表，格式为 {col: [category1, category2, ...]}
    - feature_names_: 编码后的特征名列表

    **参考样例**

    基本使用::

        >>> encoder = OneHotEncoder(cols=['color'])
        >>> X_encoded = encoder.fit_transform(X)

    删除第一列避免多重共线性::

        >>> encoder = OneHotEncoder(cols=['color'], drop='first')
        >>> X_encoded = encoder.fit_transform(X)

    参考:
        https://en.wikipedia.org/wiki/One-hot
    """

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop: Optional[str] = None,
        handle_unknown: str = 'ignore',
        handle_missing: str = 'value',
        use_cat_names: bool = True,
        return_df: bool = True,
        target: Optional[str] = None,
    ):
        """初始化独热编码器。

        :param cols: 需要编码的列名列表
        :param drop: 是否删除某一列以避免多重共线性
        :param handle_unknown: 处理未知类别的方式
        :param handle_missing: 处理缺失值的方式
        :param use_cat_names: 是否使用类别值作为列名后缀
        :param return_df: 是否返回DataFrame
        :param target: scorecardpipeline风格的目标列名
        """
        super().__init__(
            cols=cols,
            drop_invariant=False,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
        )
        self.drop = drop
        self.use_cat_names = use_cat_names

        self.categories_: Dict[str, List] = {}
        self.feature_names_: List[str] = []

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """获取需要编码的列。

        OneHotEncoder支持数值型和类别型列。

        :param X: 输入数据
        :return: 列名列表
        """
        if self.cols is not None:
            return [c for c in self.cols if c in X.columns]
        return X.columns.tolist()

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合独热编码器。

        :param X: 输入数据
        :param y: 目标变量（可选）
        """
        for col in self.cols_:
            # 获取唯一值（包括缺失值）
            categories = X[col].unique()
            
            # 分离缺失值和正常值
            has_missing = any(pd.isna(c) for c in categories)
            normal_categories = sorted([c for c in categories if not pd.isna(c)])
            
            # 处理drop参数
            if self.drop == 'first' and len(normal_categories) > 0:
                categories_to_use = normal_categories[1:]
            elif self.drop == 'if_binary' and len(normal_categories) == 2:
                categories_to_use = normal_categories[:1]
            else:
                categories_to_use = normal_categories[:]
            
            # 如果有缺失值且handle_missing='value'，添加missing
            if has_missing and self.handle_missing == 'value':
                categories_to_use = categories_to_use + ['missing']
            
            self.categories_[col] = categories_to_use
            
            # 构建mapping_（与其他编码器保持一致）
            col_mapping = {}
            for cat in categories_to_use:
                if self.use_cat_names:
                    safe_cat = str(cat).replace(' ', '_').replace('-', '_')
                    col_name = f"{col}_{safe_cat}"
                else:
                    col_name = f"{col}_{cat}"
                col_mapping[cat] = col_name
            
            self.mapping_[col] = col_mapping

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据
        :param y: 目标变量（可选）
        :return: 编码后的数据
        """
        result_dfs = []
        
        # 保留未编码的列
        other_cols = [c for c in X.columns if c not in self.cols_]
        if other_cols:
            result_dfs.append(X[other_cols].copy())
        
        # 对每个需要编码的列进行one-hot编码
        for col in self.cols_:
            if col not in self.categories_:
                continue
            
            categories = self.categories_[col]
            
            # 检查未知类别
            if self.handle_unknown == 'error':
                unique_vals = set(X[col].dropna().unique())
                known_vals = set(categories) - {'missing'}
                unknown = unique_vals - known_vals
                if unknown:
                    raise ValueError(f"列'{col}'包含未知类别: {unknown}")
            
            # 处理缺失值
            col_data = X[col].copy()
            if self.handle_missing == 'value':
                col_data = col_data.fillna('missing')
            
            # 创建one-hot列
            for cat in categories:
                if cat == 'missing':
                    col_name = f"{col}_missing"
                    result_dfs.append(pd.DataFrame({col_name: (X[col].isna()).astype(int)}))
                else:
                    if self.use_cat_names:
                        safe_cat = str(cat).replace(' ', '_').replace('-', '_')
                        col_name = f"{col}_{safe_cat}"
                    else:
                        col_name = f"{col}_{cat}"
                    result_dfs.append(pd.DataFrame({col_name: (col_data == cat).astype(int)}))
        
        # 合并所有列
        if result_dfs:
            return pd.concat(result_dfs, axis=1)
        else:
            return X.copy()

    def get_feature_names(self) -> List[str]:
        """获取编码后的特征名。

        :return: 编码后的特征名列表
        """
        return self.feature_names_
