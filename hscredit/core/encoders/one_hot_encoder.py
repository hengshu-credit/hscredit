"""One-Hot Encoder (独热编码器).

将类别特征转换为独热编码形式，支持数值型和类别型数据。
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd

from .base import BaseEncoder
from ...exceptions import NotFittedError


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

    >>> from hscredit.core.encoders import OneHotEncoder
    >>> encoder = OneHotEncoder(cols=['color'])
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 删除第一列避免多重共线性
    >>> encoder = OneHotEncoder(cols=['color'], drop='first')
    >>> X_encoded = encoder.fit_transform(X)

    **注意**

    独热编码为无监督方法，列数随类别基数线性增长，仅适合低基数特征；用于线性/逻辑回归时
    建议 ``drop='first'`` 以消除虚拟变量陷阱（多重共线性），用于树模型可保留全部列。

    **引用**

    虚拟变量（dummy variables）/ one-hot 编码是统计建模标准做法，参见 sklearn
    ``OneHotEncoder``：
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """

    # categories_ 是 transform 生成独热列所必需的状态（_transform 依赖它而非 mapping_）；
    # feature_names_ / _other_cols_ 供 get_feature_names(_out) 使用，三者须一并序列化
    _EXTRA_STATE_ATTRS = ["categories_", "feature_names_", "_other_cols_"]

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop: Optional[str] = None,
        handle_unknown: str = "ignore",
        handle_missing: str = "value",
        use_cat_names: bool = True,
        return_df: bool = True,
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
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
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.drop = drop
        self.use_cat_names = use_cat_names

        self.categories_: Dict[str, List] = {}
        self.feature_names_: List[str] = []
        self._other_cols_: List[str] = []

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
        # 保留未编码列，与 _transform 输出顺序保持一致（未编码列在前）
        self._other_cols_ = [c for c in X.columns if c not in self.cols_]
        self._fit_columns(X, y, state_attrs=("mapping_", "categories_"))
        self.feature_names_ = [
            self.mapping_[column][category]
            for column in self.cols_
            for category in self.categories_[column]
        ]

    def _fit_column(self, column, values, y=None):
        # 获取唯一值（包括缺失值）
        categories = values.unique()

        # 分离缺失值和正常值
        has_missing = any(pd.isna(c) for c in categories)
        normal_categories = self._sort_categories([c for c in categories if not pd.isna(c)])

        # 处理drop参数
        if self.drop == "first" and len(normal_categories) > 0:
            categories_to_use = normal_categories[1:]
        elif self.drop == "if_binary" and len(normal_categories) == 2:
            categories_to_use = normal_categories[:1]
        else:
            categories_to_use = normal_categories[:]

        # 如果有缺失值且handle_missing='value'，添加missing
        if has_missing and self.handle_missing == "value":
            categories_to_use = categories_to_use + ["missing"]

        # 构建mapping_（与其他编码器保持一致）
        col_mapping = {}
        for cat in categories_to_use:
            if cat == "missing":
                col_name = f"{column}_missing"
            elif self.use_cat_names:
                safe_cat = str(cat).replace(" ", "_").replace("-", "_")
                col_name = f"{column}_{safe_cat}"
            else:
                col_name = f"{column}_{cat}"
            col_mapping[cat] = col_name
        return {"mapping_": col_mapping, "categories_": categories_to_use}

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据
        :param y: 目标变量（可选）
        :return: 编码后的数据
        """
        return self._transform_columns(X, y, passthrough=True)

    def _transform_column(self, column, values, y=None, context=None):
        categories = self.categories_[column]

        # 检查未知类别
        if self.handle_unknown == "error":
            unique_vals = set(values.dropna().unique())
            known_vals = set(categories) - {"missing"}
            unknown = unique_vals - known_vals
            if unknown:
                raise ValueError(f"列'{column}'包含未知类别: {unknown}")

        # 处理缺失值
        col_data = values.copy()
        if self.handle_missing == "value":
            col_data = col_data.fillna("missing")

        # 创建one-hot列
        data = {}
        for cat in categories:
            if cat == "missing":
                col_name = f"{column}_missing"
                data[col_name] = values.isna().astype(int)
            else:
                if self.use_cat_names:
                    safe_cat = str(cat).replace(" ", "_").replace("-", "_")
                    col_name = f"{column}_{safe_cat}"
                else:
                    col_name = f"{column}_{cat}"
                data[col_name] = (col_data == cat).astype(int)
        return pd.DataFrame(data, index=values.index)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆编码，将独热编码列还原为原始类别列。

        对每个原始列，取值为 1 的独热列对应类别即为原始类别；
        若所有独热列均为 0（如 drop 删除的参考类别或未知类别），则还原为 NaN。
        缺失列（``{col}_missing``）激活时还原为 NaN。

        :param X: 编码后的数据
        :return: 逆编码后的数据
        :raises NotFittedError: 当编码器尚未拟合时抛出
        """
        if not hasattr(self, "mapping_") or self.mapping_ is None or len(self.mapping_) == 0:
            raise NotFittedError("OneHotEncoder 尚未拟合，请先调用 fit 方法")

        X = self._check_input(X).copy()

        reconstructed = {}
        consumed = set()
        for col in self.cols_ or []:
            col_map = self.mapping_.get(col, {})  # {category: col_name}
            name_to_cat = {name: cat for cat, name in col_map.items()}
            present = [n for n in col_map.values() if n in X.columns]
            if not present:
                continue
            consumed.update(present)

            sub = X[present]

            def _pick(row):
                for name in present:
                    if row[name] == 1:
                        cat = name_to_cat[name]
                        return np.nan if cat == "missing" else cat
                return np.nan

            reconstructed[col] = sub.apply(_pick, axis=1)

        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            if c not in consumed:
                out[c] = X[c]
        for col, series in reconstructed.items():
            out[col] = series

        return out

    def get_feature_names(self) -> List[str]:
        """获取独热编码生成的特征名（不含未编码的透传列）。

        :return: 独热编码后的特征名列表
        """
        return self.feature_names_

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """获取转换后的全部输出列名（sklearn 兼容接口）。

        输出顺序与 transform 一致：未编码透传列在前，独热编码列在后。

        :param input_features: 兼容 sklearn 接口的占位参数，未使用
        :return: 输出列名数组
        """
        other_cols = getattr(self, "_other_cols_", [])
        return np.asarray(list(other_cols) + list(self.feature_names_), dtype=object)
