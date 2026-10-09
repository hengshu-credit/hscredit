"""CatBoost Encoder.

基于CatBoost算法的有序目标编码器，使用排序来防止目标泄漏。
"""

from typing import Optional, List, Dict, Union, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder
from ._category_protocol import MISSING, UNKNOWN


class CatBoostEncoder(BaseEncoder):
    """CatBoost编码器.

    使用有序目标统计（Ordered Target Statistics）方法，
    通过随机排序和累积统计来防止过拟合和目标泄漏。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有列（支持类别型和数值型）
    :param sigma: 添加的高斯噪声标准差，默认为None
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True
    :param random_state: 随机种子，用于可复现性，默认为None

    **属性**

    - mapping_: 目标编码映射字典，格式为 {col: {category: encoded_value}}
    - global_mean_: 全局目标均值

    **参考样例**

    >>> from hscredit.core.encoders import CatBoostEncoder
    >>> encoder = CatBoostEncoder(cols=['category'])
    >>> X_encoded = encoder.fit_transform(X, y)
    >>>
    >>> # 添加噪声
    >>> encoder = CatBoostEncoder(cols=['category'], sigma=0.05, random_state=42)
    >>> X_encoded = encoder.fit_transform(X, y)

    **注意**

    与普通目标编码相比，本编码器对样本随机排序后只用"当前样本之前"的目标累积统计来编码
    （ordered target statistics），从而显著降低目标泄漏；``random_state`` 决定排序，
    影响结果可复现性。

    **引用**

    Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical
    features.* NeurIPS 2018. https://arxiv.org/abs/1706.09516
    """

    # global_mean_ 是 transform 时未知/缺失类别的填充值，须随映射一并序列化
    _EXTRA_STATE_ATTRS = ["global_mean_"]
    _TARGET_TYPE = "continuous"

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        CatBoostEncoder支持数值型和类别型列，因此返回所有列。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        sigma: Optional[float] = None,
        handle_unknown: str = "value",
        handle_missing: str = "value",
        drop_invariant: bool = False,
        return_df: bool = True,
        random_state: Optional[int] = None,
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        passthrough_target: bool = False,
    ):
        """初始化CatBoost编码器。

        :param cols: 需要编码的列名列表
        :param sigma: 添加的高斯噪声标准差，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param random_state: 随机种子，默认为None
        :param target: scorecardpipeline风格的目标列名。如果提供，fit时从X中提取该列作为y
        """
        super().__init__(
            cols=cols,
            drop_invariant=drop_invariant,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            passthrough_target=passthrough_target,
        )
        self.sigma = sigma
        self.random_state = random_state

        self.global_mean_: float = 0.0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合CatBoost编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量
        :raises ValueError: 当y为空时抛出
        """
        if y is None:
            raise ValueError("CatBoostEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y, name="target")
        global_mean = y.mean()
        self._fit_columns(X, y, shared_state={"global_mean_": global_mean})
        self.global_mean_ = global_mean

    def _fit_column(self, column, values, y=None):
        df_temp = pd.DataFrame({"feature": values, "target": y.values})
        category_stats = df_temp.groupby("feature")["target"].agg(["mean", "count"])

        mapping = category_stats["mean"].to_dict()

        if self.handle_missing == "value":
            mapping[MISSING] = self.global_mean_
        elif self.handle_missing == "return_nan":
            mapping[MISSING] = np.nan

        if self.handle_unknown == "value":
            mapping[UNKNOWN] = self.global_mean_
        elif self.handle_unknown == "return_nan":
            mapping[UNKNOWN] = np.nan

        return {"mapping_": mapping}

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），如果提供则使用有序统计
        :return: 编码后的数据
        """
        contexts = {}
        if y is not None:
            rng = np.random.RandomState(self.random_state)
            for column in self.cols_:
                order = rng.permutation(len(X))
                noise = rng.normal(0, self.sigma, len(X)) if self.sigma is not None else None
                contexts[column] = (order, noise)
        return self._transform_columns(X, y, contexts=contexts)

    def _transform_column(self, column, values, y=None, context=None):
        mapping = self.mapping_[column]

        if y is not None:
            order, noise = context
            result = self._transform_ordered(values, y, mapping, random_order=order)
        else:
            noise = None
            result = self._map_values(values, mapping)

        result = self._apply_missing_unknown(column, values, result, self.global_mean_)

        if noise is not None:
            result = result * (1 + noise)

        return result

    def _transform_ordered(
        self,
        x: pd.Series,
        y: pd.Series,
        mapping: Dict,
        rng: Optional[np.random.RandomState] = None,
        random_order: Optional[np.ndarray] = None,
    ) -> pd.Series:
        """使用有序统计进行转换（防止目标泄漏）。

        :param x: 特征列
        :param y: 目标变量
        :param mapping: 编码映射
        :param rng: 局部随机数发生器，None 时按 random_state 新建
        :return: 编码后的序列
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=x.index)

        n = len(x)
        if random_order is None:
            if rng is None:
                rng = np.random.RandomState(self.random_state)
            random_order = rng.permutation(n)

        ordered_x = x.iloc[random_order].reset_index(drop=True)
        ordered_y = pd.Series(np.asarray(y)[random_order])
        codes, _ = pd.factorize(ordered_x, sort=False)
        grouped = ordered_y.groupby(codes, sort=False)
        previous_sum = grouped.cumsum() - ordered_y
        previous_count = grouped.cumcount()
        encoded = (previous_sum + self.global_mean_) / (previous_count + 1.0)
        encoded.loc[codes < 0] = mapping.get(MISSING, mapping.get(np.nan, self.global_mean_))
        restored = np.empty(n, dtype=float)
        restored[random_order] = encoded.to_numpy()
        return pd.Series(restored, index=x.index)
