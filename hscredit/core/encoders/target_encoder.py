"""Target Encoder (目标编码器).

基于目标变量均值对类别特征进行编码。
"""

from typing import Optional, List, Dict, Union, Any
import numpy as np
import pandas as pd

from .base import BaseEncoder


class TargetEncoder(BaseEncoder):
    """目标编码器.

    用目标变量的均值对每个类别进行编码：
    - 对于分类任务：该类别的正样本比例
    - 对于回归任务：该类别的目标变量均值

    使用平滑技术防止过拟合：
    encoded = (count * mean + smoothing * global_mean) / (count + smoothing)

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则自动识别所有列（支持类别型和数值型）
    :param smoothing: 平滑参数，值越大收缩到全局均值的程度越大，默认为1.0
    :param min_samples_leaf: 每个类别的最小样本数，少于该值则使用全局均值，默认为1
    :param noise: 添加的高斯噪声标准差，用于防止过拟合，默认为None
    :param handle_unknown: 处理未知类别的方式，默认为'value'
    :param handle_missing: 处理缺失值的方式，默认为'value'
    :param drop_invariant: 是否删除方差为0的列，默认为False
    :param return_df: 是否返回DataFrame，默认为True

    **属性**

    - mapping_: 目标编码映射字典，格式为 {col: {category: encoded_value}}
    - global_mean_: 全局目标均值

    **参考样例**

    >>> from hscredit.core.encoders import TargetEncoder
    >>> encoder = TargetEncoder(cols=['category'])
    >>> X_encoded = encoder.fit_transform(X, y)
    >>>
    >>> # 添加噪声防止过拟合
    >>> encoder = TargetEncoder(cols=['category'], noise=0.05)
    >>> X_encoded = encoder.fit_transform(X, y)

    **注意**

    目标编码直接使用了标签信息，存在目标泄漏/过拟合风险，应配合 ``smoothing``、
    ``min_samples_leaf``、``noise`` 等正则手段，并务必在训练集 fit、在验证/测试集
    transform；若需更强的防泄漏，改用 :class:`CatBoostEncoder`（有序目标统计）。

    **引用**

    Micci-Barreca, D. (2001). *A preprocessing scheme for high-cardinality
    categorical attributes in classification and prediction problems.*
    ACM SIGKDD Explorations, 3(1). https://doi.org/10.1145/507533.507538
    """

    # global_mean_ 是 transform 时未知/缺失类别的填充值，须随映射一并序列化
    _EXTRA_STATE_ATTRS = ["global_mean_"]

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """自动识别需要编码的列。

        TargetEncoder支持数值型和类别型列，因此返回所有列。

        :param X: 输入数据
        :return: 列名列表
        """
        return X.columns.tolist()

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        noise: Optional[float] = None,
        handle_unknown: str = "value",
        handle_missing: str = "value",
        drop_invariant: bool = False,
        return_df: bool = True,
        random_state: Optional[int] = None,
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
    ):
        """初始化目标编码器。

        :param cols: 需要编码的列名列表
        :param smoothing: 平滑参数，值越大收缩到全局均值的程度越大，默认为1.0
        :param min_samples_leaf: 每个类别的最小样本数，少于该值则使用全局均值，默认为1
        :param noise: 添加的高斯噪声标准差，用于防止过拟合，默认为None
        :param handle_unknown: 处理未知类别的方式，默认为'value'
        :param handle_missing: 处理缺失值的方式，默认为'value'
        :param drop_invariant: 是否删除方差为0的列，默认为False
        :param return_df: 是否返回DataFrame，默认为True
        :param random_state: 随机种子，用于噪声生成，默认为None
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
        )
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise = noise
        self.random_state = random_state

        self.global_mean_: float = 0.0

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合目标编码器。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量
        :raises ValueError: 当y为空时抛出
        """
        if y is None:
            raise ValueError("TargetEncoder是有监督编码器，必须提供目标变量y")

        y = pd.Series(y, name="target")
        global_mean = y.mean()
        self._fit_columns(X, y, shared_state={"global_mean_": global_mean})
        self.global_mean_ = global_mean

    def _fit_column(self, column, values, y=None):
        df_temp = pd.DataFrame({"feature": values, "target": y.values})
        stats = df_temp.groupby("feature")["target"].agg(["mean", "count"])

        smoothed_means = (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_) / (
            stats["count"] + self.smoothing
        )

        small_sample_mask = stats["count"] < self.min_samples_leaf
        smoothed_means[small_sample_mask] = self.global_mean_

        mapping = smoothed_means.to_dict()

        if self.handle_missing == "value":
            mapping[np.nan] = self.global_mean_
        elif self.handle_missing == "return_nan":
            mapping[np.nan] = np.nan

        if self.handle_unknown == "value":
            mapping["__UNKNOWN__"] = self.global_mean_
        elif self.handle_unknown == "return_nan":
            mapping["__UNKNOWN__"] = np.nan

        return {"mapping_": mapping}

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据，shape (n_samples, n_features)
        :param y: 目标变量（可选），如果提供则添加噪声
        :return: 编码后的数据
        """
        contexts = {}
        if self.noise is not None and y is not None:
            rng = np.random.RandomState(self.random_state)
            contexts = {column: rng.normal(0, self.noise, len(X)) for column in self.cols_}
        return self._transform_columns(X, y, contexts=contexts)

    def _transform_column(self, column, values, y=None, context=None):
        mapping = self.mapping_[column]
        result = values.map(mapping)

        if self.handle_unknown == "value":
            result = result.fillna(self.global_mean_)
        elif self.handle_unknown == "error" and result.isna().any():
            raise ValueError(f"列'{column}'包含未知类别")

        if context is not None:
            result = result * (1 + context)

        return result
