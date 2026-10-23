"""类型筛选器.

按数据类型筛选特征。

**参考样例**

>>> from hscredit.core.selectors import TypeSelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'a': [1, 2, 3],
...     'b': ['x', 'y', 'z'],
...     'c': [1.0, 2.0, 3.0]
... })
>>> # 仅保留数值类型
>>> selector = TypeSelector(dtype_include='number')
>>> selector.fit(X)
"""

from typing import Union, List, Optional, Type, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions


def _matches_dtype_feature(task):
    """判断单列是否满足类型包含/排除条件。"""
    feature, series, dtype_include, dtype_exclude = task
    if dtype_include is None and dtype_exclude is None:
        return feature, True
    selected = series.to_frame().select_dtypes(
        include=dtype_include,
        exclude=dtype_exclude,
    )
    return feature, feature in selected.columns


class TypeSelector(BaseFeatureSelector):
    """类型筛选器.

    按数据类型筛选特征。
    可以按包含类型或排除类型进行筛选。

    **参数**

    :param target: 目标变量列名，默认为 'target'。在 scorecardpipeline 风格中用于
        从 DataFrame 中提取目标列；在 sklearn 风格中若 fit 传入了 y 参数则优先使用 y

    :type target: str
    :param include: 默认值为 ``None`` 。强制保留的特征列表，这些特征无论如何都会被保留
    :type include: Optional[List[str]]
    :param exclude: 默认值为 ``None`` 。强制剔除的特征列表，这些特征无论如何都会被剔除
    :type exclude: Optional[List[str]]
    :param force_drop: 默认值为 ``None`` 。强制剔除的特征列表，效果与 exclude 合并
    :type force_drop: Optional[List[str]]
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Optional[Union[int, float]]
    :param binner: 默认值为 ``None`` 。可选的已配置分箱器实例，支持已训练或待训练状态，不接受分箱器类
    :type binner: Optional[Any]
    :param bin_params: 默认值为 ``None`` 。可选的 ``OptimalBinning`` 构造参数字典。未传入 ``binner`` 时
        基类使用该字典创建分箱器；同时传入时 ``binner`` 优先

    :type bin_params: Optional[Dict[str, Any]]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]
    :param target_rm: 默认False保留输入中的目标列；仅显式True时移除，不影响拟合时目标隔离
    :type target_rm: bool

    :param dtype_include: 包含的数据类型，默认为None

        - numpy.number: 所有数值类型

        - 'object': 所有对象类型

        - 'category': 类别类型

    :type dtype_include: Optional[Union[str, Type, List[str]]]
    :param dtype_exclude: 排除的数据类型，默认为None
    :type dtype_exclude: Optional[Union[str, Type, List[str]]]

    **参考样例**

    ::

        >>> from hscredit.core.selectors import TypeSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': ['x', 'y', 'z'],
        ...     'c': [1.0, 2.0, 3.0]
        ... })
        >>> # 仅保留数值类型
        >>> selector = TypeSelector(dtype_include='number')
        >>> selector.fit(X)
    """

    method_name = "类型筛选"

    def __init__(
        self,
        dtype_include: Optional[Union[str, Type, List[str]]] = None,
        dtype_exclude: Optional[Union[str, Type, List[str]]] = None,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        bin_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        target_rm: bool = False,
    ):
        """初始化筛选器；默认透传已有目标列，仅target_rm=True移除。"""
        super().__init__(
            target=target,
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            bin_params=bin_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            target_rm=target_rm,
        )
        self.dtype_include = dtype_include
        self.dtype_exclude = dtype_exclude

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合类型筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        # 类型判断只读模式元数据，不为报告额外扫描整个数据集。
        self.total_counts_ = pd.Series(len(X), index=X.columns, dtype=np.int64)
        self.threshold_ = None
        self.score_name_, self.score_direction_ = "类型匹配", "满足条件"
        if self.dtype_include is None and self.dtype_exclude is None:
            selected_cols = list(X.columns)
        else:
            selected_cols = X.select_dtypes(
                include=self.dtype_include,
                exclude=self.dtype_exclude,
            ).columns.tolist()

        self.dtypes_ = X.dtypes
        self.scores_ = pd.Series(
            [1 if c in selected_cols else 0 for c in X.columns],
            index=X.columns,
        )
        self.selected_features_ = selected_cols
        self.matches_ = self.scores_.astype(bool)
        record_conditions(self, X.columns, 类型匹配=self.matches_)
        self._drop_reason = "数据类型不匹配"
