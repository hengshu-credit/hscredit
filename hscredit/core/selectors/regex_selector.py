"""正则表达式筛选器.

按特征名称正则表达式筛选特征。

**参考样例**

>>> from hscredit.core.selectors import RegexSelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'income_1': [1, 2, 3],
...     'income_2': [4, 5, 6],
...     'age': [1, 2, 3]
... })
>>> # 选择以income开头的特征
>>> selector = RegexSelector(pattern='^income')
>>> selector.fit(X)
"""

import re
from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions


def _matches_regex_feature(task):
    """判断单个特征名是否匹配正则表达式。"""
    feature, pattern, flags = task
    return feature, re.search(pattern, str(feature), flags=flags) is not None


class RegexSelector(BaseFeatureSelector):
    """正则表达式筛选器.

    按特征名称的正则表达式匹配筛选特征。

    **参数**

    :param pattern: 正则表达式模式
    :param invert: 是否反转匹配（True 表示排除匹配的特征，保留不匹配的），默认为 False
    :param flags: 正则表达式标志，默认为 0

    **参考样例**

    ::

        >>> from hscredit.core.selectors import RegexSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'income_1': [1, 2, 3],
        ...     'income_2': [4, 5, 6],
        ...     'age': [1, 2, 3]
        ... })
        >>> # 选择以income开头的特征
        >>> selector = RegexSelector(pattern='^income')
        >>> selector.fit(X)
    """

    method_name = "正则筛选"

    def __init__(
        self,
        pattern: str,
        invert: bool = False,
        flags: int = 0,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        binning_params: Optional[Dict[str, Any]] = None,
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
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            target_rm=target_rm,
        )
        self.pattern = pattern
        self.invert = invert
        self.flags = flags

    def _check_input(self, X, y=None):
        if not isinstance(self.invert, (bool, np.bool_)):
            raise ValueError("invert 必须是布尔值")
        try:
            re.compile(self.pattern, flags=self.flags)
        except (TypeError, ValueError, re.error) as exc:
            raise ValueError(f"正则表达式配置无效: {exc}") from exc
        return super()._check_input(X, y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合正则筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        # 字段名匹配不需要访问样本值，避免额外全表缺失扫描。
        self.total_counts_ = pd.Series(len(X), index=X.columns, dtype=np.int64)
        self.threshold_ = None
        self.score_name_, self.score_direction_ = "正则匹配", "满足条件"
        matches = np.array(
            [_matches_regex_feature((col, self.pattern, self.flags))[1] for col in X.columns],
            dtype=bool,
        )

        if self.invert:
            selected_cols = X.columns[~matches].tolist()
            self.scores_ = pd.Series((~matches).astype(int), index=X.columns)
        else:
            selected_cols = X.columns[matches].tolist()
            self.scores_ = pd.Series(matches.astype(int), index=X.columns)

        self.selected_features_ = selected_cols
        self.matches_ = pd.Series(matches, index=X.columns)
        record_conditions(self, X.columns, 正则策略达标=self.scores_.astype(bool))
        self._drop_reason = f"特征名{'匹配排除' if self.invert else '不匹配'}正则表达式: {self.pattern}"
