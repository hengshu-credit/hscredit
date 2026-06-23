"""特征描述统计工具.

提供特征描述统计功能。
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial


def feature_describe(
    data: pd.DataFrame,
    feature: Optional[str] = None,
    percentiles: Optional[List[float]] = None,
    missing=None,
    cardinality: Optional[int] = None
) -> pd.Series:
    """特征描述统计（数值型与类别型自动区分）。

    对数值型特征输出 样本数/非空数/查得率/最小值/平均值/各分位数/最大值；
    对类别型特征（或唯一值数 ≤ ``cardinality``）输出 样本数/非空数/查得率
    及各取值占比。

    :param data: 输入数据，``DataFrame``（需配合 ``feature`` 指定列）或单列 ``Series``
    :param feature: 特征列名，仅当 ``data`` 为 DataFrame 时需要；
        为 None 时把 ``data`` 整体当作 Series 处理
    :param percentiles: 分位数列表（0~1 之间），默认
        ``[0.01, 0.02, 0.03, 0.05, 0.1, 0.2, ..., 0.95, 0.97, 0.98, 0.99]``
    :param missing: 缺失值标记（标量或列表），统计前会被替换为 ``np.nan``；默认 None 不替换
    :param cardinality: 基数阈值（正整数），唯一值数 ≤ 该值时强制按类别型统计；
        默认 None 表示仅按 dtype 判断
    :return: 描述统计结果 ``Series``（``name`` 为特征名）
    :raises ValueError: ``feature`` 不在 ``data`` 列中，或 ``cardinality`` < 1 时

    **参考样例**

    >>> feature_describe(df, feature='age')          # 指定列
    >>> feature_describe(df['age'])                  # 直接传 Series
    >>> feature_describe(df, feature='city', cardinality=20)   # 强制按类别统计
    >>> feature_describe(df, feature='income', missing=-999)   # -999 视为缺失
    """
    if feature and feature not in data.columns:
        raise ValueError(f"feature {feature} must in columns.")

    if cardinality and cardinality < 1:
        raise ValueError(f"cardinality must greater than 1")

    if percentiles is None:
        percentiles = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]

    if feature:
        series = data[feature]
    else:
        series = data.copy()

    if missing:
        series = series.replace(missing, np.nan)

    # 类别型特征处理
    if (cardinality and series.nunique() <= cardinality) or \
       not pd.api.types.is_numeric_dtype(series):
        describe = {
            "样本数": len(series),
            "非空数": len(series) - series.isnull().sum(),
            "查得率": 1 - series.isnull().mean(),
        }
        value_counts = series.replace(np.nan, '缺失值').value_counts(dropna=False)
        describe.update((value_counts / len(series)).to_dict())
        return pd.Series(describe, name=feature)
    else:
        # 数值型特征处理
        describe = {
            "样本数": len(series),
            "非空数": len(series) - series.isnull().sum(),
            "查得率": 1 - series.isnull().mean(),
            "最小值": series.min(),
            "平均值": series.mean(),
            "最大值": series.max(),
        }
        quantile = series.quantile(percentiles)
        quantile.index = [f"{int(i * 100)}%" for i in percentiles]
        describe.update(quantile.to_dict())

        # 按固定顺序排列
        order = ['样本数', '非空数', '查得率', '最小值', '平均值'] + \
                [f"{int(i * 100)}%" for i in percentiles] + ['最大值']
        return pd.Series(describe, name=feature).reindex(order)


def groupby_feature_describe(
    data: pd.DataFrame,
    by=None,
    n_jobs: int = -1,
    **kwargs
) -> pd.DataFrame:
    """按分组进行特征描述统计。

    :param data: 数据DataFrame
    :param by: 分组字段或字段列表
    :param n_jobs: 并行任务数，-1表示使用所有CPU
    :param kwargs: 传递给feature_describe的其他参数
    :return: 描述统计结果DataFrame

    **参考样例**

    >>> groupby_feature_describe(df, by='gender')
    >>> groupby_feature_describe(df, by=['gender', 'age_group'])
    """
    if not isinstance(by, (tuple, list, np.ndarray)):
        by = [by]

    describe = pd.DataFrame()

    def __feature_describe(group, _p, f, **kwargs):
        temp = feature_describe(group[f], **kwargs)
        temp.index = pd.MultiIndex.from_product([[f], temp.index])
        temp = pd.DataFrame(temp, columns=[_p])
        return temp

    def _feature_describe(group, _p, _by=None, **kwargs):
        if len(_p) <= 1:
            _p = _p[0]

        __describe = partial(lambda f: __feature_describe(group, _p, f, **kwargs))
        return _p, pd.concat(
            Parallel(n_jobs=n_jobs)(
                delayed(__describe)(f) for f in group.columns if f not in _by
            )
        )[_p]

    for info in Parallel(n_jobs=n_jobs)(
        delayed(_feature_describe)(group, p, _by=by, **kwargs)
        for p, group in data.groupby(by=by)
    ):
        describe[info[0]] = info[1]

    if len(by) > 1:
        describe.columns = pd.MultiIndex.from_tuples(describe.columns)

    describe.index.names = ["特征名称", "统计指标"]

    return describe
