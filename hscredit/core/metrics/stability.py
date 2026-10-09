"""稳定性指标计算.

提供评估模型稳定性和分布变化的指标。

**参考样例**

>>> from hscredit.core.metrics import psi, psi_table, batch_psi, psi_rating
>>> import numpy as np
>>> np.random.seed(42)
>>> train = np.random.randn(1000)  # 训练集分布（期望）
>>> test = np.random.randn(1000) + 0.5  # 测试集分布偏移（实际）
>>> print(f"PSI={psi(train, test):.4f}")  # PSI>0.25表示分布变化显著
>>> print(psi_rating(psi(train, test)))  # 稳定性评级
>>> print(psi_table(train, test).head())

**引用**

- PSI（群体稳定性指标）/ CSI（特征稳定性指标）的定义与分级阈值见
  Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley；
  另见 Yurdakul, B. (2018). *Statistical Properties of Population Stability
  Index (PSI).* PhD Dissertation, Western Michigan University。
- PSI 本质为期望分布与实际分布之间的对称 KL 散度（Jensen 散度）之和，
  公式：PSI = Σ (实际占比 − 期望占比) · ln(实际占比 / 期望占比)。
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from scipy.stats import chi2_contingency

from ._base import _create_bin_edges


def psi(expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        method: str = 'quantile',
        max_n_bins: int = 10,
        min_bin_size: float = 0.01,
        **kwargs) -> float:
    """计算Population Stability Index (群体稳定性指标).

    PSI用于衡量两个分布之间的差异，评估模型或特征的稳定性。
    值越小表示两个分布越接近，模型越稳定。

    PSI分级标准:

    - PSI < 0.1: 没有显著变化，分布稳定
    - 0.1 <= PSI < 0.25: 有轻微变化，需关注
    - PSI >= 0.25: 有显著变化，模型可能需要重新训练

    **参数**

    :param expected: 期望分布数据（通常是训练集或基准数据的特征/评分）
    :param actual: 实际分布数据（通常是测试集或新上线数据的特征/评分）
    :param method: 分箱方法，默认为'quantile'（等频分箱）
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: 可传baseline=MonitoringBaseline复用冻结基准，include_missing默认True。
        默认quantile仅在基准期拟合；其他有监督方法保留两期联合比较口径，详情见结果attrs。
    :return: PSI值

    **参考样例**

    >>> from hscredit.core.metrics import psi, psi_rating
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> train_scores = np.random.randn(1000)  # 训练集评分分布
    >>> test_scores = np.random.randn(1000) + 0.3  # 测试集评分分布偏移
    >>> p = psi(train_scores, test_scores)  # 计算PSI评估分布稳定性
    >>> print(f"PSI={p:.4f}, 评级: {psi_rating(p)}")
    """
    table = psi_table(expected, actual, method, max_n_bins, min_bin_size, **kwargs)
    return table['PSI贡献'].sum(min_count=1)


def psi_table(expected: Union[np.ndarray, pd.Series],
              actual: Union[np.ndarray, pd.Series],
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """计算PSI详细统计表.

    返回每个分箱的期望占比、实际占比及PSI贡献，用于分析分布变化的具体来源。

    **参数**

    :param expected: 期望分布数据（通常是训练集或基准数据）
    :param actual: 实际分布数据（通常是测试集或新数据）
    :param method: 分箱方法，默认为'quantile'（等频分箱）
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: baseline为已拟合MonitoringBaseline；include_missing=False显式计算非缺失条件分布。
        默认quantile冻结基准；其他方法保留两期联合分箱并在attrs标记。
    :return: 包含各分箱详细统计的DataFrame，列包括：
        - 分箱: 分箱标签
        - 期望样本数: 该分箱内期望数据量
        - 实际样本数: 该分箱内实际数据量
        - 期望占比: 期望样本占总样本比例
        - 实际占比: 实际样本占总样本比例
        - PSI贡献: 该分箱对总PSI的贡献

    **参考样例**

    >>> from hscredit.core.metrics import psi_table
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> train = np.random.randn(1000)
    >>> test = np.random.randn(1000) + 0.5
    >>> table = psi_table(train, test)
    >>> print(table)
    """
    baseline = kwargs.pop('baseline', None)
    include_missing = kwargs.pop('include_missing', True)
    if not isinstance(include_missing, (bool, np.bool_)):
        raise ValueError("include_missing 必须为布尔值")
    if baseline is not None:
        return baseline.evaluate(actual, include_missing=include_missing)['分箱明细']
    if method == 'quantile':
        from .monitoring import MonitoringBaseline
        reference = pd.Series(expected)
        current = pd.Series(actual)
        return MonitoringBaseline(max_n_bins=max_n_bins, min_bin_size=min_bin_size, binning_params=kwargs).fit(reference).evaluate(current, include_missing=include_missing)['分箱明细']
    # 先验证公开方法和构造参数；空/全缺失数据只改变统计路径，不能绕过配置校验。
    from ..binning import OptimalBinning
    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.ndim != 1 or actual.ndim != 1:
        raise ValueError("PSI 输入必须是一维数据")

    # 移除缺失值
    expected_clean = expected[~pd.isna(expected)]
    actual_clean = actual[~pd.isna(actual)]
    total_expected = len(expected) if include_missing else len(expected_clean)
    total_actual = len(actual) if include_missing else len(actual_clean)

    if len(expected) == 0:
        raise ValueError("PSI 基准不能为空")
    if len(expected_clean) == 0 or len(actual_clean) == 0:
        from .monitoring import MonitoringBaseline
        return MonitoringBaseline(max_n_bins=max_n_bins).fit(expected).evaluate(actual, include_missing=include_missing)['分箱明细']

    # 构建DataFrame
    df_expected = pd.DataFrame({'value': expected_clean, 'is_expected': 1})
    df_actual = pd.DataFrame({'value': actual_clean, 'is_expected': 0})
    df_combined = pd.concat([df_expected, df_actual], ignore_index=True)

    # PSI 的分箱边界必须可复现。对于有监督分箱方法，用“基准/实际”作为确定性目标，
    # 避免随机 dummy target 让同一输入多次调用得到不同分箱和 PSI。
    binner.fit(df_combined[['value']], df_combined['is_expected'])

    # 分别转换expected和actual
    bins_expected = binner.transform(df_expected[['value']], metric='indices').values.flatten()
    bins_actual = binner.transform(df_actual[['value']], metric='indices').values.flatten()

    # 计算每个箱的统计
    unique_bins = sorted(set(bins_expected) | set(bins_actual))

    results = []
    epsilon = 1e-10

    for bin_idx in unique_bins:
        expected_count = np.sum(bins_expected == bin_idx)
        actual_count = np.sum(bins_actual == bin_idx)

        expected_prop = expected_count / total_expected if total_expected > 0 else epsilon
        actual_prop = actual_count / total_actual if total_actual > 0 else epsilon

        # 避免除零
        expected_prop = max(expected_prop, epsilon)
        actual_prop = max(actual_prop, epsilon)

        psi_contrib = (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)

        # 获取分箱标签
        bin_label = f"Bin_{bin_idx}"
        if 'value' in binner.bin_tables_:
            bin_table = binner.bin_tables_['value']
            if bin_idx < len(bin_table) and '分箱标签' in bin_table.columns:
                bin_label = bin_table.iloc[bin_idx]['分箱标签']

        results.append({
            '分箱': bin_label,
            '期望样本数': expected_count,
            '实际样本数': actual_count,
            '期望占比': expected_prop,
            '实际占比': actual_prop,
            'PSI贡献': psi_contrib,
        })

    if include_missing:
        exp_missing = int(pd.isna(expected).sum())
        act_missing = int(pd.isna(actual).sum())
        if exp_missing or act_missing:
            ep = max(exp_missing / total_expected, epsilon)
            ap = max(act_missing / total_actual, epsilon) if total_actual else epsilon
            results.append({'分箱': '缺失值', '期望样本数': exp_missing, '实际样本数': act_missing,
                            '期望占比': exp_missing / total_expected,
                            '实际占比': act_missing / total_actual if total_actual else 0.,
                            'PSI贡献': (ap - ep) * np.log(ap / ep)})
    result = pd.DataFrame(results)
    if total_actual == 0:
        result['PSI贡献'] = np.nan
    result.attrs.update({'分箱口径': '两期联合分箱', '缺失策略': '独立分箱' if include_missing else '排除',
                         '状态': '成功' if total_actual else '数据不足'})
    return result


def psi_rating(psi_value: float) -> str:
    """根据PSI值返回稳定性评级.

    **参数**

    :param psi_value: PSI值（通常由psi()函数计算得到）
    :return: 稳定性评级描述字符串
        - PSI < 0.1: "没有显著变化 (PSI < 0.1)"
        - 0.1 <= PSI < 0.25: "有轻微变化 (0.1 <= PSI < 0.25)"
        - PSI >= 0.25: "有显著变化 (PSI >= 0.25)"

    **参考样例**

    >>> from hscredit.core.metrics import psi_rating
    >>> psi_rating(0.05)
    '没有显著变化 (PSI < 0.1)'
    >>> psi_rating(0.3)
    '有显著变化 (PSI >= 0.25)'
    """
    if not np.isfinite(psi_value):
        return "数据不足或不可计算"
    if psi_value < 0.1:
        return "没有显著变化 (PSI < 0.1)"
    elif psi_value < 0.25:
        return "有轻微变化 (0.1 <= PSI < 0.25)"
    else:
        return "有显著变化 (PSI >= 0.25)"


def csi(expected: Union[np.ndarray, pd.Series],
        actual: Union[np.ndarray, pd.Series],
        method: str = 'quantile',
        max_n_bins: int = 10,
        min_bin_size: float = 0.01,
        **kwargs) -> float:
    """计算Characteristic Stability Index (特征稳定性指标).

    CSI是PSI的变体，专门用于衡量单个特征分布的稳定性。
    与PSI的区别在于CSI通常针对单一特征，而非模型评分。

    **参数**

    :param expected: 期望分布数据（通常是训练集的特征数据）
    :param actual: 实际分布数据（通常是测试集或新数据的特征）
    :param method: 分箱方法，默认为'quantile'（等频分箱）
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: CSI值（计算方法与PSI相同）
    :raises ValueError: 数据为空时

    **参考样例**

    >>> from hscredit.core.metrics import csi
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> train = np.random.randn(1000)
    >>> test = np.random.randn(1000) + 0.5
    >>> csi(train, test)
    0.34
    """
    return psi(expected, actual, method, max_n_bins, min_bin_size, **kwargs)


def csi_table(expected: Union[np.ndarray, pd.Series],
              actual: Union[np.ndarray, pd.Series],
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """计算CSI详细统计表.

    **参数**

    :param expected: 期望分布数据（通常是训练集的特征数据）
    :param actual: 实际分布数据（通常是测试集或新数据的特征）
    :param method: 分箱方法，默认为'quantile'（等频分箱）
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 包含各分箱详细统计的DataFrame，列与psi_table相同，PSI贡献列重命名为CSI贡献

    **参考样例**

    >>> from hscredit.core.metrics import csi_table
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> train = np.random.randn(1000)
    >>> test = np.random.randn(1000) + 0.5
    >>> table = csi_table(train, test)
    >>> print(table)
    """
    table = psi_table(expected, actual, method, max_n_bins, min_bin_size, **kwargs)
    table = table.rename(columns={'PSI贡献': 'CSI贡献'})
    return table


def batch_psi(X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              features: Optional[List[str]] = None,
              method: str = 'quantile',
              max_n_bins: int = 10,
              min_bin_size: float = 0.01,
              **kwargs) -> pd.DataFrame:
    """批量计算多特征的PSI.

    对指定的多个特征同时计算PSI，返回各特征的PSI值和稳定性评级。

    **参数**

    :param X_train: 训练集特征DataFrame
    :param X_test: 测试集特征DataFrame（与X_train列结构一致）
    :param features: 需要计算PSI的特征列表，默认为None（计算全部共同列）
    :param method: 分箱方法，默认为'quantile'（等频分箱）
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 包含各特征PSI结果的DataFrame，列包括：
        - 特征: 特征名称
        - PSI: PSI值
        - 评级: 稳定性评级（由psi_rating函数返回）

    **参考样例**

    >>> from hscredit.core.metrics import batch_psi
    >>> import numpy as np
    >>> import pandas as pd
    >>> np.random.seed(42)
    >>> cols = ['age', 'income', 'credit_score']
    >>> X_train = pd.DataFrame(np.random.randn(1000, 3), columns=cols)
    >>> X_test = pd.DataFrame(np.random.randn(1000, 3) + 0.5, columns=cols)
    >>> result = batch_psi(X_train, X_test)
    >>> print(result)
    """
    if features is None:
        features = list(X_train.columns)

    results = []
    for feature in features:
        if feature in X_train.columns and feature in X_test.columns:
            try:
                psi_value = psi(
                    X_train[feature], X_test[feature],
                    method, max_n_bins, min_bin_size, **kwargs
                )
                rating = psi_rating(psi_value)
                results.append({
                    '特征': feature,
                    'PSI': psi_value,
                    '评级': rating,
                })
            except Exception as e:
                results.append({
                    '特征': feature,
                    'PSI': np.nan,
                    '评级': f'计算失败: {str(e)}',
                })

    return pd.DataFrame(results)
