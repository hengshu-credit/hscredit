"""稳定性指标计算.

提供评估模型稳定性和分布变化的指标。

**参考样例**

>>> from hscredit.core.metrics import psi, psi_table, batch_psi, psi_rating
>>> import numpy as np
>>> np.random.seed(42)
>>> train = np.random.randn(1000)
>>> test = np.random.randn(1000) + 0.5  # 测试集分布偏移
>>> print(f"PSI={psi(train, test):.4f}")
>>> print(psi_rating(psi(train, test)))
>>> print(psi_table(train, test).head())
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
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: PSI值

    **参考样例**

    >>> from hscredit.core.metrics import psi, psi_rating
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> train_scores = np.random.randn(1000)
    >>> test_scores = np.random.randn(1000) + 0.3
    >>> p = psi(train_scores, test_scores)
    >>> print(f"PSI={p:.4f}, 评级: {psi_rating(p)}")
    """
    table = psi_table(expected, actual, method, max_n_bins, min_bin_size, **kwargs)
    return table['PSI贡献'].sum()


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
    :param kwargs: 其他传递给OptimalBinning的参数
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
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # 移除缺失值
    expected_clean = expected[~pd.isna(expected)]
    actual_clean = actual[~pd.isna(actual)]

    # 合并数据确定分箱边界
    combined = np.concatenate([expected_clean, actual_clean])

    # 使用OptimalBinning进行分箱
    from ..binning import OptimalBinning

    # 构建DataFrame
    df_expected = pd.DataFrame({'value': expected_clean, 'is_expected': 1})
    df_actual = pd.DataFrame({'value': actual_clean, 'is_expected': 0})
    df_combined = pd.concat([df_expected, df_actual], ignore_index=True)

    # 创建临时目标用于分箱
    dummy_target = np.random.randint(0, 2, size=len(df_combined))

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df_combined[['value']], dummy_target)

    # 分别转换expected和actual
    bins_expected = binner.transform(df_expected[['value']], metric='indices').values.flatten()
    bins_actual = binner.transform(df_actual[['value']], metric='indices').values.flatten()

    # 计算每个箱的统计
    unique_bins = sorted(set(bins_expected) | set(bins_actual))

    results = []
    epsilon = 1e-10

    total_expected = len(expected_clean)
    total_actual = len(actual_clean)

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

    return pd.DataFrame(results)


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
