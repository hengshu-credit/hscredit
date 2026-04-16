"""特征评估指标.

提供评估特征预测能力和质量的指标。

**参考样例**

>>> from hscredit.core.metrics import iv, iv_table, chi2_test, cramers_v
>>> import numpy as np
>>> np.random.seed(42)
>>> y_true = np.random.randint(0, 2, 1000)
>>> feature = np.random.randn(1000)
>>> print(f"IV={iv(y_true, feature):.4f}")
>>> chi2, p = chi2_test(feature, y_true)
>>> print(f"chi2={chi2:.4f}, p={p:.4f}")
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from ._base import _validate_same_length, _validate_binary_target, _create_bin_edges
from ._binning import compute_bin_stats, _chi2_by_bin


def iv(y_true: Union[np.ndarray, pd.Series],
       feature: Union[np.ndarray, pd.Series],
       method: str = 'quantile',
       max_n_bins: int = 10,
       min_bin_size: float = 0.01,
       **kwargs) -> float:
    """计算Information Value (信息价值).

    IV用于衡量特征的预测能力，值越大表示特征的区分能力越强。

    IV分级标准:

    - IV < 0.02: 无预测能力，应剔除
    - 0.02 <= IV < 0.1: 弱预测能力
    - 0.1 <= IV < 0.3: 中等预测能力
    - 0.3 <= IV < 0.5: 强预测能力
    - IV >= 0.5: 极强预测能力，但需警惕过拟合

    **参数**

    :param y_true: 目标变量 (0/1)，0为负样本（好样本），1为正样本（坏样本）
    :param feature: 特征变量（支持数值型和分类型，自动处理缺失值）
    :param method: 分箱方法，默认为'quantile'（等频分箱），
        支持与OptimalBinning相同的全部分箱方法
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: IV值
    :raises ValueError: 数据全部为缺失值或y_true非二值时
    :raises ValueError: y_true和feature长度不一致时

    **参考样例**

    >>> from hscredit.core.metrics import iv
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> y = np.random.randint(0, 2, 1000)
    >>> x = np.random.randn(1000) + y * 0.5   # 与目标有一定关联的特征
    >>> iv(y, x)
    0.15
    """
    table = iv_table(y_true, feature, method, max_n_bins, min_bin_size, **kwargs)
    return table['分档IV值'].sum()


def iv_table(y_true: Union[np.ndarray, pd.Series],
             feature: Union[np.ndarray, pd.Series],
             method: str = 'quantile',
             max_n_bins: int = 10,
             min_bin_size: float = 0.01,
             **kwargs) -> pd.DataFrame:
    """计算IV详细统计表.

    对特征进行分箱后，计算每箱的好坏样本数、占比、WOE、IV贡献等详细指标。

    **参数**

    :param y_true: 目标变量 (0/1)，0为负样本（好样本），1为正样本（坏样本）
    :param feature: 特征变量（支持数值型和分类型，自动处理缺失值）
    :param method: 分箱方法，默认为'quantile'（等频分箱）
    :param max_n_bins: 最大分箱数，默认为10
    :param min_bin_size: 每箱最小样本占比，默认为0.01
    :param kwargs: 其他传递给OptimalBinning的参数
    :return: 包含各分箱详细统计的DataFrame，列包括：分箱标签、样本数、坏样本数、好样本数、
        坏样本率、好样本率、WOE值、分档IV值、累积IV值等
    :raises ValueError: 数据全部为缺失值时

    **参考样例**

    >>> from hscredit.core.metrics import iv_table
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> y = np.random.randint(0, 2, 1000)
    >>> x = np.random.randn(1000) + y * 0.5
    >>> table = iv_table(y, x, max_n_bins=5)
    >>> print(table[['分箱标签', '样本数', '坏样本率', '分档IV值']])
    """
    y_true = np.asarray(y_true)
    feature = np.asarray(feature)

    _validate_same_length(y_true, feature, ("y_true", "feature"))
    _validate_binary_target(y_true)

    # 移除缺失值
    valid_mask = ~(pd.isna(feature) | pd.isna(y_true))
    feature_clean = feature[valid_mask]
    y_true_clean = y_true[valid_mask]

    if len(feature_clean) == 0:
        raise ValueError("没有有效数据（全部为缺失值）")

    # 使用OptimalBinning进行分箱
    from ..binning import OptimalBinning

    df = pd.DataFrame({'feature': feature_clean, 'target': y_true_clean})

    binner = OptimalBinning(
        method=method,
        max_n_bins=max_n_bins,
        min_bin_size=min_bin_size,
        verbose=False,
        **kwargs
    )
    binner.fit(df[['feature']], df['target'])
    bins = binner.transform(df[['feature']], metric='indices').values.flatten()

    bin_labels = None
    if 'feature' in binner.bin_tables_:
        bin_table = binner.bin_tables_['feature']
        if '分箱标签' in bin_table.columns:
            bin_labels = bin_table['分箱标签'].tolist()

    return compute_bin_stats(bins, y_true_clean, bin_labels=bin_labels)


def chi2_test(x: Union[np.ndarray, pd.Series],
              y: Union[np.ndarray, pd.Series]) -> Tuple[float, float]:
    """计算卡方独立性检验 (Chi-Square Test).

    检验特征变量与目标变量之间是否存在统计学显著的关联关系。

    **参数**

    :param x: 特征变量（可以是分类或数值型，数值型会自动进行等频分箱）
    :param y: 目标变量 (0/1)
    :return: 二元组 (卡方统计量, p值)
        - 卡方统计量: 值越大表示偏离独立假设越远
        - p值: 小于显著性水平（通常0.05）时拒绝独立假设
    :raises ValueError: x和y长度不一致时

    **参考样例**

    >>> from hscredit.core.metrics import chi2_test
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> y = np.random.randint(0, 2, 1000)
    >>> x = np.random.randn(1000)
    >>> chi2, p = chi2_test(x, y)
    >>> print(f"chi2={chi2:.4f}, p={p:.4f}")
    """
    x = np.asarray(x)
    y = np.asarray(y)

    _validate_same_length(x, y, ("x", "y"))

    # 如果x是数值型，进行分箱
    if np.issubdtype(x.dtype, np.number):
        bin_edges = _create_bin_edges(x, 10)
        x = np.digitize(x, bin_edges[1:-1])

    contingency = pd.crosstab(x, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0, 1.0

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    return chi2_stat, p_value


def cramers_v(x: Union[np.ndarray, pd.Series],
              y: Union[np.ndarray, pd.Series]) -> float:
    """计算Cramer's V关联强度 (Cramér's V).

    Cramer's V是卡方检验的效应量，衡量两个分类变量之间的关联强度，
    取值范围0-1，值越大表示关联越强。

    **参数**

    :param x: 特征变量（数值型会自动分箱为10个区间）
    :param y: 目标变量 (0/1)
    :return: Cramer's V值，取值范围[0, 1]
        - 0: 完全独立
        - 0.1: 弱关联
        - 0.3: 中等关联
        - 0.5+: 强关联
    :raises ValueError: x和y长度不一致时

    **参考样例**

    >>> from hscredit.core.metrics import cramers_v
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> y = np.random.randint(0, 2, 1000)
    >>> x = np.random.randn(1000)
    >>> cramers_v(x, y)
    0.05
    """
    x = np.asarray(x)
    y = np.asarray(y)

    _validate_same_length(x, y, ("x", "y"))

    # 如果x是数值型，进行分箱
    if np.issubdtype(x.dtype, np.number):
        bin_edges = _create_bin_edges(x, 10)
        x = np.digitize(x, bin_edges[1:-1])

    contingency = pd.crosstab(x, y).values

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0

    chi2, _, _, _ = stats.chi2_contingency(contingency)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1

    if min_dim == 0:
        return 0.0

    return np.sqrt(chi2 / (n * min_dim))


def feature_importance(X: Union[pd.DataFrame, np.ndarray],
                       y: Union[np.ndarray, pd.Series],
                       method: str = 'gini',
                       **kwargs) -> pd.Series:
    """计算特征重要性.

    使用树模型基于特征对目标变量的分裂增益计算特征重要性。

    **参数**

    :param X: 特征矩阵（DataFrame或numpy数组，DataFrame时使用列名作为索引名）
    :param y: 目标变量 (0/1)
    :param method: 计算方法
        - 'gini': 使用决策树（max_depth=3）基于基尼重要性计算，默认为此值
        - 'entropy': 使用随机森林（默认100棵树，max_depth=3）基于信息熵计算
    :param kwargs: 其他传递给模型的参数（如n_estimators、max_depth等）
    :return: 特征重要性Series，索引为特征名（DataFrame输入时）或feature_0, feature_1...，
        值为重要性得分（归一化和为1）

    **参考样例**

    >>> from hscredit.core.metrics import feature_importance
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = pd.DataFrame({f'f{i}': np.random.randn(500) for i in range(5)})
    >>> y = np.random.randint(0, 2, 500)
    >>> importance = feature_importance(X, y, method='gini')
    >>> print(importance.sort_values(ascending=False))
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    if method == 'gini':
        max_depth = kwargs.get('max_depth', 3)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif method == 'entropy':
        n_estimators = kwargs.get('n_estimators', 100)
        max_depth = kwargs.get('max_depth', 3)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=feature_names)


def feature_summary(feature: Union[np.ndarray, pd.Series],
                   y: Optional[Union[np.ndarray, pd.Series]] = None) -> pd.DataFrame:
    """计算特征描述统计.

    提供特征的完整描述性统计信息，包括样本量、缺失情况、分布特征，
    以及（当提供目标变量时）IV值和目标关联统计。

    **参数**

    :param feature: 特征变量（支持数值型和分类型）
    :param y: 目标变量（可选，0/1。如果提供则计算IV和目标关联统计）
    :return: 单行DataFrame，包含以下列（数值型特征额外包含均值、标准差、最小最大值、中位数）：
        - 样本数: 特征总样本数
        - 缺失数: 缺失值数量
        - 缺失率: 缺失值占比
        - 唯一值数: 不同取值的数量
        - 均值/标准差/最小值/最大值/中位数: （仅数值型特征）
        - IV: （仅当y不为None时）

    **参考样例**

    >>> from hscredit.core.metrics import feature_summary
    >>> import numpy as np
    >>> feature = np.random.randn(1000)
    >>> print(feature_summary(feature))
    >>> y = np.random.randint(0, 2, 1000)
    >>> print(feature_summary(feature, y))
    """
    feature = np.asarray(feature)

    result = {
        '样本数': len(feature),
        '缺失数': np.sum(pd.isna(feature)),
        '缺失率': np.mean(pd.isna(feature)),
        '唯一值数': len(np.unique(feature[~pd.isna(feature)])),
    }

    # 数值型统计
    if np.issubdtype(feature.dtype, np.number):
        valid_feature = feature[~np.isnan(feature)]
        result.update({
            '均值': np.mean(valid_feature),
            '标准差': np.std(valid_feature),
            '最小值': np.min(valid_feature),
            '最大值': np.max(valid_feature),
            '中位数': np.median(valid_feature),
        })

    # 如果有目标变量，计算相关性
    if y is not None:
        y = np.asarray(y)
        valid_mask = ~(pd.isna(feature) | pd.isna(y))
        if valid_mask.sum() > 0:
            # 计算IV
            try:
                iv_value = iv(y[valid_mask], feature[valid_mask])
                result['IV'] = iv_value
            except:
                result['IV'] = np.nan

    return pd.DataFrame([result])
