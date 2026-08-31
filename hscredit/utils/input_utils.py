"""输入数据处理工具函数.

提供统一的输入处理函数，用于处理两种API风格的输入数据：
1. sklearn风格: fit(X, y)
2. scorecardpipeline风格: fit(df) - target列在df中
"""

from typing import Union, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd

from ..exceptions import FeatureNotFoundError, InputTypeError, InputValidationError


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, List]


def normalize_dpd_values(dpds) -> List[Union[int, float]]:
    """规范化 DPD 阈值，保留非整数小数并按首次出现顺序去重。

    标量会转换为单元素列表，``None`` 按既有报告口径转换为 ``[0]``；
    numpy/pandas 标量与常用序列输入均受支持。整数值统一输出为 Python
    ``int``，非整数值输出为 Python ``float``。
    """
    if isinstance(dpds, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        values = list(dpds)
    else:
        values = [dpds] if dpds is not None else [0]

    normalized_values: List[Union[int, float]] = []
    seen = set()
    for value in values:
        if value is None:
            normalized: Union[int, float] = 0
        else:
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"DPD 阈值必须是有限数值，收到: {value!r}") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"DPD 阈值必须是有限数值，收到: {value!r}")
            normalized = int(numeric) if numeric.is_integer() else numeric
        if normalized not in seen:
            seen.add(normalized)
            normalized_values.append(normalized)
    return normalized_values


def check_xy_inputs(
    X: ArrayLike,
    y: Optional[Union[np.ndarray, pd.Series, List]] = None,
    target: str = 'target',
    accept_numpy: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """统一检查并处理 X 和 y 输入（适配双 API 调用风格）。

    支持两种 API 风格，并按 ``传入的 y > 从 X 提取 target 列`` 的优先级确定目标变量：

    - sklearn 风格：``fit(X, y)``，X、y 分别传入
    - scorecardpipeline 风格：``fit(df)``，目标列包含在 df 中、由 ``target`` 指定

    :param X: 输入数据，``DataFrame`` 或 numpy 数组（自动转为 DataFrame）
    :param y: 目标变量（可选），提供时优先于从 X 提取 target 列
    :param target: 目标列名，当 ``y`` 为 None 时从 X 中提取，默认 ``'target'``
    :param accept_numpy: 是否接受 numpy 数组输入，默认 True
    :return: 三元组 ``(X_df, y_series, feature_names)``：

        - ``X_df`` (DataFrame)：处理后的特征数据
        - ``y_series`` (Series)：目标变量
        - ``feature_names`` (list)：特征名称列表

    :raises InputValidationError: X 与 y 长度不匹配，或数据为空时
    :raises FeatureNotFoundError: y 为 None 且 X 中无 ``target`` 列时
    :raises InputTypeError: 输入类型不受支持时

    **参考样例**

    >>> # sklearn 风格
    >>> X_df, y_series, features = check_xy_inputs(X_train, y_train)
    >>>
    >>> # scorecardpipeline 风格
    >>> X_df, y_series, features = check_xy_inputs(df, target='target')
    """
    # 1. 转换X为DataFrame
    X_df = convert_to_dataframe(X)
    
    # 2. 确定目标变量y
    if y is not None:
        # sklearn风格: 使用传入的y
        y_series = _convert_to_series(y, name=target)
        
        # 检查长度是否匹配
        if len(X_df) != len(y_series):
            raise InputValidationError(
                f"X和y的长度不匹配: X有{len(X_df)}行, y有{len(y_series)}个元素"
            )
    else:
        # scorecardpipeline风格: 从X中提取target列
        if target not in X_df.columns:
            raise FeatureNotFoundError(
                f"当y为None时，X中必须包含target列 '{target}'，"
                f"当前列: {list(X_df.columns)}"
            )
        X_df, y_series = extract_target_from_df(X_df, target=target, drop=True)
    
    # 3. 获取特征名称
    feature_names = list(X_df.columns)
    
    # 4. 验证数据
    _validate_data(X_df, y_series)
    
    return X_df, y_series, feature_names


def convert_to_dataframe(
    X: ArrayLike,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """将输入转换为 DataFrame。

    支持 ``DataFrame`` / numpy 数组（1 维或 2 维）/ ``Series`` / ``list`` / ``tuple``。
    1 维输入转为单列 DataFrame；DataFrame 与 Series 会复制后返回。

    :param X: 输入数据，不能为 None
    :param columns: 列名列表，仅当 X 不是 DataFrame 时使用；为 None 时自动生成
        ``feature_0``、``feature_1`` …
    :return: 转换后的 ``DataFrame``
    :raises InputValidationError: X 为 None、维度 >2，或列名数量与列数不匹配时
    :raises InputTypeError: 输入类型不受支持时

    **参考样例**

    >>> df = convert_to_dataframe(np.array([[1, 2], [3, 4]]), columns=['a', 'b'])
    >>> df = convert_to_dataframe([[1, 2], [3, 4]])
    """
    if X is None:
        raise InputValidationError("输入数据X不能为None")
    
    # 已经是DataFrame
    if isinstance(X, pd.DataFrame):
        return X.copy()
    
    # numpy数组
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            # 一维数组转为单列DataFrame
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise InputValidationError(f"不支持{X.ndim}维数组，只支持1维或2维数组")
        
        # 生成默认列名或验证列名
        if columns is None:
            columns = [f'feature_{i}' for i in range(X.shape[1])]
        elif len(columns) != X.shape[1]:
            raise InputValidationError(
                f"列名数量({len(columns)})与数据列数({X.shape[1]})不匹配"
            )
        
        return pd.DataFrame(X, columns=columns)
    
    # pandas Series
    if isinstance(X, pd.Series):
        return X.to_frame()
    
    # list或其他可迭代对象
    if isinstance(X, (list, tuple)):
        arr = np.array(X)
        return convert_to_dataframe(arr, columns=columns)
    
    raise InputTypeError(
        f"不支持的输入类型: {type(X).__name__}，"
        "只支持DataFrame, numpy数组, list, tuple或Series"
    )


def extract_target_from_df(
    df: pd.DataFrame,
    target: str = 'target',
    drop: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """从 DataFrame 中提取目标变量。

    :param df: 输入 ``DataFrame``
    :param target: 目标列名，默认 ``'target'``
    :param drop: 是否从返回的特征表中删除 ``target`` 列，默认 True
    :return: 二元组 ``(X, y)``，``X`` 为特征 DataFrame，``y`` 为目标 Series
    :raises InputTypeError: 输入不是 DataFrame 时
    :raises FeatureNotFoundError: ``target`` 列不存在时

    **参考样例**

    >>> X, y = extract_target_from_df(df, target='target')
    >>> X, y = extract_target_from_df(df, target='label', drop=False)
    """
    if not isinstance(df, pd.DataFrame):
        raise InputTypeError(f"输入必须是DataFrame，而不是{type(df).__name__}")
    
    if target not in df.columns:
        raise FeatureNotFoundError(
            f"DataFrame中不存在target列 '{target}'，"
            f"可用列: {list(df.columns)}"
        )
    
    # 提取目标变量
    y = df[target].copy()
    
    # 创建特征DataFrame
    if drop:
        X = df.drop(columns=[target]).copy()
    else:
        X = df.copy()
    
    return X, y


def _convert_to_series(
    y: Union[np.ndarray, pd.Series, List],
    name: str = 'target'
) -> pd.Series:
    """将y转换为pandas Series.
    
    Args:
        y: 目标变量
        name: Series的名称
        
    Returns:
        转换后的Series
        
    Raises:
        TypeError: 当输入类型不支持时
    """
    if isinstance(y, pd.Series):
        return y.copy()
    
    if isinstance(y, np.ndarray):
        if y.ndim > 1:
            # 处理二维数组（如(n, 1)）
            y = y.ravel()
        return pd.Series(y, name=name)
    
    if isinstance(y, (list, tuple)):
        return pd.Series(y, name=name)
    
    raise InputTypeError(
        f"不支持的y类型: {type(y).__name__}，"
        "只支持numpy数组, list, tuple或Series"
    )


def _validate_data(
    X: pd.DataFrame,
    y: pd.Series
) -> None:
    """验证X和y的数据有效性.
    
    Args:
        X: 特征DataFrame
        y: 目标Series
        
    Raises:
        ValueError: 当数据验证失败时
    """
    # 检查空数据
    if X.empty:
        raise InputValidationError("特征数据X为空")
    
    if y.empty:
        raise InputValidationError("目标变量y为空")
    
    # 检查索引一致性
    if not X.index.equals(y.index):
        warnings.warn(
            "X和y的索引不一致，将使用X的索引",
            UserWarning
        )
        y.index = X.index
    
    # 检查目标变量是否全为缺失值
    if y.isna().all():
        raise InputValidationError("目标变量y全部为缺失值")
    
    # 检查特征是否全为缺失值
    if X.isna().all().all():
        raise InputValidationError("特征数据X全部为缺失值")
    
    # 检查目标变量的类型（二分类问题的常见检查）
    unique_values = y.dropna().unique()
    if len(unique_values) < 2:
        warnings.warn(
            f"目标变量y只有{len(unique_values)}个唯一值: {unique_values}，"
            "这可能不是分类问题",
            UserWarning
        )


def check_array_1d(
    arr: ArrayLike,
    name: str = 'array'
) -> pd.Series:
    """检查并转换为一维 ``Series``。

    :param arr: 输入数组，``ndarray`` / ``Series`` / ``list`` / ``tuple``；
        二维数组会被展平为一维
    :param name: 数组名称，用于生成错误信息与 Series 名称，默认 ``'array'``
    :return: 一维 ``Series``
    :raises InputValidationError: 数组为空时
    :raises InputTypeError: 类型不受支持时

    **参考样例**

    >>> s = check_array_1d([0, 1, 1, 0], name='y')
    """
    series = _convert_to_series(arr, name=name)
    
    if series.empty:
        raise InputValidationError(f"{name}不能为空")
    
    return series


def get_feature_dtypes(
    X: pd.DataFrame
) -> dict:
    """获取特征的数据类型分类（数值型 / 类别型）。

    :param X: 特征 ``DataFrame``
    :return: 字典，含三个键：

        - ``'numeric'``：数值型特征名列表（int/uint/float 各精度）
        - ``'categorical'``：非数值型特征名列表（object、category、bool 等）
        - ``'all'``：全部特征名列表

    **参考样例**

    >>> dtypes = get_feature_dtypes(df)
    >>> numeric_features = dtypes['numeric']
    >>> categorical_features = dtypes['categorical']
    """
    numeric_dtypes = ['int8', 'int16', 'int32', 'int64',
                      'uint8', 'uint16', 'uint32', 'uint64',
                      'float16', 'float32', 'float64']
    
    numeric_features = []
    categorical_features = []
    
    for col in X.columns:
        if X[col].dtype.name in numeric_dtypes:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'all': list(X.columns)
    }


def check_missing_values(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    raise_error: bool = False
) -> dict:
    """检查缺失值情况。

    :param X: 特征 ``DataFrame``
    :param y: 目标 ``Series``（可选），提供时一并统计其缺失情况
    :param raise_error: 存在缺失值时是否抛出异常，默认 False（仅统计不报错）
    :return: 缺失值统计字典，含键 ``X_missing``（总缺失数）、
        ``X_missing_by_col``（各列缺失数）、``X_missing_ratio``（整体缺失率）；
        当 ``y`` 非 None 时另含 ``y_missing``、``y_missing_ratio``
    :raises InputValidationError: ``raise_error=True`` 且 X 或 y 存在缺失值时

    **参考样例**

    >>> stats = check_missing_values(X, y)
    >>> print(stats['X_missing_ratio'])
    >>> check_missing_values(X, raise_error=True)   # 有缺失则报错
    """
    result = {
        'X_missing': X.isna().sum().sum(),
        'X_missing_by_col': X.isna().sum().to_dict(),
        'X_missing_ratio': X.isna().sum().sum() / (X.shape[0] * X.shape[1]),
    }
    
    if y is not None:
        result['y_missing'] = y.isna().sum()
        result['y_missing_ratio'] = y.isna().sum() / len(y)
    
    if raise_error and result['X_missing'] > 0:
        raise InputValidationError(f"特征数据X存在{result['X_missing']}个缺失值")
    
    if raise_error and y is not None and result['y_missing'] > 0:
        raise InputValidationError(f"目标变量y存在{result['y_missing']}个缺失值")
    
    return result
