"""EDA工具函数.

提供各模块共用的工具函数和常量定义.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union


# ==================== 评级标准 ====================

IV_RATING = [
    (0, 0.02, '无预测能力'),
    (0.02, 0.1, '弱预测能力'),
    (0.1, 0.3, '中等预测能力'),
    (0.3, 0.5, '强预测能力'),
    (0.5, float('inf'), '极强(需检查)'),
]

PSI_RATING = [
    (0, 0.1, '非常稳定'),
    (0.1, 0.25, '相对稳定'),
    (0.25, float('inf'), '不稳定'),
]

VIF_RATING = [
    (0, 5, '无共线性'),
    (5, 10, '中度共线性'),
    (10, float('inf'), '严重共线性'),
]

CORR_RATING = [
    (0, 0.3, '弱相关'),
    (0.3, 0.7, '中等相关'),
    (0.7, 0.9, '强相关'),
    (0.9, float('inf'), '极强相关'),
]


# ==================== 评级函数 ====================

def iv_rating(iv: float) -> str:
    """IV值评级.
    
    :param iv: IV值
    :return: 评级字符串
    """
    for low, high, rating in IV_RATING:
        if low <= iv < high:
            return rating
    return '极强(需检查)'


def psi_rating(psi: float) -> str:
    """PSI值评级.
    
    :param psi: PSI值
    :return: 评级字符串
    """
    for low, high, rating in PSI_RATING:
        if low <= psi < high:
            return rating
    return '不稳定'


def vif_rating(vif: float) -> str:
    """VIF值评级.
    
    :param vif: VIF值
    :return: 评级字符串
    """
    for low, high, rating in VIF_RATING:
        if low <= vif < high:
            return rating
    return '严重共线性'


def corr_rating(corr: float) -> str:
    """相关系数评级.
    
    :param corr: 相关系数绝对值
    :return: 评级字符串
    """
    corr = abs(corr)
    for low, high, rating in CORR_RATING:
        if low <= corr < high:
            return rating
    return '极强相关'


# ==================== 数据类型推断 ====================

def infer_feature_types(df: pd.DataFrame,
                        categorical_threshold: int = 20,
                        unique_ratio_threshold: float = 0.05,
                        numeric_as_categorical: Optional[List[str]] = None,
                        force_numeric: Optional[List[str]] = None) -> Dict[str, str]:
    """推断特征类型.

    默认严格按照实际数据类型判断，不做任何自动转换：
    - 数值类型（int/float）-> 'numerical'
    - 非数值类型（object/string/category）-> 'categorical'
    - 日期时间类型 -> 'datetime'
    - 常数特征 -> 'constant'
    - ID特征 -> 'id'

    仅当用户指定参数时才进行特殊处理：
    - numeric_as_categorical: 将指定的数值列视为 categorical
    - force_numeric: 将指定的列视为 numerical

    :param df: 输入数据
    :param categorical_threshold: 保留参数，不再用于默认类型判断
    :param unique_ratio_threshold: 保留参数，不再用于默认类型判断
    :param numeric_as_categorical: 强制视为分类变量的数值列名列表
    :param force_numeric: 强制视为数值变量的列名列表
    :return: 特征类型字典 {特征名: 类型}
        类型包括: 'numerical', 'categorical', 'datetime', 'constant', 'id'
    """
    feature_types = {}
    numeric_as_categorical = set(numeric_as_categorical or [])
    force_numeric = set(force_numeric or [])

    for col in df.columns:
        series = df[col]
        n_unique = series.nunique(dropna=True)
        n_total = len(series)

        # 常数特征
        if n_unique <= 1:
            feature_types[col] = 'constant'
            continue

        # ID特征（唯一值比例接近1）
        if n_unique / n_total > 0.95:
            feature_types[col] = 'id'
            continue

        # 日期时间类型
        if pd.api.types.is_datetime64_any_dtype(series):
            feature_types[col] = 'datetime'
            continue

        # 尝试转换为日期
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().iloc[:100])
                feature_types[col] = 'datetime'
                continue
            except:
                pass

        # 数值类型判断
        if pd.api.types.is_numeric_dtype(series):
            # 默认：数值类型就是 numerical
            # 只有当用户明确指定 numeric_as_categorical 时才视为 categorical
            if col in numeric_as_categorical:
                feature_types[col] = 'categorical'
            else:
                feature_types[col] = 'numerical'
        else:
            # 非数值类型（object, string, category等）
            # 默认就是 categorical
            # 只有当用户明确指定 force_numeric 时才视为 numerical
            if col in force_numeric:
                feature_types[col] = 'numerical'
            else:
                feature_types[col] = 'categorical'

    return feature_types


# ==================== 数据验证 ====================

def validate_dataframe(df: pd.DataFrame, 
                       required_cols: List[str] = None,
                       check_empty: bool = True) -> None:
    """验证DataFrame.
    
    :param df: 输入数据
    :param required_cols: 必需列名列表
    :param check_empty: 是否检查空数据
    :raises ValueError: 验证失败时抛出
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是pandas DataFrame")
    
    if check_empty and df.empty:
        raise ValueError("输入数据为空")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"缺少必需列: {missing}")


def validate_binary_target(series: pd.Series) -> None:
    """验证二元目标变量.
    
    :param series: 目标变量
    :raises ValueError: 验证失败时抛出
    """
    if not set(series.dropna().unique()).issubset({0, 1}):
        raise ValueError("目标变量必须是0/1二元变量")


# ==================== 统计工具 ====================

def safe_divide(numerator: Union[float, np.ndarray, pd.Series],
                denominator: Union[float, np.ndarray, pd.Series],
                default: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """安全除法，避免除以0.
    
    :param numerator: 分子
    :param denominator: 分母
    :param default: 除数为0时的返回值
    :return: 除法结果
    """
    if isinstance(denominator, (np.ndarray, pd.Series)):
        result = np.divide(numerator, denominator, 
                          out=np.full_like(denominator, default, dtype=float),
                          where=denominator!=0)
        return result
    else:
        return numerator / denominator if denominator != 0 else default


def calculate_gini(x: np.ndarray) -> float:
    """计算Gini系数.
    
    :param x: 输入数组
    :return: Gini系数
    """
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    
    x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(x)
    
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


# ==================== 数据预处理 ====================

def remove_outliers_iqr(df: pd.DataFrame, 
                        columns: List[str],
                        multiplier: float = 1.5) -> pd.DataFrame:
    """使用IQR方法移除异常值.
    
    :param df: 输入数据
    :param columns: 数值列列表
    :param multiplier: IQR倍数
    :return: 过滤后的数据
    """
    mask = pd.Series(True, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        
        col_mask = (df[col] >= lower) & (df[col] <= upper) | df[col].isna()
        mask = mask & col_mask
    
    return df[mask]


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """减少DataFrame内存使用.
    
    :param df: 输入数据
    :return: 优化后的数据
    """
    result = df.copy()
    
    for col in result.columns:
        col_type = result[col].dtype
        
        if col_type != object:
            c_min = result[col].min()
            c_max = result[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    result[col] = result[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    result[col] = result[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    result[col] = result[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    result[col] = result[col].astype(np.float32)
                else:
                    result[col] = result[col].astype(np.float32)
        else:
            num_unique = result[col].nunique()
            num_total = len(result[col])
            if num_unique / num_total < 0.5:
                result[col] = result[col].astype('category')
    
    return result
