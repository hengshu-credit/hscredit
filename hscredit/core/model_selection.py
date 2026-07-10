"""风控数据集切分工具.

提供按时间、时间外样本（OOT）和业务分组切分数据的统一函数，避免随机切分造成
时间穿越或同一客户、订单组同时进入训练集和测试集。
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _validate_frame(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data 必须是 DataFrame，实际类型为 {type(data).__name__}")
    if data.empty:
        raise ValueError("data 不能为空")
    return data


def _time_order(data: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    _validate_frame(data)
    if time_col not in data.columns:
        raise KeyError(f"时间列 {time_col!r} 不在数据中")
    try:
        time_values = pd.to_datetime(data[time_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"时间列 {time_col!r} 无法转换为日期时间: {exc}") from exc
    if time_values.isna().any():
        raise ValueError(f"时间列 {time_col!r} 包含缺失值")

    order = np.argsort(time_values.to_numpy(), kind="stable")
    ordered = data.iloc[order].copy()
    ordered_time = time_values.iloc[order]
    return ordered, ordered_time


def time_train_test_split(
    data: pd.DataFrame,
    time_col: str,
    test_size: Union[float, int] = 0.2,
    cutoff=None,
    gap: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分训练集和测试集.

    :param data: 输入数据
    :param time_col: 时间列
    :param test_size: 未指定 cutoff 时的测试集比例或样本数
    :param cutoff: 可选切分时间；训练集早于该时间，测试集不早于该时间
    :param gap: 训练集与测试集之间排除的样本行数，默认 0
    :return: ``(训练集, 测试集)``
    """
    if not isinstance(gap, (int, np.integer)) or gap < 0:
        raise ValueError("gap 必须是大于等于 0 的整数")

    ordered, ordered_time = _time_order(data, time_col)
    n_samples = len(ordered)

    if cutoff is not None:
        try:
            cutoff_value = pd.Timestamp(cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cutoff 无法转换为日期时间: {cutoff}") from exc
        test_mask = ordered_time >= cutoff_value
        if not test_mask.any():
            raise ValueError("cutoff 之后没有测试样本")
        first_test = int(np.flatnonzero(test_mask.to_numpy())[0])
        train_end = first_test - int(gap)
        if train_end <= 0:
            raise ValueError("cutoff 与 gap 设置后训练集为空")
        train = ordered.iloc[:train_end].copy()
        test = ordered.loc[test_mask].copy()
    else:
        if isinstance(test_size, (float, np.floating)):
            if not 0 < float(test_size) < 1:
                raise ValueError("test_size 为浮点数时必须位于 0 和 1 之间")
            n_test = int(np.ceil(n_samples * float(test_size)))
        elif isinstance(test_size, (int, np.integer)):
            n_test = int(test_size)
            if not 0 < n_test < n_samples:
                raise ValueError("test_size 为整数时必须大于 0 且小于总样本数")
        else:
            raise TypeError("test_size 必须是浮点数或整数")

        train_end = n_samples - n_test - int(gap)
        if train_end <= 0:
            raise ValueError("test_size 与 gap 设置后训练集为空")
        train = ordered.iloc[:train_end].copy()
        test = ordered.iloc[n_samples - n_test :].copy()

    if train.empty or test.empty:
        raise ValueError("训练集和测试集都必须包含样本")
    if pd.to_datetime(train[time_col]).max() > pd.to_datetime(test[time_col]).min():
        raise RuntimeError("时间切分结果发生穿越，请检查时间列")
    return train, test


def oot_split(
    data: pd.DataFrame,
    time_col: str,
    oot_start,
    oot_end=None,
    gap: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按明确时间边界切分开发集和时间外样本集.

    :param data: 输入数据
    :param time_col: 时间列
    :param oot_start: OOT 起始时间，包含该边界
    :param oot_end: OOT 结束时间，包含该边界；默认使用全部后续样本
    :param gap: OOT 前排除的样本行数
    :return: ``(开发集, OOT集)``
    """
    development, oot = time_train_test_split(
        data=data,
        time_col=time_col,
        cutoff=oot_start,
        gap=gap,
    )
    if oot_end is not None:
        try:
            end_value = pd.Timestamp(oot_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"oot_end 无法转换为日期时间: {oot_end}") from exc
        if end_value < pd.Timestamp(oot_start):
            raise ValueError("oot_end 不能早于 oot_start")
        oot = oot.loc[pd.to_datetime(oot[time_col]) <= end_value].copy()
        if oot.empty:
            raise ValueError("指定 OOT 时间范围内没有样本")
    return development, oot


def group_train_test_split(
    X,
    y=None,
    *,
    groups=None,
    group_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
):
    """按组随机切分，保证同一组不会同时出现在训练集和测试集.

    :param X: 特征数据
    :param y: 可选目标变量
    :param groups: 与 X 等长的分组数组
    :param group_col: X 为 DataFrame 时可直接指定分组列
    :param test_size: 测试组比例
    :param random_state: 随机种子
    :return: 未传 y 时返回 ``X_train, X_test``；传 y 时返回
        ``X_train, X_test, y_train, y_test``
    """
    if groups is not None and group_col is not None:
        raise ValueError("groups 和 group_col 只能指定一个")
    if group_col is not None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("使用 group_col 时 X 必须是 DataFrame")
        if group_col not in X.columns:
            raise KeyError(f"分组列 {group_col!r} 不在 X 中")
        groups = X[group_col]
    if groups is None:
        raise ValueError("必须提供 groups 或 group_col")
    if len(X) != len(groups):
        raise ValueError("X 与 groups 的长度必须一致")
    if y is not None and len(X) != len(y):
        raise ValueError("X 与 y 的长度必须一致")
    if pd.isna(np.asarray(groups, dtype=object)).any():
        raise ValueError("groups 不能包含缺失值")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(np.zeros(len(X)), groups=groups))

    def _take(values, indices):
        if hasattr(values, "iloc"):
            return values.iloc[indices].copy()
        return np.asarray(values)[indices]

    X_train = _take(X, train_idx)
    X_test = _take(X, test_idx)
    if y is None:
        return X_train, X_test
    return X_train, X_test, _take(y, train_idx), _take(y, test_idx)


__all__ = [
    "time_train_test_split",
    "oot_split",
    "group_train_test_split",
]
