"""数据集加载工具.

用于加载 hscredit.xlsx 数据集并创建目标变量.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_hscredit_data(bad_threshold: int = 15, target_col: str = 'target'):
    """加载 hscredit 数据集.

    使用 MOB1 或 MOB2（逾期天数）创建目标变量.

    :param bad_threshold: 坏样本阈值，逾期天数超过此值为坏样本(1)，默认15
    :param target_col: 目标变量列名，默认 'target'
    :return: (X, y, feature_cols, numeric_cols, categorical_cols)

    示例:
        >>> X, y, features, num_cols, cat_cols = load_hscredit_data(bad_threshold=15)
        >>> print(f"样本量: {len(X)}, 坏样本率: {y.mean():.2%}")
    """
    # 数据路径
    data_path = Path(__file__).parent / "hscredit.xlsx"

    # 读取数据
    df = pd.read_excel(data_path)

    # 创建目标变量
    df[target_col] = ((df['MOB1'] > bad_threshold) | (df['MOB2'] > bad_threshold)).astype(int)

    # 选择特征列（排除 MOB1, MOB2 和目标变量）
    feature_cols = [col for col in df.columns if col not in ['MOB1', 'MOB2', target_col]]

    # 分离数值型和类别型特征
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()

    # 分离 X 和 y
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y, feature_cols, numeric_cols, categorical_cols


def get_data_info():
    """获取数据集基本信息.

    :return: 数据集信息字典
    """
    X, y, features, num_cols, cat_cols = load_hscredit_data()

    info = {
        '样本量': len(X),
        '特征数': len(features),
        '数值型特征': num_cols,
        '类别型特征': cat_cols,
        '坏样本数': y.sum(),
        '好样本数': len(y) - y.sum(),
        '坏样本率': f"{y.mean():.2%}",
    }

    return info


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载...")
    X, y, features, num_cols, cat_cols = load_hscredit_data()

    print(f"\n数据集概况:")
    print(f"  样本量: {len(X)}")
    print(f"  特征数: {len(features)}")
    print(f"  数值型特征: {len(num_cols)}个 - {num_cols}")
    print(f"  类别型特征: {len(cat_cols)}个 - {cat_cols}")
    print(f"  坏样本率: {y.mean():.2%}")

    print(f"\n特征预览:")
    print(X.head())
