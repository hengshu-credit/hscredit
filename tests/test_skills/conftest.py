"""Agent Skills 测试数据。"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def credit_frame():
    """返回包含分箱、报告和绘图所需字段的确定性信贷样本。"""
    rng = np.random.default_rng(42)
    size = 120
    score = rng.normal(600, 65, size).round(2)
    age = rng.integers(21, 61, size)
    target = ((score < 590).astype(int) ^ (rng.random(size) < 0.12)).astype(int)
    frame = pd.DataFrame(
        {
            "score": score,
            "age": age,
            "amount": rng.integers(1_000, 50_000, size).astype(float),
            "apply_date": pd.date_range("2024-01-01", periods=size, freq="3D"),
            "segment": np.where(age < 35, "青年", "成熟"),
            "MOB1": np.where(target == 1, rng.integers(1, 31, size), 0),
            "target": target,
        }
    )
    assert frame["target"].nunique() == 2
    return frame
