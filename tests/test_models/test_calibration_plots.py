"""概率校准图的覆盖参数、完整曲线和中文契约。"""

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hscredit.core.models.calibration import ProbabilityCalibrator, plot_calibration_comparison


def test_calibration_comparison_draws_every_model_and_honors_overrides():
    """模型数超过默认配色时不能静默漏画。"""
    y = np.array([0, 1] * 10)
    probabilities = {f"模型{i}": np.linspace(0.05 + i * 0.005, 0.85 + i * 0.005, len(y)) for i in range(7)}

    figure = plot_calibration_comparison(y, probabilities, figsize=(7, 3), title="校准比较", show=False)

    assert isinstance(figure, matplotlib.figure.Figure)
    assert tuple(figure.get_size_inches()) == pytest.approx((7, 3))
    assert len(figure.axes[0].lines) - 1 == 7
    assert figure._suptitle.get_text() == "校准比较"


def test_calibration_comparison_rejects_empty_mapping():
    """空模型集合应得到中文参数错误而不是除零。"""
    with pytest.raises(ValueError, match="不能为空"):
        plot_calibration_comparison(np.array([0, 1]), {}, show=False)


def test_probability_calibrator_plot_reuses_custom_target_by_default():
    """拟合时的自定义目标列应自动贯穿绘图入口。"""
    values, labels = make_classification(n_samples=80, n_features=4, random_state=18)
    frame = pd.DataFrame(values, columns=list("甲乙丙丁"))
    model = LogisticRegression(max_iter=300).fit(frame, labels)
    calibration_frame = frame.copy()
    calibration_frame["坏样本"] = labels
    calibrator = ProbabilityCalibrator(model=model, target="坏样本", calib_ratio=None).fit(calibration_frame)

    figure = calibrator.plot_reliability_diagram(calibration_frame, show=False)

    assert isinstance(figure, matplotlib.figure.Figure)
    assert {axis.get_title() for axis in figure.axes if axis.get_title()} >= {"可靠性曲线", "概率分布"}
