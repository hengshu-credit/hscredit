"""测试演示样例使用的分析图表。"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from hscredit.core.viz import (
    metric_comparison_plot,
    rule_swap_plot,
    strategy_simulation_plot,
    tree_leaf_comparison_plot,
)


def test_metric_comparison_plot_supports_semantic_colors_and_reference_lines():
    data = pd.DataFrame({"特征": ["A", "B", "C"], "IV": [0.01, 0.08, 0.35]})

    fig = metric_comparison_plot(data, "特征", "IV", color_scheme="iv")

    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) == 3
    assert len(fig.axes[0].lines) == 3


def test_strategy_simulation_plot_uses_dual_axes():
    simulation = pd.DataFrame(
        {
            "评分阈值": [550, 600, 650],
            "通过率(%)": [90.0, 75.0, 50.0],
            "通过人群坏率(%)": [20.0, 15.0, 10.0],
        }
    )

    fig = strategy_simulation_plot(simulation)

    assert len(fig.axes) == 2
    assert len(fig.axes[0].patches) == 3
    assert len(fig.axes[1].lines) == 1


def test_rule_swap_plot_contains_three_analysis_panels():
    pipeline = pd.DataFrame(
        {
            "规则分类": ["OUT-OUT拒绝", "IN-OUT置出", "OUT-IN置入"],
            "指标名称": ["规则A", "规则B", "规则C"],
            "样本总数": [100, 80, 60],
            "坏样本率": [0.4, 0.3, 0.2],
            "LIFT值": [1.5, 1.1, 0.8],
            "通过率(绝对值)": [80.0, 75.0, 78.0],
        }
    )

    fig = rule_swap_plot(pipeline)

    assert len(fig.axes) == 3
    assert len(fig.axes[0].patches) == 3
    assert len(fig.axes[1].patches) == 3
    assert len(fig.axes[2].lines) == 1


def test_tree_leaf_comparison_plot_creates_one_panel_per_tree():
    leaf_table = pd.DataFrame(
        {
            "节点编号": [1, 2],
            "坏样本率": [0.4, 0.2],
            "LIFT值": [1.5, 0.8],
        }
    )

    fig = tree_leaf_comparison_plot(
        {"自动树": leaf_table, "人工树": leaf_table.copy()},
        overall_bad_rate=0.25,
    )

    assert len(fig.axes) == 2
    assert all(len(ax.patches) == 2 for ax in fig.axes)
