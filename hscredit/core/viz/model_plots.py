# -*- coding: utf-8 -*-
"""
模型可视化函数.

提供模型相关的可视化功能，包括逻辑回归系数误差图等。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 默认配色方案
DEFAULT_COLORS = ["#2639E9", "#F76E6C", "#FE7715"]


def plot_weights(summary, save=None, figsize=(15, 8), fontsize=14, colors=None, ax=None):
    """
    逻辑回归模型系数误差图.
    
    展示逻辑回归模型各特征的系数估计值及其95%置信区间，
    用于评估特征的显著性及系数的稳定性。

    **参数**

    :param summary: 逻辑回归模型的统计摘要，可以是以下两种形式之一：
        - pd.DataFrame: LogisticRegression.summary() 的返回结果
        - LogisticRegression: hscredit 的 LogisticRegression 模型对象
    :param save: 图片保存路径，如果传入路径中有文件夹不存在，会自动创建，默认 None
    :param figsize: 图片大小（创建新图时使用），默认 (15, 8)
    :param fontsize: 字体大小，默认 14
    :param colors: 图片主题颜色列表，长度为3，默认为 ["#2639E9", "#F76E6C", "#FE7715"]
    :param ax: 可选的 matplotlib Axes 对象，用于在已有画布上绘图

    **返回**

    :return: matplotlib Figure 或 Axes 对象

    **参考样例**

    使用 DataFrame 作为输入::

        >>> from hscredit.core.models import LogisticRegression
        >>> from hscredit.core.viz import plot_weights
        >>> 
        >>> # 训练模型
        >>> model = LogisticRegression(calculate_stats=True)
        >>> model.fit(X_train, y_train)
        >>> 
        >>> # 方式1：传入 summary DataFrame
        >>> summary = model.summary()
        >>> fig = plot_weights(summary)
        >>> 
        >>> # 方式2：直接传入模型对象
        >>> fig = plot_weights(model)

    在已有画布上绘图::

        >>> fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        >>> plot_weights(model1, ax=axes[0])
        >>> plot_weights(model2, ax=axes[1])

    保存图片::

        >>> fig = plot_weights(model, save='./output/weight_plot.png')

    自定义样式::

        >>> fig = plot_weights(
        ...     model,
        ...     figsize=(12, 6),
        ...     fontsize=12,
        ...     colors=['#1f77b4', '#ff7f0e', '#2ca02c']
        ... )

    **说明**

    图表展示内容：
        - 横轴：系数估计值 (Weight Estimates)
        - 纵轴：特征变量名称 (Variable)
        - 误差线：95% 置信区间
        - 垂直虚线：x=0 参考线

    解释指南：
        - 误差线不跨越0：特征显著 (p<0.05)
        - 误差线跨越0：特征不显著 (p≥0.05)
        - 系数为正：特征与目标正相关
        - 系数为负：特征与目标负相关
    """
    # 处理输入参数
    if colors is None:
        colors = DEFAULT_COLORS
    
    # 支持两种输入：DataFrame 或 LogisticRegression 对象
    if hasattr(summary, 'summary'):
        # 如果是 LogisticRegression 对象
        summary_df = summary.summary()
    else:
        # 如果已经是 DataFrame
        summary_df = summary.copy()
    
    # 检查必要的列是否存在
    required_cols = ['Coef.']
    for col in required_cols:
        if col not in summary_df.columns:
            raise ValueError(f"summary DataFrame 必须包含 '{col}' 列")
    
    # 检查置信区间列（兼容不同的列名格式）
    # hscredit: "[0.025", "0.975]"
    # scorecardpipeline: "[ 0.025", "0.975 ]"
    ci_lower_col = None
    ci_upper_col = None
    
    for col in summary_df.columns:
        if '0.025' in col:
            ci_lower_col = col
        if '0.975' in col:
            ci_upper_col = col
    
    if ci_lower_col is None or ci_upper_col is None:
        # 如果没有置信区间列，尝试使用 Std.Err 计算
        if 'Std.Err' in summary_df.columns:
            summary_df['ci_lower'] = summary_df['Coef.'] - 1.96 * summary_df['Std.Err']
            summary_df['ci_upper'] = summary_df['Coef.'] + 1.96 * summary_df['Std.Err']
            ci_lower_col = 'ci_lower'
            ci_upper_col = 'ci_upper'
        else:
            raise ValueError(
                "summary DataFrame 必须包含置信区间列（如 '[0.025' 和 '0.975]'）"
                "或标准误差列 'Std.Err'"
            )
    
    # 准备数据
    x = summary_df['Coef.']
    y = summary_df.index
    
    # 计算误差线
    lower_error = summary_df['Coef.'] - summary_df[ci_lower_col]
    upper_error = summary_df[ci_upper_col] - summary_df['Coef.']
    
    # 获取或创建 Axes
    if ax is not None:
        return_ax = True
        fig = ax.figure
    else:
        return_ax = False
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 绘制误差图
    ax.errorbar(
        x, y, 
        xerr=[lower_error, upper_error], 
        fmt="o", 
        ecolor=colors[0], 
        elinewidth=2, 
        capthick=2, 
        capsize=4, 
        ms=6, 
        mfc=colors[0], 
        mec=colors[0]
    )
    
    # 添加垂直参考线
    ax.axvline(0, color=colors[0], linestyle='--', alpha=0.5)
    
    # 设置边框样式
    ax.spines['top'].set_color(colors[0])
    ax.spines['bottom'].set_color(colors[0])
    ax.spines['right'].set_color(colors[0])
    ax.spines['left'].set_color(colors[0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置标题和标签
    ax.set_title("逻辑回归系数分析 - 权重图\n", fontsize=fontsize, fontweight="bold")
    ax.set_xlabel("系数估计值", fontsize=fontsize, weight="bold")
    ax.set_ylabel("特征变量", fontsize=fontsize, weight="bold")
    
    # 设置网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    if not return_ax:
        # 自动调整布局
        plt.tight_layout()
        
        # 保存图片
        if save:
            save_dir = os.path.dirname(save)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            fig.savefig(save, dpi=240, format="png", bbox_inches="tight")
        
        return fig
    else:
        return ax
