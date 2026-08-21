# -*- coding: utf-8 -*-
"""
模型可视化函数.

提供模型相关的可视化功能，包括统一特征重要性看板、逻辑回归系数误差图等。
"""

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from .utils import DEFAULT_COLORS, setup_axis_style, save_figure, _create_subplots, _tight_layout
from .style import PRIMARY_COLORS


PredictionMethod = Union[str, Callable[[Union[np.ndarray, pd.DataFrame]], Any]]


def _feature_names(model: Any, X: Union[np.ndarray, pd.DataFrame]) -> list:
    """按 DataFrame、模型属性、人工名称的顺序解析特征名。"""
    if isinstance(X, pd.DataFrame):
        return X.columns.astype(str).tolist()

    n_features = np.asarray(X).shape[1]
    for attr in ("feature_names_in_", "feature_names_", "feature_names"):
        names = getattr(model, attr, None)
        if names is not None and len(names) == n_features:
            return [str(name) for name in names]
    return [f"feature_{i}" for i in range(n_features)]


def _as_feature_frame(
    values: Union[np.ndarray, pd.DataFrame],
    feature_names: list,
    preserve_dataframe: bool,
) -> Union[np.ndarray, pd.DataFrame]:
    """将 SHAP 传入的数组恢复为模型拟合时常用的 DataFrame 契约。"""
    if preserve_dataframe:
        if isinstance(values, pd.DataFrame):
            return values.loc[:, feature_names]
        return pd.DataFrame(np.asarray(values), columns=feature_names)
    return np.asarray(values)


def _resolve_prediction_function(
    model: Any,
    prediction_method: PredictionMethod,
    class_index: int,
    feature_names: list,
    preserve_dataframe: bool,
) -> Tuple[Callable[[Union[np.ndarray, pd.DataFrame]], np.ndarray], str]:
    """把模型预测方法或用户 callable 统一为一维连续输出函数。"""
    valid_methods = ("predict", "predict_score", "predict_proba")
    if isinstance(prediction_method, str):
        if prediction_method not in valid_methods:
            raise ValueError(f"不支持的预测方法 '{prediction_method}'，可选: {valid_methods} 或 callable")
        predictor = getattr(model, prediction_method, None)
        if not callable(predictor):
            raise ValueError(f"模型未提供可调用的预测方法 '{prediction_method}'")
        method_name = prediction_method
    elif callable(prediction_method):
        predictor = prediction_method
        method_name = getattr(prediction_method, "__name__", "callable")
    else:
        raise TypeError("prediction_method 必须是预测方法名称或 callable")

    def predict_1d(values):
        model_input = _as_feature_frame(values, feature_names, preserve_dataframe)
        result = np.asarray(predictor(model_input))
        if result.ndim == 2:
            if not 0 <= class_index < result.shape[1]:
                raise ValueError(
                    f"class_index={class_index} 超出预测结果列范围，预测结果共有 {result.shape[1]} 列"
                )
            result = result[:, class_index]
        if result.ndim != 1:
            raise ValueError(f"预测方法必须返回一维结果或二维类别结果，当前形状为 {result.shape}")
        return result.astype(float, copy=False)

    return predict_1d, method_name


def _native_feature_importance(model: Any, feature_names: list) -> pd.Series:
    """在没有 y 时读取模型原生重要性并统一为降序 Series。"""
    getter = getattr(model, "get_feature_importances", None)
    if callable(getter):
        importance = getter()
        if isinstance(importance, pd.Series):
            return importance.astype(float).sort_values(ascending=False)
        values = np.asarray(importance, dtype=float)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)
        values = np.abs(coefficients) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
    else:
        raise ValueError("未提供 y，且模型没有可用的特征重要性或系数")

    values = np.asarray(values, dtype=float).ravel()
    if len(values) != len(feature_names):
        raise ValueError(f"特征重要性数量 {len(values)} 与特征数量 {len(feature_names)} 不一致")
    return pd.Series(values, index=feature_names, name="importance").sort_values(ascending=False)


def _resolve_top_n(top_n: Optional[int], n_features: int, parameter_name: str) -> int:
    """校验并限制 Top N。"""
    if top_n is None:
        return n_features
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n <= 0:
        raise ValueError(f"{parameter_name} 必须是正整数或 None")
    return min(int(top_n), n_features)


def _shap_classic_colormap():
    """返回 SHAP 经典蓝紫红色阶，兼容不同 SHAP 版本。"""
    try:
        import shap

        return shap.plots.colors.red_blue
    except (AttributeError, ImportError):
        return LinearSegmentedColormap.from_list("shap_blue_purple_red", ["#008BFB", "#7B2CBF", "#FF0051"])


def _normalized_feature_values(values: np.ndarray) -> np.ndarray:
    """把数值或类别特征转换为用于 SHAP 颜色映射的 0-1 分位数。"""
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() != series.notna().sum():
        codes, _ = pd.factorize(series, sort=True)
        numeric = pd.Series(codes, dtype=float).where(series.notna())

    ranks = numeric.rank(method="average", pct=True)
    if ranks.notna().any():
        ranks = ranks.fillna(0.5)
        minimum = float(ranks.min())
        maximum = float(ranks.max())
        if maximum > minimum:
            return ((ranks - minimum) / (maximum - minimum)).to_numpy(dtype=float)
    return np.full(len(series), 0.5, dtype=float)


def _figure_size(n_left: int, n_right: int, show_dependence: bool) -> Tuple[float, float]:
    """根据左右特征数计算画布尺寸，并保持两模块宽度 1:1。"""
    has_right = show_dependence and n_right > 0
    module_width = 10.5
    width = module_width * (2 if has_right else 1)
    left_height = max(6.0, 0.42 * n_left + 2.8)
    right_rows = int(np.ceil(n_right / 2)) if has_right else 0
    right_height = max(6.0, 3.6 * right_rows) if has_right else 0.0
    return width, max(left_height, right_height)


def _style_colorbar(colorbar) -> None:
    """为颜色条应用 hscredit 坐标轴颜色。"""
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=PRIMARY_COLORS[0])
    colorbar.ax.xaxis.label.set_color(PRIMARY_COLORS[0])
    colorbar.ax.yaxis.label.set_color(PRIMARY_COLORS[0])


def _plot_native_importance(
    importance: pd.Series,
    left_top_n: Optional[int],
    figsize: Optional[Tuple[float, float]],
    title: Optional[str],
    save: Optional[str],
    show: bool,
):
    """绘制缺少 y 时的原生特征重要性降级图。"""
    n_left = _resolve_top_n(left_top_n, len(importance), "left_top_n")
    displayed = importance.head(n_left)
    figure_size = figsize or _figure_size(n_left, 0, False)
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    ax.set_label("模型特征重要性")

    positions = np.arange(n_left)
    bars = ax.barh(
        positions,
        displayed.to_numpy()[::-1],
        color=PRIMARY_COLORS[0],
        alpha=0.65,
        edgecolor=PRIMARY_COLORS[0],
        linewidth=0.8,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(displayed.index[::-1])
    ax.set_xlabel("特征重要性")
    ax.set_ylabel("特征")
    ax.set_title(title or "模型特征重要性")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    setup_axis_style(ax, PRIMARY_COLORS, hide_top_right=True)

    maximum = float(np.nanmax(displayed.to_numpy())) if len(displayed) else 0.0
    padding = maximum * 0.015 if maximum > 0 else 0.01
    for bar, value in zip(bars, displayed.to_numpy()[::-1]):
        ax.text(value + padding, bar.get_y() + bar.get_height() / 2, f"{value:.4g}", va="center", fontsize=9)

    save_figure(fig, save)
    if show:
        plt.show()
    return fig


def plot_model_feature_importance(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series]] = None,
    prediction_method: PredictionMethod = "predict_proba",
    class_index: int = 1,
    left_top_n: Optional[int] = None,
    right_top_n: int = 6,
    show_dependence: bool = True,
    background_size: Optional[int] = 100,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    save: Optional[str] = None,
    random_state: Optional[int] = 42,
    show: bool = True,
):
    """绘制模型的 SHAP 特征重要性综合看板。

    有 ``y`` 时，左侧叠加平均绝对 SHAP 柱状图与 SHAP 蜂群分布，右侧展示
    Top N 特征的依赖关系；没有 ``y`` 时按约定仅绘制模型原生特征重要性。
    ``prediction_method`` 决定 SHAP 解释的模型输出，可传 ``'predict'``、
    ``'predict_score'``、``'predict_proba'`` 或接收 ``X`` 的 callable。

    **参数**

    :param model: 已拟合模型
    :param X: 用于解释的特征矩阵
    :param y: 真实标签；为 None 时仅绘制原生特征重要性
    :param prediction_method: SHAP 使用的预测方法或 callable，默认 ``'predict_proba'``
    :param class_index: 二维预测结果中要解释的列索引，默认 1（正类）
    :param left_top_n: 左侧显示特征数，默认 None，表示全部特征
    :param right_top_n: 右侧依赖图显示特征数，默认 6
    :param show_dependence: 是否显示右侧依赖图，默认 True
    :param background_size: SHAP 背景样本数，None 表示使用全部 X，默认 100
    :param figsize: 画布大小；默认根据左右 Top N 自动计算
    :param title: 总标题，默认包含模型名和预测方法
    :param save: 图片保存路径，默认 None
    :param random_state: 背景抽样和蜂群抖动随机种子，默认 42
    :param show: 是否调用 ``plt.show()``，默认 True
    :return: matplotlib Figure

    **参考样例**

    >>> from hscredit.core.viz import plot_model_feature_importance
    >>> fig = plot_model_feature_importance(
    ...     model, X_test, y_test,
    ...     prediction_method='predict_proba',
    ...     left_top_n=None,
    ...     right_top_n=6,
    ... )
    >>> fig.savefig('模型特征重要性.png', dpi=300, bbox_inches='tight')
    """
    X_values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    if X_values.ndim != 2:
        raise ValueError(f"X 必须是二维特征矩阵，当前维度为 {X_values.ndim}")
    if X_values.shape[0] == 0 or X_values.shape[1] == 0:
        raise ValueError("X 不能为空")

    names = _feature_names(model, X)
    if y is None:
        return _plot_native_importance(
            _native_feature_importance(model, names), left_top_n, figsize, title, save, show
        )

    y_values = np.asarray(y).ravel()
    if len(y_values) != X_values.shape[0]:
        raise ValueError(f"X 与 y 样本数不一致: {X_values.shape[0]} != {len(y_values)}")
    if len(y_values) == 0:
        raise ValueError("y 不能为空")

    predict_function, method_name = _resolve_prediction_function(
        model,
        prediction_method,
        class_index,
        names,
        isinstance(X, pd.DataFrame),
    )

    if background_size is not None:
        if isinstance(background_size, bool) or not isinstance(background_size, (int, np.integer)):
            raise TypeError("background_size 必须是正整数或 None")
        if background_size <= 0:
            raise ValueError("background_size 必须是正整数或 None")

    try:
        import shap
    except ImportError as exc:
        raise ImportError("绘制 SHAP 特征重要性图需要安装 shap") from exc

    n_samples = X_values.shape[0]
    rng = np.random.default_rng(random_state)
    if background_size is None or background_size >= n_samples:
        background_indices = np.arange(n_samples)
    else:
        background_indices = np.sort(rng.choice(n_samples, size=int(background_size), replace=False))
    background = X.iloc[background_indices] if isinstance(X, pd.DataFrame) else X_values[background_indices]

    explainer = shap.Explainer(predict_function, background)
    explanation = explainer(X)
    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.ndim == 3:
        if not 0 <= class_index < shap_values.shape[2]:
            raise ValueError(f"class_index={class_index} 超出 SHAP 输出列范围 {shap_values.shape[2]}")
        shap_values = shap_values[:, :, class_index]
    if shap_values.ndim != 2 or shap_values.shape != X_values.shape:
        raise ValueError(f"SHAP 输出形状 {shap_values.shape} 与 X 形状 {X_values.shape} 不一致")

    shap_importance = np.mean(np.abs(shap_values), axis=0)
    ranking = np.argsort(-shap_importance, kind="stable")
    n_left = _resolve_top_n(left_top_n, len(names), "left_top_n")
    n_right = _resolve_top_n(right_top_n, len(names), "right_top_n") if show_dependence else 0
    has_right = show_dependence and n_right > 0
    figure_size = figsize or _figure_size(n_left, n_right, has_right)

    fig = plt.figure(figsize=figure_size, constrained_layout=True)
    if has_right:
        outer_grid = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.12)
        main_ax = fig.add_subplot(outer_grid[0, 0])
        right_columns = 2 if n_right > 1 else 1
        right_rows = int(np.ceil(n_right / right_columns))
        right_grid = outer_grid[0, 1].subgridspec(right_rows, right_columns, wspace=0.06, hspace=0.34)
        dependence_axes = [fig.add_subplot(right_grid[i // right_columns, i % right_columns]) for i in range(n_right)]
    else:
        main_ax = fig.add_subplot(111)
        dependence_axes = []

    main_ax.set_label("SHAP分布")
    display_indices = ranking[:n_left]
    display_positions = np.arange(n_left)
    reversed_indices = display_indices[::-1]
    cmap = _shap_classic_colormap()

    importance_ax = main_ax.twiny()
    importance_ax.set_label("SHAP重要性")
    importance_ax.barh(
        display_positions,
        shap_importance[reversed_indices],
        color=PRIMARY_COLORS[0],
        alpha=0.22,
        height=0.72,
        edgecolor=PRIMARY_COLORS[0],
        linewidth=0.6,
    )
    importance_ax.set_xlabel("平均 |SHAP 值|")
    importance_ax.grid(False)
    setup_axis_style(importance_ax, PRIMARY_COLORS)
    importance_ax.spines["bottom"].set_visible(False)
    importance_ax.spines["right"].set_visible(False)
    importance_ax.set_zorder(0)
    main_ax.set_zorder(1)
    main_ax.patch.set_alpha(0.0)

    shap_scatter = None
    for position, feature_index in enumerate(reversed_indices):
        jitter = rng.normal(0.0, 0.075, n_samples)
        shap_scatter = main_ax.scatter(
            shap_values[:, feature_index],
            position + jitter,
            c=_normalized_feature_values(X_values[:, feature_index]),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=22,
            alpha=0.78,
            edgecolors="none",
            rasterized=n_samples > 2000,
        )

    main_ax.set_yticks(display_positions)
    main_ax.set_yticklabels([names[index] for index in reversed_indices])
    main_ax.set_xlabel("SHAP 值（对模型输出的影响）")
    main_ax.set_ylabel("特征")
    main_ax.set_title("全局特征重要性与 SHAP 分布")
    main_ax.axvline(0.0, color="#8A8FA3", linestyle="--", linewidth=1.0, alpha=0.8)
    main_ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    setup_axis_style(main_ax, PRIMARY_COLORS, hide_top_right=True)

    if shap_scatter is not None:
        shap_colorbar = fig.colorbar(
            shap_scatter,
            ax=main_ax,
            orientation="horizontal",
            pad=0.075,
            fraction=0.045,
            aspect=38,
        )
        shap_colorbar.set_ticks([0.0, 1.0])
        shap_colorbar.set_ticklabels(["低", "高"])
        shap_colorbar.set_label("特征值")
        _style_colorbar(shap_colorbar)

    if dependence_axes:
        y_numeric = pd.to_numeric(pd.Series(y_values), errors="coerce")
        y_labels = None
        if y_numeric.notna().sum() != len(y_values):
            y_codes, y_uniques = pd.factorize(y_values, sort=True)
            y_numeric = pd.Series(y_codes, dtype=float)
            y_labels = [str(value) for value in y_uniques]
        y_color = y_numeric.to_numpy(dtype=float)
        y_min = float(np.nanmin(y_color))
        y_max = float(np.nanmax(y_color))
        y_norm = Normalize(vmin=y_min, vmax=y_max if y_max > y_min else y_min + 1.0)

        dependence_scatter = None
        for rank_position, (ax, feature_index) in enumerate(zip(dependence_axes, ranking[:n_right]), start=1):
            ax.set_label("SHAP依赖")
            raw_feature = X_values[:, feature_index]
            numeric_feature = pd.to_numeric(pd.Series(raw_feature), errors="coerce").to_numpy(dtype=float)
            if np.isfinite(numeric_feature).sum() == len(raw_feature):
                x_plot = numeric_feature
            else:
                x_plot = pd.Series(raw_feature).astype(str).to_numpy()

            dependence_scatter = ax.scatter(
                x_plot,
                shap_values[:, feature_index],
                c=y_color,
                cmap=cmap,
                norm=y_norm,
                s=34,
                alpha=0.72,
                edgecolors=PRIMARY_COLORS[0],
                linewidths=0.35,
            )

            finite = np.isfinite(numeric_feature) & np.isfinite(shap_values[:, feature_index])
            if finite.sum() > 1 and np.unique(numeric_feature[finite]).size > 1:
                coefficients = np.polyfit(numeric_feature[finite], shap_values[finite, feature_index], 1)
                line_x = np.linspace(numeric_feature[finite].min(), numeric_feature[finite].max(), 100)
                ax.plot(line_x, np.polyval(coefficients, line_x), color="#333333", linestyle="--", linewidth=1.2)
                correlation = float(np.corrcoef(numeric_feature[finite], shap_values[finite, feature_index])[0, 1])
                if np.isfinite(correlation):
                    ax.text(
                        0.04,
                        0.96,
                        f"r = {correlation:.2f}",
                        transform=ax.transAxes,
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.82, "edgecolor": PRIMARY_COLORS[0]},
                    )

            ax.set_title(f"Top {rank_position}: {names[feature_index]}", fontsize=11)
            ax.set_xlabel(names[feature_index], fontsize=10)
            ax.set_ylabel("SHAP 值", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.3)
            setup_axis_style(ax, PRIMARY_COLORS, hide_top_right=True)

        label_colorbar = fig.colorbar(dependence_scatter, ax=dependence_axes, pad=0.025, fraction=0.04, aspect=28)
        unique_y = np.unique(y_color[np.isfinite(y_color)])
        if len(unique_y) <= 6:
            label_colorbar.set_ticks(unique_y)
            if y_labels is not None and len(y_labels) == len(unique_y):
                label_colorbar.set_ticklabels(y_labels)
        label_colorbar.set_label("真实标签")
        _style_colorbar(label_colorbar)

    model_name = model.__class__.__name__
    fig.suptitle(title or f"SHAP 综合特征重要性：{model_name}（{method_name}）", fontsize=16, fontweight="bold")
    save_figure(fig, save)
    if show:
        plt.show()
    return fig


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
        ...     colors=['#2639E9', '#F76E6C', '#FE7715']
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
    # 注意：必须先判断 DataFrame —— import hscredit 会为 DataFrame 注册 .summary() 扩展方法，
    # 若先用 hasattr(summary, 'summary') 判断，传入的系数 DataFrame 会被误当作模型对象，
    # 错误调用 df.summary()（describe 风格）而丢失 'Coef.' 列。
    if isinstance(summary, pd.DataFrame):
        summary_df = summary.copy()
    elif hasattr(summary, 'summary') and callable(getattr(summary, 'summary')):
        # LogisticRegression 等模型对象
        summary_df = summary.summary()
    else:
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
        fig, ax = _create_subplots(1, 1, figsize=figsize)
    
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

    # 统一坐标轴样式：主题色边框（隐藏上/右）+ 主题色刻度，与 bin_plot 参考样式一致
    setup_axis_style(ax, colors, hide_top_right=True)
    ax.tick_params(axis='both', colors=colors[0])

    # 设置标题和标签
    ax.set_title("逻辑回归系数分析 - 权重图\n", fontsize=fontsize, fontweight="bold")
    ax.set_xlabel("系数估计值", fontsize=fontsize, weight="bold", color=colors[0])
    ax.set_ylabel("特征变量", fontsize=fontsize, weight="bold", color=colors[0])

    # 设置网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

    if not return_ax:
        # 自动调整布局
        _tight_layout(fig)
        save_figure(fig, save)
        return fig
    else:
        return ax
