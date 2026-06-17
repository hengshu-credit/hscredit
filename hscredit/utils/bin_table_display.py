"""分箱表美化展示工具.

提供在 Jupyter Notebook 中美观展示特征分箱表的功能。
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Dict, Any


def style_bin_table(
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    highlight_iv: bool = True,
    highlight_bad_rate: bool = True,
    highlight_lift: bool = True,
    highlight_ks: bool = True,
    compact: bool = False,
    precision: Optional[Dict[str, int]] = None,
    index_as_bin: bool = False,
    percent_format: bool = True,
    high_tech_style: bool = False,
) -> pd.DataFrame.style:
    """美化分箱表展示.

    使用 pandas Styler 对分箱表进行格式化和高亮，使其在 Jupyter 中更易读。

    :param df: 分箱统计表 DataFrame
    :param max_rows: 最大显示行数，None 表示显示全部
    :param highlight_iv: 是否高亮 IV 值
    :param highlight_bad_rate: 是否高亮坏样本率（进度条）
    :param highlight_lift: 是否高亮 LIFT 值（进度条）
    :param highlight_ks: 是否高亮 KS 值（进度条）
    :param compact: 是否使用紧凑模式（隐藏部分列）
    :param precision: 自定义小数位数，格式为 {'列名': 位数}
    :param index_as_bin: 是否将分箱作为索引显示
    :param percent_format: 是否将百分比相关列显示为百分比格式（默认True）
    :param high_tech_style: 是否使用高科技/AI风格样式
    :return: 格式化后的 Styler 对象

    **参考样例**

    >>> table = feature_bin_stats(data, 'score', target='target')
    >>> style_bin_table(table).show()
    """
    # 检查是否为多级表头
    is_multi_level = isinstance(df.columns, pd.MultiIndex)

    # 创建副本避免修改原数据
    df_display = df.copy()

    # index_as_bin: 将分箱列设置为索引，并合并指标名称/含义列
    merge_first_cols = False
    if index_as_bin:
        # 查找分箱标签列
        bin_label_col = None
        bin_col = None
        indicator_name_col = None
        indicator_desc_col = None
        
        for col in df_display.columns:
            col_name = col[1] if isinstance(col, tuple) else col
            if col_name == '分箱标签':
                bin_label_col = col
            if col_name == '分箱':
                bin_col = col
            if col_name == '指标名称':
                indicator_name_col = col
            if col_name == '指标含义':
                indicator_desc_col = col
        
        # 先合并指标名称和指标含义
        if indicator_name_col is not None and indicator_desc_col is not None:
            # 合并为新列
            if is_multi_level:
                df_display['指标名称_合并'] = df_display[indicator_name_col].astype(str) + ' - ' + df_display[indicator_desc_col].astype(str)
                df_display = df_display.drop(columns=[indicator_name_col, indicator_desc_col])
                # 重建多级索引列名
                new_columns = []
                for col in df_display.columns:
                    if col == '指标名称_合并':
                        new_columns.append(('基本信息', '指标'))
                    elif isinstance(col, tuple):
                        new_columns.append(col)
                    else:
                        new_columns.append(col)
                df_display.columns = pd.MultiIndex.from_tuples(new_columns)
                is_multi_level = isinstance(df_display.columns, pd.MultiIndex)
                merge_first_cols = True
        
        if bin_label_col is not None:
            # 将分箱标签设为索引名称
            df_display = df_display.set_index(bin_label_col)
            # 如果有分箱列，也删除它
            if bin_col is not None:
                try:
                    df_display = df_display.drop(columns=[bin_col])
                except KeyError:
                    pass
        elif bin_col is not None:
            df_display = df_display.set_index(bin_col)
        
        # 重新检查是否为多级表头（因为可能改变了列结构）
        is_multi_level = isinstance(df_display.columns, pd.MultiIndex)

    # 限制行数
    if max_rows is not None and len(df_display) > max_rows:
        df_display = df_display.head(max_rows)

    # 紧凑模式下隐藏部分列
    if compact:
        # 确定要隐藏的列
        hide_cols = []
        if is_multi_level:
            # 多级表头
            all_cols = df_display.columns.tolist()
            # 保留核心列
            keep_patterns = ['指标', '分箱标签', '样本总数', 
                           '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '分档KS值']
            for col in all_cols:
                col_name = col[1] if isinstance(col, tuple) else col
                if not any(p in col_name for p in keep_patterns):
                    hide_cols.append(col)
        else:
            # 单层表头
            all_cols = df_display.columns.tolist()
            keep_patterns = ['指标名称', '指标含义', '指标', '分箱标签', '样本总数', 
                           '坏样本率', '分档WOE值', '分档IV值', '指标IV值', 'LIFT值', '分档KS值']
            hide_cols = [c for c in all_cols if not any(p in str(c) for p in keep_patterns)]

        if hide_cols:
            df_display = df_display.drop(columns=hide_cols)

    # 创建 Styler
    styler = df_display.style

    # 定义默认小数位数
    default_precision = {
        '样本总数': 0,
        '好样本数': 0,
        '坏样本数': 0,
        '样本占比': 2,
        '好样本占比': 2,
        '坏样本占比': 2,
        '坏样本率': 2,
        '分档WOE值': 4,
        '分档IV值': 4,
        '指标IV值': 4,
        'LIFT值': 2,
        '坏账改善': 2,
        '累积LIFT值': 2,
        '累积坏账改善': 2,
        '分档KS值': 2,
    }

    # 定义需要显示为百分比的列
    percent_columns = {
        '样本占比': 2,
        '好样本占比': 2,
        '坏样本占比': 2,
        '坏样本率': 2,
        'LIFT值': 2,
        '坏账改善': 2,
        '累积LIFT值': 2,
        '累积坏账改善': 2,
        '分档KS值': 2,
    }

    # 预定义的百分比格式
    percent_formats = {
        2: '{:.2%}',
        3: '{:.3%}',
        1: '{:.1%}',
        0: '{:.0f}',
    }

    # 更新自定义精度
    if precision:
        default_precision.update(precision)

    # 格式化数字
    if is_multi_level:
        # 多级表头
        format_dict = {}
        for col in df_display.columns:
            col_name = col[1]
            if percent_format and col_name in percent_columns:
                col_precision = percent_columns[col_name]
                format_dict[col] = percent_formats.get(col_precision, '{:.2%}')
            elif col_name in default_precision:
                precision_val = default_precision[col_name]
                if precision_val == 0:
                    format_dict[col] = '{:.0f}'
                else:
                    format_dict[col] = f'{{:.{precision_val}f}}'
        if format_dict:
            styler = styler.format(format_dict, na_rep='-')
    else:
        # 单层表头
        format_dict = {}
        for col in df_display.columns:
            col_name = str(col)
            if percent_format and col_name in percent_columns:
                col_precision = percent_columns[col_name]
                format_dict[col] = percent_formats.get(col_precision, '{:.2%}')
            elif col_name in default_precision:
                precision_val = default_precision[col_name]
                if precision_val == 0:
                    format_dict[col] = '{:.0f}'
                else:
                    format_dict[col] = f'{{:.{precision_val}f}}'
        if format_dict:
            styler = styler.format(format_dict, na_rep='-')

    # ===== 条件格式：只对坏样本率、LIFT、KS使用进度条 =====
    # 进度条颜色
    bad_rate_color = '#5B8FF9'   # 蓝色
    lift_color = '#5AD8A6'       # 绿色
    ks_color = '#F6BD16'         # 金色

    # 高亮坏样本率（进度条）
    if highlight_bad_rate:
        bad_rate_cols = _find_columns(df_display, '坏样本率')
        for col in bad_rate_cols:
            values = df_display[col].dropna()
            vmin = 0
            if len(values) > 0:
                vmax = max(values.quantile(0.95), 0.1)
                vmax = min(vmax * 1.2, 1)
            else:
                vmax = 1
            styler = styler.bar(
                subset=[col],
                color=bad_rate_color,
                vmin=vmin,
                vmax=vmax,
                axis=0
            )

    # 高亮 IV 值（已禁用：指标IV值和分档IV值不显示条件格式）

    # 高亮 LIFT 值（进度条）
    if highlight_lift:
        lift_cols = _find_columns(df_display, 'LIFT值')
        for col in lift_cols:
            lift_values = df_display[col].dropna()
            vmin = 0
            if len(lift_values) > 0:
                vmax = max(lift_values.quantile(0.95), 1.5)
                vmax = min(vmax * 1.2, 5)
            else:
                vmax = 2
            styler = styler.bar(
                subset=[col],
                color=lift_color,
                vmin=vmin,
                vmax=vmax,
                axis=0
            )

    # 高亮 KS 值（进度条）
    if highlight_ks:
        ks_cols = _find_columns(df_display, '分档KS值')
        for col in ks_cols:
            ks_values = df_display[col].dropna()
            vmin = 0
            if len(ks_values) > 0:
                vmax = max(ks_values.quantile(0.95), 0.1)
                vmax = min(vmax * 1.2, 1)
            else:
                vmax = 1
            styler = styler.bar(
                subset=[col],
                color=ks_color,
                vmin=vmin,
                vmax=vmax,
                axis=0
            )

    # 设置表格样式 - 列名不换行，横向滚动
    styler = styler.set_properties(**{
        'white-space': 'nowrap',
        'text-align': 'center',
        'font-size': '12px',
    })

    # 表头和行列样式 - 简洁清爽风格
    styler = styler.set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#f5f5f5'),
            ('color', '#333333'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '12px'),
            ('padding', '8px'),
            ('border', '1px solid #dddddd'),
            ('white-space', 'nowrap'),
        ]},
        {'selector': 'td', 'props': [
            ('padding', '6px 8px'),
            ('border', '1px solid #dddddd'),
        ]},
        # 奇偶行颜色不同
        {'selector': 'tr:nth-child(odd)', 'props': [
            ('background-color', '#ffffff'),
        ]},
        {'selector': 'tr:nth-child(even)', 'props': [
            ('background-color', '#fafafa'),
        ]},
        {'selector': 'tr:hover', 'props': [
            ('background-color', '#f0f0f0'),
        ]},
        # 进度条样式
        {'selector': '.pd-bar', 'props': [
            ('opacity', '0.7'),
        ]},
        # 选中样式 - 柔和的颜色对比
        {'selector': 'tr.selected, td.selected', 'props': [
            ('background-color', '#e3f2fd'),
            ('color', '#1565C0'),
        ]},
    ])

    return styler
    """查找匹配的列名（支持多级表头）."""
    if isinstance(df.columns, pd.MultiIndex):
        return [col for col in df.columns if pattern in str(col[1])]
    else:
        return [col for col in df.columns if pattern in str(col)]


def _color_by_iv(series: pd.Series, iv_threshold: float) -> List[str]:
    """根据 IV 值大小设置背景色."""
    colors = []
    for val in series:
        if pd.isna(val):
            colors.append('')
        elif val < 0.02:
            colors.append('background-color: #f5f5f5')  # 浅灰，无预测力
        elif val < 0.1:
            colors.append('background-color: #e3f2fd')  # 浅蓝，弱预测力
        elif val < 0.3:
            colors.append('background-color: #bbdefb')  # 中蓝，中等预测力
        elif val < 0.5:
            colors.append('background-color: #64b5f6')  # 深蓝，强预测力
        else:
            colors.append('background-color: #1976D2')  # 深蓝，超强预测力
    return colors


class BinTableDisplay:
    """分箱表展示器.
    
    提供链式调用接口，方便在 Jupyter 中展示美观的分箱表。
    
    **参考样例**
    
    >>> table = feature_bin_stats(data, 'score', target='target')
    >>> table.show()
    >>> table.show(compact=True)
    >>> table.show(highlight_iv=False)
    """
    
    def __init__(self, df: pd.DataFrame):
        """初始化展示器.
        
        :param df: 分箱统计表
        """
        self._df = df
        self._styler = None
    
    def show(
        self,
        max_rows: Optional[int] = None,
        highlight_iv: bool = True,
        highlight_bad_rate: bool = True,
        highlight_lift: bool = True,
        highlight_ks: bool = True,
        compact: bool = False,
        precision: Optional[Dict[str, int]] = None,
        index_as_bin: bool = False,
        percent_format: bool = True,
        high_tech_style: bool = False,
        **kwargs
    ) -> 'BinTableDisplay':
        """展示美化的分箱表.
        
        :param max_rows: 最大显示行数
        :param highlight_iv: 是否高亮 IV 值
        :param highlight_bad_rate: 是否高亮坏样本率
        :param highlight_lift: 是否高亮 LIFT 值
        :param highlight_ks: 是否高亮 KS 值
        :param compact: 是否使用紧凑模式
        :param precision: 自定义小数位数
        :param index_as_bin: 是否将分箱作为索引显示
        :param percent_format: 是否将百分比相关列显示为百分比格式（默认True）
        :param high_tech_style: 是否使用高科技/AI风格样式
        :param kwargs: 其他参数
        :return: self，支持链式调用
        """
        self._styler = style_bin_table(
            self._df,
            max_rows=max_rows,
            highlight_iv=highlight_iv,
            highlight_bad_rate=highlight_bad_rate,
            highlight_lift=highlight_lift,
            highlight_ks=highlight_ks,
            compact=compact,
            precision=precision,
            index_as_bin=index_as_bin,
            percent_format=percent_format,
            high_tech_style=high_tech_style
        )
        try:
            from IPython.display import display
            display(self._styler)
        except ImportError:
            # 非Jupyter环境下不执行展示
            pass
        return self
    
    def highlight_bins(self, bins: Union[int, List[int]], color: str = '#e3f2fd') -> 'BinTableDisplay':
        """高亮指定的分箱行.
        
        :param bins: 要高亮的分箱索引或索引列表
        :param color: 高亮颜色
        :return: self，支持链式调用
        """
        if self._styler is None:
            self._styler = style_bin_table(self._df)
        
        if isinstance(bins, int):
            bins = [bins]
        
        def highlight_row(row):
            if row.name in bins:
                return ['background-color: #e3f2fd; color: #1565C0'] * len(row)
            return [''] * len(row)
        
        self._styler = self._styler.apply(highlight_row, axis=1)
        try:
            from IPython.display import display
            display(self._styler)
        except ImportError:
            pass
        return self
    
    def export_html(self, filename: str) -> 'BinTableDisplay':
        """导出为 HTML 文件.
        
        :param filename: 文件名
        :return: self，支持链式调用
        """
        if self._styler is None:
            self._styler = style_bin_table(self._df)
        
        html = self._styler.to_html()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>分箱统计表</title>
                <style>
                    .dataframe {{ overflow-x: auto; }}
                    .dataframe th {{ white-space: nowrap; }}
                    .dataframe td {{ white-space: nowrap; }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            ''')
        print(f"已导出到: {filename}")
        return self
    
    def to_excel(self, filename: str, sheet_name: str = '分箱统计') -> 'BinTableDisplay':
        """导出为 Excel 文件.
        
        :param filename: 文件名
        :param sheet_name: 工作表名称
        :return: self，支持链式调用
        """
        if self._styler is None:
            self._styler = style_bin_table(self._df)
        
        self._styler.to_excel(filename, sheet_name=sheet_name, engine='openpyxl')
        print(f"已导出到: {filename}")
        return self


# 注意：DataFrame.show() 方法现在在 pandas_extensions 模块中统一注册
# 保留此处的 BinTableDisplay 类供其他代码使用


# ============================================================================
# 规则表美化函数（供 ManualTreeExtractor 等模块使用）
# ============================================================================


def style_rule_table(
    df: pd.DataFrame,
    overall_badrate: Optional[float] = None,
    precision: Optional[dict] = None,
) -> pd.DataFrame.style:
    """美化决策树规则表展示。

    使用 pandas Styler 对规则表进行格式化、高亮和颜色标注，
    使其在 Jupyter 中更易读。

    :param df: 规则表 DataFrame，列包括：
        节点编号、是否叶子、规则表达式、样本数、样本占比、
        好样本数、坏样本数、坏账率、LIFT值
    :param overall_badrate: 全局坏样本率（用于计算颜色梯度），默认从数据推断
    :param precision: 自定义小数位数，格式为 {'列名': 位数}
    :return: 格式化后的 Styler 对象

    **参考样例**

    >>> from hscredit.report.mining import ManualTreeExtractor
    >>> ext = ManualTreeExtractor(target='IS_BAD')
    >>> ext.fit(df, feature_list=['age', 'income'])
    >>> styled = style_rule_table(ext.get_rule_table())
    >>> styled.show()
    """
    df_d = df.copy()

    # 推断全局坏样本率
    if overall_badrate is None:
        total_samples = df_d['样本数'].sum()
        total_bad = df_d['坏样本数'].sum()
        overall_badrate = total_bad / total_samples if total_samples > 0 else 0.0

    # 创建 Styler
    styler = df_d.style

    # 默认精度
    default_precision = {
        '节点编号': 0,
        '样本数': 0,
        '好样本数': 0,
        '坏样本数': 0,
        '样本占比': 2,
        '坏账率': 2,
        'LIFT值': 2,
    }
    if precision:
        default_precision.update(precision)

    # 格式化为百分比和数字
    format_dict = {}
    for col, prec in default_precision.items():
        if col in df_d.columns:
            if col in ('样本占比', '坏账率'):
                # 百分数格式显示
                pct_prec = max(prec, 2)
                format_dict[col] = f'{{:.{pct_prec}%}}'
            elif prec == 0:
                format_dict[col] = '{:.0f}'
            else:
                format_dict[col] = f'{{:.{prec}f}}'
    if format_dict:
        styler = styler.format(format_dict, na_rep='-')

    # --- 坏账率颜色梯度：低=绿，高=红 ---
    if '坏账率' in df_d.columns:
        bad_vals = df_d['坏账率'].fillna(0)
        vmin, vmax = 0.0, max(bad_vals.max() * 1.2, max(overall_badrate * 1.5, 0.3), 0.01)
        styler = styler.background_gradient(
            subset=['坏账率'],
            cmap='RdYlGn_r',
            vmin=vmin,
            vmax=vmax,
        )

    # --- LIFT值颜色梯度：高=深色，低=浅色 ---
    if 'LIFT值' in df_d.columns:
        lift_vals = df_d['LIFT值'].fillna(1)
        vmin_l, vmax_l = max(0, lift_vals.min() * 0.8), lift_vals.max() * 1.2
        styler = styler.background_gradient(
            subset=['LIFT值'],
            cmap='RdYlGn',
            vmin=vmin_l,
            vmax=vmax_l,
        )

    # --- 样本占比进度条 ---
    if '样本占比' in df_d.columns:
        styler = styler.bar(
            subset=['样本占比'],
            color='#5B8FF9',
            vmin=0,
            vmax=1,
            align='left',
        )

    # --- 是否叶子列高亮 ---
    if '是否叶子' in df_d.columns:
        def _leaf_color(val):
            if val == '是':
                return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold'
            elif val == '否':
                return 'background-color: #fff8e1; color: #f57f17; font-weight: bold'
            return ''
        styler = styler.applymap(_leaf_color, subset=['是否叶子'])

    # --- 基础样式 ---
    styler = styler.set_properties(**{
        'white-space': 'nowrap',
        'text-align': 'center',
        'font-size': '12px',
        'font-family': 'Arial, sans-serif',
    })

    # --- 表头样式 ---
    styler = styler.set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#2639E9'),
            ('color', '#ffffff'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-size', '12px'),
            ('padding', '8px 10px'),
            ('border', '1px solid rgba(255,255,255,0.3)'),
        ]},
        {'selector': 'tbody td', 'props': [
            ('padding', '6px 10px'),
            ('border', '1px solid #e0e0e0'),
        ]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [
            ('background-color', '#ffffff'),
        ]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [
            ('background-color', '#f8f9ff'),
        ]},
        {'selector': 'tbody tr:hover', 'props': [
            ('background-color', '#e8eaff'),
        ]},
        # 规则表达式列左对齐
        {'selector': 'td:nth-child(3)', 'props': [
            ('text-align', 'left'),
            ('font-family', 'monospace'),
            ('font-size', '11px'),
        ]},
        # 进度条样式
        {'selector': '.pd-bar', 'props': [
            ('opacity', '0.75'),
        ]},
    ])

    return styler
