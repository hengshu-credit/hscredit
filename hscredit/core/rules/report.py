"""规则分析报告模块.

提供对单条规则或规则集的全面分析报告，包括：
- 覆盖率、精确率、召回率、F1、Lift、KS 等核心指标
- 训练集 vs 测试集对比
- 规则间重叠分析（覆盖重叠矩阵）
- 特征覆盖分布图
- HTML / DataFrame 报告导出
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .rule import Rule
from ..viz.utils import DEFAULT_COLORS, setup_axis_style, save_figure, get_or_create_ax


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────────────────────

def _eval_rule(rule: Rule, X: pd.DataFrame) -> pd.Series:
    """安全执行规则，返回 bool Series."""
    try:
        return rule.predict(X).astype(bool)
    except Exception:
        return pd.Series(False, index=X.index)


def _single_metrics(
    hit: pd.Series,
    y: pd.Series,
    overall_bad_rate: float,
) -> Dict[str, float]:
    """计算单条规则在一个数据集上的指标."""
    n_total   = len(y)
    n_hit     = int(hit.sum())
    n_pass    = n_total - n_hit          # 规则通过（未命中）
    n_bad     = int(y.sum())

    # 命中样本中的坏率
    hit_bad   = int(y[hit].sum()) if n_hit > 0 else 0
    hit_br    = hit_bad / n_hit if n_hit > 0 else 0.0

    # 通过样本（未被规则拦截）中的坏率
    pass_mask = ~hit
    pass_bad  = int(y[pass_mask].sum()) if n_pass > 0 else 0
    pass_br   = pass_bad / n_pass if n_pass > 0 else 0.0

    coverage  = n_hit / n_total if n_total > 0 else 0.0
    # 召回率：规则命中的坏样本占全部坏样本比例
    recall    = hit_bad / n_bad if n_bad > 0 else 0.0
    # 精确率：命中样本中坏样本比例
    precision = hit_br
    # Lift = 命中坏率 / 整体坏率
    lift      = hit_br / overall_bad_rate if overall_bad_rate > 0 else 0.0
    # 通过坏率降低量
    bad_reduce = (overall_bad_rate - pass_br) / overall_bad_rate if overall_bad_rate > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return dict(
        coverage   = coverage,
        n_hit      = n_hit,
        n_pass     = n_pass,
        hit_bad    = hit_bad,
        hit_badrate = hit_br,
        pass_bad   = pass_bad,
        pass_badrate = pass_br,
        recall     = recall,
        precision  = precision,
        f1         = f1,
        lift       = lift,
        bad_reduce = bad_reduce,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class RuleReport:
    """规则分析报告.

    对单条规则或规则列表进行全面的效果分析，
    支持训练集 / 测试集对比、规则间重叠分析和可视化。

    :param rules: 单条 Rule 或 Rule 列表
    :param target: 目标变量列名，默认 'target'

    Example:
        >>> from hscredit.core.rules import Rule
        >>> from hscredit.core.rules.report import RuleReport
        >>>
        >>> rules = [
        ...     Rule("age < 22",        name="年龄过小"),
        ...     Rule("income < 3000",   name="收入过低"),
        ...     Rule("debt_ratio > 0.8",name="负债率高"),
        ... ]
        >>> report = RuleReport(rules, target='is_bad')
        >>> df_metrics = report.evaluate(df_train, df_test)
        >>> report.plot_metrics()
        >>> report.plot_overlap(df_train)
        >>> report.to_html('rule_report.html')
    """

    def __init__(
        self,
        rules: Union[Rule, List[Rule]],
        target: str = 'target',
    ):
        if isinstance(rules, Rule):
            rules = [rules]
        self.rules  = rules
        self.target = target
        self._train_metrics: Optional[pd.DataFrame] = None
        self._test_metrics:  Optional[pd.DataFrame] = None
        self._overlap_matrix: Optional[pd.DataFrame] = None

    # ── 核心评估 ──────────────────────────────────────────────────────

    def evaluate(
        self,
        df_train: pd.DataFrame,
        df_test:  Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """计算所有规则的评估指标.

        :param df_train: 训练集（含目标列）
        :param df_test:  测试集（含目标列，可选）
        :return: 汇总 DataFrame（含 train / test 对比列）
        """
        y_tr = df_train[self.target]
        br_tr = float(y_tr.mean())

        rows = []
        for rule in self.rules:
            hit_tr = _eval_rule(rule, df_train)
            m_tr   = _single_metrics(hit_tr, y_tr, br_tr)
            row    = {'规则名称': rule.name, '规则表达式': rule.expr}
            for k, v in m_tr.items():
                row[f'训练_{k}'] = v

            if df_test is not None:
                y_te  = df_test[self.target]
                br_te = float(y_te.mean())
                hit_te = _eval_rule(rule, df_test)
                m_te   = _single_metrics(hit_te, y_te, br_te)
                for k, v in m_te.items():
                    row[f'测试_{k}'] = v
                # 稳定性：覆盖率漂移
                row['覆盖率漂移'] = m_te['coverage'] - m_tr['coverage']
                row['命中坏率漂移'] = m_te['hit_badrate'] - m_tr['hit_badrate']

            rows.append(row)

        self._train_metrics = pd.DataFrame(rows)
        return self._train_metrics

    def summary(
        self,
        df: pd.DataFrame,
        sort_by: str = 'lift',
        ascending: bool = False,
    ) -> pd.DataFrame:
        """返回简洁的规则摘要表（关键指标）.

        :param df: 数据集（含目标列）
        :param sort_by: 排序字段，默认 'lift'
        :param ascending: 是否升序，默认 False
        :return: 简洁摘要 DataFrame
        """
        y  = df[self.target]
        br = float(y.mean())
        rows = []
        for rule in self.rules:
            hit = _eval_rule(rule, df)
            m   = _single_metrics(hit, y, br)
            rows.append({
                '规则名称':   rule.name,
                '规则表达式': rule.expr,
                '覆盖率':     m['coverage'],
                '命中坏率':   m['hit_badrate'],
                '通过坏率':   m['pass_badrate'],
                '召回率':     m['recall'],
                '精确率':     m['precision'],
                'F1':         m['f1'],
                'Lift':       m['lift'],
                '坏率降低量': m['bad_reduce'],
                '命中样本数': m['n_hit'],
                '命中坏样本': m['hit_bad'],
            })
        df_out = pd.DataFrame(rows)
        col_map = {
            'lift':       'Lift',
            'coverage':   '覆盖率',
            'recall':     '召回率',
            'precision':  '精确率',
            'f1':         'F1',
            'bad_reduce': '坏率降低量',
        }
        sort_col = col_map.get(sort_by.lower(), sort_by)
        if sort_col in df_out.columns:
            df_out = df_out.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        return df_out

    # ── 重叠分析 ──────────────────────────────────────────────────────

    def overlap_matrix(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算规则间覆盖重叠矩阵.

        矩阵 [i, j] = 同时被规则 i 和规则 j 命中的样本占规则 i 命中样本的比例。

        :param df: 数据集
        :return: 重叠率矩阵 DataFrame
        """
        hits = {}
        for rule in self.rules:
            hits[rule.name] = _eval_rule(rule, df)

        names = [r.name for r in self.rules]
        mat   = pd.DataFrame(index=names, columns=names, dtype=float)
        for ni, ri in zip(names, self.rules):
            for nj, rj in zip(names, self.rules):
                n_i = hits[ni].sum()
                if n_i == 0:
                    mat.loc[ni, nj] = 0.0
                else:
                    mat.loc[ni, nj] = float((hits[ni] & hits[nj]).sum()) / n_i

        self._overlap_matrix = mat
        return mat

    def combined_coverage(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """计算规则集整体联合覆盖率（任意一条命中即覆盖）.

        :param df: 数据集
        :return: dict 含 coverage / hit_badrate / pass_badrate / recall
        """
        y  = df[self.target]
        br = float(y.mean())
        any_hit = pd.Series(False, index=df.index)
        for rule in self.rules:
            any_hit = any_hit | _eval_rule(rule, df)
        return _single_metrics(any_hit, y, br)

    # ── 可视化 ────────────────────────────────────────────────────────

    def plot_metrics(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        figsize: tuple = (14, 6),
        title: str = '规则效果对比',
        save: Optional[str] = None,
    ) -> plt.Figure:
        """绘制规则关键指标对比柱状图.

        :param df: 数据集（含目标列）
        :param metrics: 要展示的指标列表，默认 ['Lift','覆盖率','召回率','命中坏率']
        :param figsize: 图像尺寸
        :param title: 图标题
        :param save: 保存路径
        :return: Figure
        """
        if metrics is None:
            metrics = ['Lift', '覆盖率', '召回率', '命中坏率']

        summ = self.summary(df, sort_by='Lift')
        names = summ['规则名称'].tolist()
        n_rules  = len(names)
        n_metrics = len(metrics)

        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(figsize[0], figsize[1]),
            sharey=False
        )
        if n_metrics == 1:
            axes = [axes]

        x = np.arange(n_rules)
        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            vals = summ[metric].values.astype(float)
            color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
            bars = ax.barh(x, vals, color=color, alpha=0.85, edgecolor='white')

            # 数值标注
            for bar, val in zip(bars, vals):
                fmt = f'{val:.2f}' if metric == 'Lift' else f'{val:.1%}'
                ax.text(
                    bar.get_width() + bar.get_width() * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    fmt, va='center', fontsize=8, color=color
                )

            ax.set_yticks(x)
            ax.set_yticklabels(
                [n[:20] for n in names], fontsize=9)
            ax.set_xlabel(metric, fontsize=10)
            ax.set_title(metric, fontsize=11, fontweight='bold')
            setup_axis_style(ax, hide_top_right=True)

            # Lift 图添加基准线
            if metric == 'Lift':
                ax.axvline(1.0, color=DEFAULT_COLORS[7],
                           linewidth=1.2, linestyle='--', alpha=0.7)

        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
        fig.tight_layout()
        if save:
            save_figure(fig, save)
        return fig

    def plot_overlap(
        self,
        df: pd.DataFrame,
        figsize: tuple = (8, 6),
        title: str = '规则覆盖重叠热力图',
        save: Optional[str] = None,
    ) -> plt.Figure:
        """绘制规则间覆盖重叠热力图.

        :param df: 数据集
        :param figsize: 图像尺寸
        :param title: 图标题
        :param save: 保存路径
        :return: Figure
        """
        import seaborn as sns
        mat = self.overlap_matrix(df)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mat.astype(float),
            annot=True, fmt='.1%',
            cmap=sns.diverging_palette(340, 267, n=256, s=90, l=40),
            vmin=0, vmax=1,
            ax=ax,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': '重叠率', 'shrink': 0.8}
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('规则 j', fontsize=10)
        ax.set_ylabel('规则 i', fontsize=10)
        ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        if save:
            save_figure(fig, save)
        return fig

    def plot_coverage_bar(
        self,
        df: pd.DataFrame,
        figsize: tuple = (12, 5),
        title: str = '规则覆盖分布',
        save: Optional[str] = None,
    ) -> plt.Figure:
        """绘制每条规则的命中/未命中好坏样本堆叠柱状图.

        :param df: 数据集（含目标列）
        :param figsize: 图像尺寸
        :param title: 图标题
        :param save: 保存路径
        :return: Figure
        """
        y  = df[self.target]
        br = float(y.mean())
        names, n_good_hit, n_bad_hit, n_good_pass, n_bad_pass, lifts = [], [], [], [], [], []

        for rule in self.rules:
            hit = _eval_rule(rule, df)
            m   = _single_metrics(hit, y, br)
            names.append(rule.name)
            n_bad_hit.append(m['hit_bad'])
            n_good_hit.append(m['n_hit'] - m['hit_bad'])
            n_bad_pass.append(m['pass_bad'])
            n_good_pass.append(m['n_pass'] - m['pass_bad'])
            lifts.append(m['lift'])

        x = np.arange(len(names))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                        gridspec_kw={'width_ratios': [3, 1]})

        # 左图：堆叠柱状图
        ax1.bar(x, n_good_hit, label='命中好样本', color=DEFAULT_COLORS[0], alpha=0.85)
        ax1.bar(x, n_bad_hit,  bottom=n_good_hit,
                label='命中坏样本', color=DEFAULT_COLORS[1], alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels([n[:18] for n in names], rotation=30, ha='right', fontsize=9)
        ax1.set_ylabel('样本数', fontsize=11)
        ax1.set_title('命中样本分布', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper right')
        setup_axis_style(ax1, hide_top_right=True)

        # 右图：Lift 柱状图
        bar_colors = [
            DEFAULT_COLORS[1] if lv >= 2.0
            else DEFAULT_COLORS[3] if lv >= 1.5
            else DEFAULT_COLORS[5] if lv >= 1.0
            else DEFAULT_COLORS[7]
            for lv in lifts
        ]
        ax2.barh(x, lifts, color=bar_colors, alpha=0.85, edgecolor='white')
        ax2.axvline(1.0, color=DEFAULT_COLORS[7], linewidth=1.2,
                    linestyle='--', alpha=0.7, label='Lift=1')
        for xi, lv in zip(x, lifts):
            ax2.text(lv + 0.02, xi, f'{lv:.2f}',
                     va='center', fontsize=8, color=DEFAULT_COLORS[1])
        ax2.set_yticks(x)
        ax2.set_yticklabels([n[:16] for n in names], fontsize=9)
        ax2.set_xlabel('Lift', fontsize=11)
        ax2.set_title('规则 Lift', fontsize=11, fontweight='bold')
        setup_axis_style(ax2, hide_top_right=True)

        fig.suptitle(title, fontsize=13, fontweight='bold')
        fig.tight_layout()
        if save:
            save_figure(fig, save)
        return fig

    # ── 报告导出 ──────────────────────────────────────────────────────

    def to_dataframe(
        self,
        df: pd.DataFrame,
        sort_by: str = 'Lift',
    ) -> pd.DataFrame:
        """导出规则摘要 DataFrame."""
        return self.summary(df, sort_by=sort_by)

    def to_html(
        self,
        filepath: str,
        df: Optional[pd.DataFrame] = None,
        title: str = 'hscredit 规则分析报告',
    ) -> str:
        """导出 HTML 报告.

        :param filepath: 输出 HTML 文件路径
        :param df: 数据集（含目标列）；若已调用 evaluate()，可传 None
        :param title: 报告标题
        :return: HTML 字符串
        """
        from datetime import datetime
        import os

        if df is not None:
            summ = self.summary(df)
        elif self._train_metrics is not None:
            summ = self._train_metrics
        else:
            raise ValueError('请先调用 evaluate() 或传入 df 参数')

        # 格式化百分比列
        pct_cols = [c for c in summ.columns
                    if any(k in c for k in ['率', 'recall', 'precision', 'coverage', 'f1'])]
        summ_fmt = summ.copy()
        for col in pct_cols:
            if col in summ_fmt.columns:
                summ_fmt[col] = summ_fmt[col].apply(
                    lambda v: f'{v:.2%}' if isinstance(v, float) else v)

        table_html = summ_fmt.to_html(
            index=False, classes='rule-table', border=0,
            escape=False
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
       background: #f7f8fc; color: #333; padding: 24px; }}
h1 {{ color: {DEFAULT_COLORS[0]}; border-bottom: 3px solid {DEFAULT_COLORS[0]};
     padding-bottom: 8px; }}
h2 {{ color: {DEFAULT_COLORS[0]}; margin-top: 32px; }}
.rule-table {{ border-collapse: collapse; width: 100%;
               background: #fff; border-radius: 8px;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.rule-table th {{ background: {DEFAULT_COLORS[0]}; color: #fff;
                   padding: 10px 14px; text-align: center; font-size: 13px; }}
.rule-table td {{ padding: 8px 14px; text-align: center;
                   border-bottom: 1px solid #eee; font-size: 12px; }}
.rule-table tr:nth-child(even) {{ background: #f0f4ff; }}
.rule-table tr:hover {{ background: #dde8ff; }}
.timestamp {{ color: #999; font-size: 12px; margin-top: 40px; }}
</style></head>
<body>
<h1>{title}</h1>
<p>规则数量：<strong>{len(self.rules)}</strong></p>
<h2>规则效果摘要</h2>
{table_html}
<p class="timestamp">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</body></html>"""

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        return html

    def __repr__(self) -> str:
        return f'RuleReport(n_rules={len(self.rules)}, target={self.target!r})'
 