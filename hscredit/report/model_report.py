# -*- coding: utf-8 -*-
"""模型评估报告一键生成函数.

提供 ``auto_model_report()`` 函数，一行代码输出训练+测试集全面报告：
- 控制台打印核心指标/LIFT表/单调性
- 可选导出 Excel（每个分析为一个Sheet）
- 可选导出 HTML（可在浏览器查看）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def auto_model_report(
    model,
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    feature_names: Optional[List[str]] = None,
    ratios: List[float] = None,
    excel_path: Optional[str] = None,
    html_path: Optional[str] = None,
    show_lift: bool = True,
    show_importance: bool = True,
    verbose: bool = True,
) -> 'ModelReport':
    """一键生成风控模型完整评估报告.

    整合 ModelReport 所有功能，支持控制台打印、Excel 和 HTML 导出。

    :param model: 已训练的风控模型（BaseRiskModel 子类）
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征（可选）
    :param y_test: 测试集标签（可选）
    :param feature_names: 特征名称列表（可选，自动从 X_train.columns 推断）
    :param ratios: LIFT 分析覆盖率列表，默认 [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]
    :param excel_path: Excel 报告保存路径（None 则不导出）
    :param html_path: HTML 报告保存路径（None 则不导出）
    :param show_lift: 是否打印 LIFT 分析，默认 True
    :param show_importance: 是否打印特征重要性，默认 True
    :param verbose: 是否打印报告到控制台，默认 True
    :return: ModelReport 对象（可进一步调用其他方法）

    Example:
        >>> from hscredit.report.model_report import auto_model_report
        >>> report = auto_model_report(
        ...     model, X_train, y_train, X_test, y_test,
        ...     excel_path='model_report.xlsx',
        ...     html_path='model_report.html',
        ... )
        >>> # 后续可单独获取
        >>> lift_df = report.get_lift_analysis()
        >>> summary  = report.summary()
    """
    from ..core.models.report import ModelReport

    report = ModelReport(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )

    if verbose:
        report.print_report(show_lift=show_lift)

    if excel_path:
        try:
            report.to_excel(excel_path)
            if verbose:
                print(f'\nExcel 报告已保存: {excel_path}')
        except ImportError:
            print('[警告] 导出 Excel 需要 openpyxl，请安装: pip install openpyxl')
        except Exception as e:
            print(f'[警告] Excel 导出失败: {e}')

    if html_path:
        try:
            report.to_html(html_path)
            if verbose:
                print(f'HTML 报告已保存: {html_path}')
        except Exception as e:
            print(f'[警告] HTML 导出失败: {e}')

    return report


def compare_models(
    models: Dict[str, object],
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    ratios: List[float] = None,
    excel_path: Optional[str] = None,
) -> pd.DataFrame:
    """多模型指标对比表.

    :param models: 模型字典，键=模型名称，值=已训练模型
    :param X_train: 训练集特征
    :param y_train: 训练集标签
    :param X_test: 测试集特征（可选）
    :param y_test: 测试集标签（可选）
    :param ratios: LIFT 覆盖率列表
    :param excel_path: 保存路径（可选）
    :return: 对比 DataFrame，行=模型，列=各指标

    Example:
        >>> from hscredit.report.model_report import compare_models
        >>> comp = compare_models(
        ...     {'XGBoost': xgb, 'LightGBM': lgb, 'LR': lr},
        ...     X_train, y_train, X_test, y_test
        ... )
        >>> print(comp)
    """
    from ..core.models.report import ModelReport

    if ratios is None:
        ratios = [0.01, 0.03, 0.05, 0.10]

    rows = []
    for name, model in models.items():
        try:
            rep = ModelReport(
                model=model,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
            )
            row = rep.summary().iloc[0].to_dict()
            row['模型'] = name
            rows.append(row)
        except Exception as e:
            rows.append({'模型': name, '错误': str(e)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 把 '模型' 移到最前
    cols = ['模型'] + [c for c in df.columns if c != '模型']
    df = df[cols].reset_index(drop=True)

    if excel_path:
        try:
            df.to_excel(excel_path, index=False)
            print(f'模型对比表已保存: {excel_path}')
        except Exception as e:
            print(f'[警告] 保存失败: {e}')

    return df


__all__ = ['auto_model_report', 'compare_models']
