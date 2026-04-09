"""多标签规则分析报告.

基于 MultiLabelRuleMiner 生成可读的规则有效性分析 Excel 报告。
"""

import pandas as pd
from typing import Dict, List, Optional, Union

from .mining.multi_label import MultiLabelRuleMiner


def multi_label_rule_report(
    df: pd.DataFrame,
    features: List[str],
    labels: Dict[str, str],
    miner_params: Optional[dict] = None,
    output_path: str = 'rule_analysis_report.xlsx',
) -> str:
    """多标签规则挖掘分析报告（Excel输出）.

    报告包含：
    - Sheet 1 - 规则汇总：各规则在每个标签下的覆盖率/坏率/LIFT/有效性分类
    - Sheet 2 - 有效性矩阵：行=规则，列=标签，格=LIFT值
    - Sheet 3 - 规则分类统计：按规则类型分组的汇总统计

    :param df: 输入数据 DataFrame
    :param features: 参与挖掘的特征列表
    :param labels: 标签映射 {中文名: 列名}，如 {'短期标签(MOB3@30)': 'mob3_30', '长期标签(MOB6@30)': 'mob6_30'}
    :param miner_params: 传递给 MultiLabelRuleMiner 的额外参数
    :param output_path: 输出 Excel 文件路径
    :return: 输出文件路径

    示例::

        >>> multi_label_rule_report(
        ...     df=df,
        ...     features=['age', 'income', 'credit_score'],
        ...     labels={'短期标签(MOB3@30)': 'mob3_30', '长期标签(MOB6@30)': 'mob6_30'},
        ...     output_path='rule_report.xlsx',
        ... )
    """
    label_cols = list(labels.values())
    label_names = list(labels.keys())

    params = dict(
        labels=label_cols,
        label_names=label_names,
        min_support=0.02,
        min_lift=1.5,
    )
    if miner_params:
        params.update(miner_params)

    miner = MultiLabelRuleMiner(**params)
    miner.fit(df, features=features)

    # 获取各类规则
    all_rules = miner.get_report()
    matrix = miner.get_effectiveness_matrix()

    # 规则分类统计
    if len(all_rules) > 0:
        category_stats = all_rules.groupby('规则类型').agg(
            规则条数=('规则', 'count'),
            平均覆盖率=('覆盖率', 'mean'),
        ).reset_index()
    else:
        category_stats = pd.DataFrame(columns=['规则类型', '规则条数', '平均覆盖率'])

    # 输出 Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        all_rules.to_excel(writer, sheet_name='规则汇总', index=False)
        matrix.to_excel(writer, sheet_name='有效性矩阵', index=False)
        category_stats.to_excel(writer, sheet_name='规则分类统计', index=False)

    return output_path
