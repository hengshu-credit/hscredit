"""Vintage分析模块.

提供账龄(Vintage)分析、滚动率分析等金融风控特有功能.

**引用**

Vintage（账龄/世代）分析与 Roll Rate（滚动率）矩阵是消费信贷资产质量监控的
标准工具，见 Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley，及
Thomas, L. C. (2009). *Consumer Credit Models.* Oxford University Press。
"""

import numpy as np
import pandas as pd
import warnings
from typing import List, Dict, Optional, Union

from .utils import validate_dataframe, validate_binary_target


def vintage_analysis(df: pd.DataFrame,
                    vintage_col: str,
                    mob_col: str,
                    target_col: str,
                    max_mob: int = 12,
                    *, entity_col: Optional[str] = None,
                    input_mode: str = "legacy",
                    label_mode: Optional[str] = None,
                    observation_end=None,
                    cohort_sizes: Optional[Dict] = None,
                    exposure_col: Optional[str] = None) -> pd.DataFrame:
    """Vintage账龄分析.
    
    追踪不同放款批次（Vintage）随账龄（MOB）的风险表现变化
    
    :param df: 输入数据
    :param vintage_col: Vintage批次列（如放款月份）
    :param mob_col: 账龄列（Month on Book）
    :param target_col: 目标变量列（如是否逾期）
    :param max_mob: 最大账龄
    :param entity_col: 显式 panel/snapshot/event 模式所需的唯一账户或合同字段
    :param input_mode: legacy 保留旧累计行口径（不是账户面板）；panel 为账户×MOB；
        snapshot 每账户一行；event 为新增违约事件，必须提供 cohort_sizes 和 observation_end
    :param label_mode: panel/snapshot 必须指定 ever 或 current；event 必须为 new_default
    :param observation_end: 观察截止日期；按放款批次自然月与截止自然月之差判断成熟 MOB，
        未成熟格输出 NaN，不视为好样本。指定时 vintage_col 必须可解析为日期/月
    :param cohort_sizes: 每批次独立开户数；event 必需，panel 可用于声明完整账户总体
    :param exposure_col: panel/snapshot 每条观测的非负金额字段；输出同一MOB的好/坏金额及金额坏账率，
        不把跨期余额相加。缺失金额拒绝计算，零金额分母输出NaN
    :return: Vintage分析DataFrame
    
    **参考样例**

    >>> vintage = vintage_analysis(df, 'issue_month', 'mob', 'ever_dpd30', max_mob=12)
    >>> print(vintage.pivot(index='MOB', columns='Vintage批次', values='累积坏账率(%)'))
    """
    validate_dataframe(df, required_cols=[vintage_col, mob_col, target_col], check_empty=input_mode != "event")
    validate_binary_target(df[target_col])
    if isinstance(max_mob, bool) or not isinstance(max_mob, (int, np.integer)) or max_mob < 0:
        raise ValueError("max_mob 必须为非负整数")
    if input_mode not in {"legacy", "panel", "snapshot", "event"}:
        raise ValueError("input_mode 必须为 legacy、panel、snapshot 或 event")
    if input_mode != "legacy":
        return _explicit_vintage(df, vintage_col, mob_col, target_col, max_mob,
                                 entity_col, input_mode, label_mode, observation_end, cohort_sizes, exposure_col)
    if entity_col is not None or observation_end is not None or cohort_sizes is not None or label_mode is not None or exposure_col is not None:
        raise ValueError("指定账户、标签或成熟度口径时必须显式选择 input_mode，不能混用 legacy")
    warnings.warn("legacy Vintage 保留累计行口径，不适用于账户×MOB的 ever 面板；请显式设置 input_mode、entity_col、label_mode", FutureWarning, stacklevel=2)
    
    # 按Vintage和MOB汇总
    grouped = df.groupby([vintage_col, mob_col]).agg({
        target_col: ['count', 'sum']
    }).reset_index()
    
    grouped.columns = ['Vintage批次', 'MOB', '开户数', '坏账户数']
    
    # 筛选最大账龄
    grouped = grouped[grouped['MOB'] <= max_mob]
    
    # 计算累积坏账率
    results = []
    
    for vintage in grouped['Vintage批次'].unique():
        vintage_data = grouped[grouped['Vintage批次'] == vintage].sort_values('MOB')
        
        # 计算累积
        total_accounts = vintage_data['开户数'].sum()
        
        cum_bad = 0
        for _, row in vintage_data.iterrows():
            cum_bad += row['坏账户数']
            bad_rate = cum_bad / total_accounts * 100 if total_accounts > 0 else 0
            
            results.append({
                'Vintage批次': vintage,
                'MOB': int(row['MOB']),
                '当期开户数': int(row['开户数']),
                '当期坏账户数': int(row['坏账户数']),
                '累积坏账户数': int(cum_bad),
                '累积坏账率(%)': round(bad_rate, 2),
            })
    
    result = pd.DataFrame(results)
    result.attrs["统计口径"] = {"input_mode": "legacy", "说明": "跨MOB累计行口径，非账户面板"}
    return result


def _explicit_vintage(df, vintage_col, mob_col, target_col, max_mob, entity_col,
                      input_mode, label_mode, observation_end, cohort_sizes, exposure_col):
    """有明确账户、标签及观察窗口契约的账龄统计。"""
    if entity_col is None:
        raise ValueError("显式 Vintage 模式必须指定 entity_col")
    validate_dataframe(df, required_cols=[entity_col], check_empty=input_mode != "event")
    expected_modes = {"new_default"} if input_mode == "event" else {"ever", "current"}
    if label_mode not in expected_modes:
        raise ValueError(f"{input_mode} 模式必须显式指定 label_mode 为 {sorted(expected_modes)}")
    columns = [vintage_col, mob_col, target_col, entity_col]
    if exposure_col is not None:
        if input_mode == "event":
            raise ValueError("event模式不能由违约事件金额推断整体金额分母；exposure_col仅支持panel/snapshot")
        validate_dataframe(df, required_cols=[exposure_col])
        columns.append(exposure_col)
    data = df[list(dict.fromkeys(columns))].copy()
    if exposure_col is not None:
        exposure = pd.to_numeric(data[exposure_col], errors="coerce")
        if exposure.isna().any() or not np.isfinite(exposure).all() or exposure.lt(0).any():
            raise ValueError("exposure_col金额必须为非缺失、非负有限数值")
        data[exposure_col] = exposure
    if data[[vintage_col, entity_col, mob_col]].isna().any().any():
        raise ValueError("Vintage批次、账户标识和MOB不能缺失")
    mobs = pd.to_numeric(data[mob_col], errors="coerce")
    if mobs.isna().any() or not np.isfinite(mobs).all() or (mobs < 0).any() or (mobs % 1 != 0).any():
        raise ValueError("MOB必须为非负整数")
    data[mob_col] = mobs.astype(int)
    if data.groupby(entity_col, sort=False)[vintage_col].nunique().gt(1).any():
        raise ValueError("同一账户不能属于多个Vintage批次")
    if input_mode == "panel":
        if data.duplicated([entity_col, mob_col]).any():
            raise ValueError("账户×MOB存在重复记录，请先明确去重规则")
        if label_mode == "ever":
            observed = data.dropna(subset=[target_col]).sort_values(mob_col)
            if observed.groupby(entity_col, sort=False)[target_col].diff().lt(0).any():
                raise ValueError("ever标签不能从1恢复为0；当期逾期请使用 label_mode='current'")
    elif data.duplicated(entity_col).any():
        raise ValueError("snapshot/event模式每个账户只能有一条记录")
    if input_mode == "event":
        if not isinstance(cohort_sizes, dict) or not cohort_sizes or observation_end is None:
            raise ValueError("event模式必须声明 cohort_sizes 和 observation_end，事件表本身不能推断开户分母")
        if not data[target_col].eq(1).all():
            raise ValueError("event模式只接受首次新增违约事件，目标必须均为1")
    cutoff = None
    if observation_end is not None:
        try:
            cutoff = pd.Timestamp(observation_end).to_period("M")
        except (TypeError, ValueError) as exc:
            raise ValueError("observation_end 必须为有效日期") from exc
        if pd.isna(cutoff):
            raise ValueError("observation_end 不能缺失")
    points = list(range(max_mob + 1)) if cutoff is not None else sorted(data.loc[data[mob_col] <= max_mob, mob_col].unique())
    rows = []
    cohorts = {vintage: cohort for vintage, cohort in data.groupby(vintage_col, sort=False)}
    if input_mode == "event":
        for vintage in cohort_sizes:
            cohorts.setdefault(vintage, data.iloc[:0])
    for vintage, cohort in cohorts.items():
        seen = cohort[entity_col].nunique()
        count = (cohort_sizes or {}).get(vintage, seen)
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < seen:
            raise ValueError(f"批次 {vintage} 的开户数必须为不少于已知账户数的整数")
        if input_mode == "event" and vintage not in cohort_sizes:
            raise ValueError(f"cohort_sizes 缺少批次: {vintage}")
        mature_mob = None
        if cutoff is not None:
            try:
                origin = vintage.asfreq("M") if isinstance(vintage, pd.Period) else pd.Timestamp(vintage).to_period("M")
                mature_mob = cutoff.ordinal - origin.ordinal
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"批次 {vintage} 无法解析为自然月") from exc
            if cohort[mob_col].gt(mature_mob).any():
                raise ValueError(f"批次 {vintage} 含观察截止日之后的MOB记录")
        for mob in points:
            snapshot = cohort[cohort[mob_col] == mob]
            valid = snapshot[target_col].notna()
            observed_count = int(valid.sum())
            mature = mature_mob is None or mob <= mature_mob
            if input_mode == "event":
                observed_count = count if mature else 0
                bad_count = int(cohort[mob_col].le(mob).sum())
                denominator = count
            else:
                bad_count = int(snapshot.loc[valid, target_col].sum())
                denominator = count if input_mode == "panel" else len(snapshot)
            status = "未成熟" if not mature else ("观测不足" if observed_count < denominator or observed_count == 0 else "成功")
            rate = bad_count / denominator * 100 if status == "成功" and denominator else np.nan
            rows.append({"Vintage批次": vintage, "MOB": int(mob), "总开户数": int(count),
                         "当期开户数": observed_count, "当期坏账户数": int(snapshot[target_col].sum()),
                         "累积坏账户数": bad_count if label_mode != "current" and status == "成功" else np.nan,
                         "累积坏账率(%)": round(rate, 2) if label_mode != "current" else np.nan,
                         "统计坏账户数": bad_count if status == "成功" else np.nan,
                         "统计坏账率(%)": round(rate, 2), "有效分母": denominator if mature else np.nan,
                         "状态": status, "标签口径": label_mode})
            if exposure_col is not None:
                amount = float(snapshot[exposure_col].sum())
                bad_amount = float(snapshot.loc[snapshot[target_col].eq(1), exposure_col].sum())
                available = status == "成功"
                rows[-1].update({"有效金额分母": amount if available else np.nan,
                                 "好账户金额": amount - bad_amount if available else np.nan,
                                 "坏账金额": bad_amount if available else np.nan,
                                 "金额坏账率(%)": round(bad_amount / amount * 100, 2) if available and amount > 0 else np.nan,
                                 "金额状态": status if not available else ("成功" if amount > 0 else "零金额分母")})
    result = pd.DataFrame(rows)
    result.attrs["统计口径"] = {"input_mode": input_mode, "label_mode": label_mode,
                                "entity_col": entity_col, "observation_end": str(observation_end),
                                "exposure_col": exposure_col,
                                "成熟口径": "批次自然月与观察截止自然月之差"}
    return result


def vintage_summary(df: pd.DataFrame,
                   vintage_col: str,
                   mob_col: str,
                   target_col: str,
                   max_mob: int = 12, **kwargs) -> pd.DataFrame:
    """Vintage汇总统计.
    
    :param df: 输入数据
    :param vintage_col: Vintage批次列
    :param mob_col: 账龄列
    :param target_col: 目标变量列
    :param max_mob: 最大账龄
    :return: Vintage汇总DataFrame
    
    **参考样例**

    >>> summary = vintage_summary(df, 'issue_month', 'mob', 'ever_dpd30')
    >>> print(summary[['Vintage批次', '总开户数', f'MOB{max_mob}坏账率']])
    """
    vintage_df = vintage_analysis(df, vintage_col, mob_col, target_col, max_mob, **kwargs)
    
    if vintage_df.empty:
        return pd.DataFrame()
    
    # 汇总每个Vintage
    results = []
    
    for vintage in vintage_df['Vintage批次'].unique():
        vintage_data = vintage_df[vintage_df['Vintage批次'] == vintage]
        
        total_accounts = vintage_data['总开户数'].iloc[0] if '总开户数' in vintage_data else vintage_data['当期开户数'].sum()
        
        # 各MOB点的坏账率
        result = {
            'Vintage批次': vintage,
            '总开户数': int(total_accounts),
        }
        
        for mob in range(max_mob + 1):
            mob_data = vintage_data[vintage_data['MOB'] == mob]
            if len(mob_data) > 0:
                rate_col = '统计坏账率(%)' if '统计坏账率(%)' in mob_data else '累积坏账率(%)'
                result[f'MOB{mob}坏账率(%)'] = mob_data[rate_col].values[0]
                if '金额坏账率(%)' in mob_data:
                    result[f'MOB{mob}金额坏账率(%)'] = mob_data['金额坏账率(%)'].iloc[0]
            else:
                result[f'MOB{mob}坏账率(%)'] = np.nan
        
        results.append(result)
    
    return pd.DataFrame(results)


def roll_rate_analysis(df: pd.DataFrame,
                      overdue_cols: List[str],
                      labels: List[str] = None) -> pd.DataFrame:
    """滚动率分析.
    
    分析不同逾期状态之间的转化情况
    
    :param df: 输入数据
    :param overdue_cols: 各期逾期状态列（如['mob1_status', 'mob2_status', 'mob3_status']）
    :param labels: 状态标签（如['M0', 'M1', 'M2+']）
    :return: 滚动率分析DataFrame
    
    **参考样例**

    >>> roll = roll_rate_analysis(df, ['mob1', 'mob2', 'mob3'], ['M0', 'M1', 'M2+'])
    >>> print(roll)
    """
    validate_dataframe(df, required_cols=overdue_cols)
    
    if labels is None:
        labels = [f'状态{i+1}' for i in range(len(overdue_cols))]
    
    # 计算各状态分布
    results = []
    
    for i, col in enumerate(overdue_cols):
        value_counts = df[col].value_counts()
        total = len(df)
        
        for status, count in value_counts.items():
            results.append({
                '期数': labels[i],
                '状态': status,
                '户数': int(count),
                '占比(%)': round(count / total * 100, 2),
            })
    
    return pd.DataFrame(results)
