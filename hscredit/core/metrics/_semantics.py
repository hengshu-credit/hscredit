"""结果表共用的拒绝指标、统计口径与末端格式化。"""

import numpy as np


def validate_amounts(values, length=None):
    """校验金额的形状、有限性与非负性，禁止把非法金额静默改为0。"""
    try:
        amounts = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("amount金额必须为有限非负的一维数值") from exc
    if (
        amounts.ndim != 1
        or (length is not None and len(amounts) != length)
        or not np.isfinite(amounts).all()
        or np.any(amounts < 0)
    ):
        raise ValueError("amount金额必须为与数据等长的有限非负一维数组，不能包含缺失")
    return amounts


def rejection_statistics(total, bad, rejected_total, rejected_bad):
    """返回剩余坏率、兼容改善值和拒绝效率；全拒绝的效率为 NaN。"""
    total, bad, rejected_total, rejected_bad = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in (total, bad, rejected_total, rejected_bad)]
    )
    remaining = total - rejected_total
    # 全量累计可能因浮点求和次序留下舍入残差，不能视作真实通过人群。
    full_rejection = np.isclose(rejected_total, total, rtol=8 * np.finfo(float).eps, atol=0)
    valid_remaining = (total > 0) & (remaining > 0) & ~full_rejection
    remaining_rate = np.full(total.shape, np.nan)
    np.divide(np.maximum(bad - rejected_bad, 0), remaining, out=remaining_rate, where=valid_remaining)
    baseline = np.zeros(total.shape)
    np.divide(bad, total, out=baseline, where=total > 0)
    improvement = np.zeros(total.shape)
    compatible_remaining_rate = np.where(valid_remaining, remaining_rate, 0.0)
    np.divide(baseline - compatible_remaining_rate, baseline, out=improvement, where=baseline > 0)
    rejected_share = np.zeros(total.shape)
    np.divide(rejected_total, total, out=rejected_share, where=total > 0)
    efficiency = np.zeros(total.shape)
    np.divide(improvement, rejected_share, out=efficiency, where=rejected_share > 0)
    efficiency = np.where((total > 0) & full_rejection, np.nan, efficiency)
    return remaining_rate, improvement, efficiency


def metric_metadata(target_type="binary", *, total_good=None, total_bad=None, woe_clip=None, round_digits=False):
    """生成可序列化的口径说明，不保存逐行数据或修改调用方状态。"""
    insufficient = total_good is not None and total_bad is not None and (total_good <= 0 or total_bad <= 0)
    return {
        "口径版本": "2026-10-27-r2",
        "目标类型": target_type,
        "统计单位": "金额" if target_type == "amount_weighted" else "样本",
        "正类": 1,
        "坏样本率分母": "当前箱样本量或金额",
        "好坏样本占比分母": "各自类别的全量样本量或金额；空类别占比为0",
        "坏账改善": "(总体坏率-拒绝后剩余人群坏率)/总体坏率",
        "风险拒绝比": "坏账改善/拒绝占比",
        "边界值策略": "坏账改善保留兼容值；全部拒绝时风险拒绝比为NaN；结果表只展示数值",
        "累计方向": "从正常箱坏率较高的一端累计，保留箱最后加入；非按坏率重排所有箱",
        "KS状态": "不适用：单类或某类有效金额为0，返回兼容值0" if insufficient else "按全量好坏类别各自累计",
        "KS单类策略": "缺少任一类有效样本或金额时为不适用，返回0兼容值；多标签分别判断",
        "WOE方向": "ln(坏样本占比/好样本占比)",
        "IV口径": "截断后WOE参与IV；原始WOE/IV另列" if woe_clip is not None else "未截断WOE参与IV",
        "WOE截断": woe_clip,
        "展示舍入": bool(round_digits),
    }


def format_bin_stats(table, *, target_type="binary", round_digits=True):
    """只在输出末端舍入，兼容平面和多级列，不原地修改表。"""
    if not round_digits:
        return table
    decimals = {
        **{
            name: 6
            for name in [
                "样本占比",
                "好样本占比",
                "坏样本占比",
                "坏样本率",
                "分档WOE值",
                "分档IV值",
                "指标IV值",
                "分档KS值",
                "原始分档WOE值",
                "原始分档IV值",
                "原始指标IV值",
                "目标值占比",
            ]
        },
        **{
            name: 4
            for name in [
                "LIFT值",
                "坏账改善",
                "风险拒绝比",
                "累积LIFT值",
                "累积坏账改善",
                "累计风险拒绝比",
                "平均LIFT值",
                "贡献改善",
                "累积贡献改善",
            ]
        },
    }
    if target_type == "amount_weighted":
        decimals.update({name: 2 for name in ["样本总数", "好样本数", "坏样本数", "累积好样本数", "累积坏样本数"]})
    elif target_type == "continuous":
        decimals.update(
            {name: 2 for name in ["目标值均值", "目标值标准差", "目标值最小值", "目标值最大值", "累积平均值"]}
        )
    result = table.copy()
    for column in result.columns:
        name = column[-1] if isinstance(column, tuple) else column
        if name in decimals:
            result[column] = result[column].round(decimals[name])
    if "统计口径" in result.attrs:
        result.attrs["统计口径"] = {**result.attrs["统计口径"], "展示舍入": True}
    return result
