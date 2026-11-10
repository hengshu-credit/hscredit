"""图表、HTML 表格与 Excel 共用的品牌颜色，不依赖绘图库。"""

PRIMARY_COLORS = ["#2639E9", "#F76E6C", "#FE7715"]

# 与品牌色系卡共用十种不透明实色。
SECONDARY_COLORS = ["#F76E6C", "#C00000", "#D4145A", "#FE7715", "#FF8BCE"]
ORDINARY_COLORS = ["#2639E9", "#D4145A", "#E52DD0", "#E52B88", "#EF4638", "#EC2D59"]
EXTENDED_COLORS = list(dict.fromkeys([*PRIMARY_COLORS, *SECONDARY_COLORS, *ORDINARY_COLORS]))

# 条件格式使用独立的浅蓝实色，品牌主题蓝与表头仍为 #2639E9。
CONDITIONAL_BLUE = "#A3ABF5"

# 指标色作为色阶终点或数据条实色，颜色区分指标类型。
METRIC_ACCENT_COLORS = {
    "share": "#BADC58",
    "hit_rate": PRIMARY_COLORS[1],  # 规则命中率使用正样本强调色。
    "good_share": CONDITIONAL_BLUE,
    "good_rate": CONDITIONAL_BLUE,
    "good_count": CONDITIONAL_BLUE,
    "risk": "#F76E6C",
    "bad_share": "#F76E6C",
    "bad_count": "#F76E6C",
    "lift": "#D4145A",
    "ks": "#E52DD0",
    "iv": "#E52B88",
    "auc": "#EF4638",
    "psi": "#FE7715",
    "missing": "#EC2D59",
    "risk_ratio": "#C00000",
    "improvement": "#F76E6C",
}
