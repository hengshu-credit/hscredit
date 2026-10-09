"""生成 README 筛选、规则、策略和手工树的原生完整结果。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA_CODE = '''
from pathlib import Path

import hscredit
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split

strategy_dir = Path("docs/assets/readme/strategy")
strategy_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_excel("examples/hscredit_yyp.xlsx")
features = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "手机号近一个月非银多头机构数"]
X = df[features]
y = df["FPD"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42,
)
train_df = df.loc[X_train.index]
test_df = df.loc[X_test.index]
'''

SELECTION_CODE = '''
from hscredit.core.selectors import (
    CompositeFeatureSelector, NullSelector, ModeSelector,
    IVSelector, CorrSelector, collect_selection_report,
)
from hscredit.excel import ExcelWriter

selector = CompositeFeatureSelector(
    [
        ("缺失率", NullSelector(threshold=0.95)),
        ("集中度", ModeSelector(threshold=0.98)),
        ("区分度", IVSelector(threshold=0.02)),
        ("相关性", CorrSelector(threshold=0.85)),
    ],
    strategy="sequential",
    n_jobs=1,
).fit(X_train, y_train)

selection_details = selector.get_selection_report_df()
selection_report = collect_selection_report(selector)
display(selection_details)
display(selection_report.summary)
display(selection_report.metrics)
display(selection_report.history)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selection_paths = selection_report.save(strategy_dir / "组合筛选完整报告", overwrite=True)
'''

SELECTION_STYLE_CODE = '''
# 读取完整原生工作簿，将显示样式另存为展示工作簿。
selection_file = selection_paths["路径"]["xlsx"]
selection_display_file = strategy_dir / "组合筛选展示.xlsx"
with ExcelWriter(style_excel=selection_file) as style_writer:
    for sheet_name, column_name, number_format in (
        ("筛选汇总", "保留率", "0.00%"),
        ("特征决策", "指标值", "0.0000"),
    ):
        worksheet = style_writer.get_sheet_by_name(sheet_name)
        column = next(cell.column_letter for cell in worksheet[1]
                      if cell.value == column_name)
        start, end = f"{column}2", f"{column}{worksheet.max_row}"
        style_writer.set_number_format(worksheet, f"{start}:{end}", number_format)
        style_writer.add_conditional_formatting(worksheet, start, end)
        style_writer.adjust_columns_width(worksheet)
    style_writer.save(selection_display_file)
'''

RULE_CODE = '''
from hscredit.core.rules import Rule
from hscredit.report import ruleset_analysis

risk_score = Rule("衡枢鉴真分老客版 >= 0.15", name="风险分偏高")
high_multi = Rule("近六个月非银多头机构数 >= 70", name="多头偏高")
phone_multi = Rule("手机号近一个月非银多头机构数 >= 28", name="近期多头偏高")
whitelist = Rule("衡枢鉴真分老客版 < 0.03", name="低风险白名单")
prior_rule = Rule("商品类别 == '礼包'", name="存量商品规则")
combined_rule = ((risk_score & high_multi) | phone_multi) & ~whitelist

rule_report = combined_rule.report(
    df, target="FPD", overdue=["MOB1"], dpds=[7, 3, 0],
    prior_rules=prior_rule, amount="放款金额", margins=True, n_jobs=1,
)
display(rule_report)
rule_report.save(
    strategy_dir / "组合规则完整评估.xlsx", auto_width=True,
    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率",
                  "坏账改善", "准确率", "精确率", "召回率", "F1分数"],
    condition_cols=["LIFT值"],
)

ruleset_report = ruleset_analysis(
    df, rules=[risk_score, high_multi, phone_multi], target="FPD",
    overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额", n_jobs=1,
)
display(ruleset_report)
ruleset_report.save(
    strategy_dir / "规则集完整评估.xlsx", auto_width=True,
    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率",
                  "坏账改善", "准确率", "精确率", "召回率", "F1分数"],
    condition_cols=["LIFT值"],
)
'''

SWAP_CODE = '''
from hscredit.report import rule_swap_analysis

swap = rule_swap_analysis(
    data=test_df,
    score="衡枢鉴真分老客版",
    rules_base=[Rule("衡枢鉴真分老客版 >= 0.25", name="生产基础拒绝")],
    rules_out=[high_multi, phone_multi],
    rules_in=[Rule("近六个月非银多头机构数 < 50", name="低多头置入")],
    reference_data=train_df,
    overdue=["MOB1"], dpds=[7, 3, 0],
    amount="放款金额", sample_survival_rate=0.70,
    out_in_uplift=2.0, max_n_bins=4, n_jobs=1,
)
display(swap["swap_pipeline"])
display(swap["swap_result"])

with ExcelWriter() as writer:
    swap["swap_pipeline"].save(
        writer, sheet_name="完整置换流水线", auto_width=True,
        percent_cols=["样本占比", "金额占比", "通过率(相对值)",
                      "好样本占比", "坏样本占比", "坏样本率",
                      "原始坏样本率", "调整后坏样本率", "坏账改善",
                      "原始坏样本率(金额)", "调整后坏样本率(金额)"],
        condition_cols=["LIFT值"],
    )
    swap["swap_result"].save(
        writer, sheet_name="完整置换对比", auto_width=True,
        percent_cols=["相对变化"], condition_cols=["相对变化"],
    )
    writer.save(strategy_dir / "规则置换完整分析.xlsx")
'''

SWAP_OUT_CODE = '''
from hscredit.report import swap_out_report

strategy_writer = swap_out_report(
    df,
    rules=[risk_score, high_multi, phone_multi],
    background="比较衡枢风险分、近六个月多头和手机号近期多头规则的覆盖与风险表现。",
    target="FPD", overdue=["MOB1"], dpds=[7, 3, 0],
    amount="放款金额", date_col="放款时间", freq="M",
    features=features, methods="quantile",
    bin_params={"max_n_bins": 4},
    current_pass_rate=0.70, n_jobs=1,
    save=str(strategy_dir / "策略迭代完整报告.xlsx"),
)
display(strategy_writer)
'''

TREE_AUTO_CODE = '''
from hscredit.report.mining import ManualTreeExtractor
from hscredit.core.viz import plot_tree_matplotlib

manual_tree = ManualTreeExtractor(
    target="FPD", features=features,
    max_depth=2, min_samples_leaf=40, random_state=42, n_jobs=1,
)
manual_tree.fit(train_df)
automatic_figure = plot_tree_matplotlib(
    manual_tree, save=str(strategy_dir / "manual-tree-auto.png"),
)
display(automatic_figure)
automatic_rules = manual_tree.get_rule_table(
    test_df, overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额",
)
display(automatic_rules)
automatic_rules.save(
    strategy_dir / "自动决策树完整规则.xlsx", auto_width=True,
    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率",
                  "坏账改善", "准确率", "精确率", "召回率", "F1分数"],
    condition_cols=["LIFT值"],
)
'''

TREE_MANUAL_CODE = '''
manual_tree.manual_split(train_df, feature="衡枢鉴真分老客版", threshold=0.10, node=0)
manual_tree.manual_split(
    train_df, feature="近六个月非银多头机构数", threshold=55, node=1,
)
manual_figure = plot_tree_matplotlib(
    manual_tree, save=str(strategy_dir / "manual-tree-intervention.png"),
)
display(manual_figure)
manual_rules = manual_tree.get_rule_table(
    test_df, overdue=["MOB1"], dpds=[7, 3, 0], amount="放款金额",
)
display(manual_rules)
manual_rules.save(
    strategy_dir / "手工决策树完整规则.xlsx", auto_width=True,
    percent_cols=["样本占比", "好样本占比", "坏样本占比", "坏样本率",
                  "坏账改善", "准确率", "精确率", "召回率", "F1分数"],
    condition_cols=["LIFT值"],
)
'''


def _code(source):
    return "```python\n" + textwrap.dedent(source).strip() + "\n```\n"


def _native_table(frame, stem, label, directory, metadata):
    """完整原生序列化，不筛行列、不重命名、不另设显示精度。"""
    html = frame.to_html()
    path = directory / f"{stem}.html"
    path.write_text(html, encoding="utf-8")
    metadata[stem] = {
        "返回类型": type(frame).__name__,
        "行数": frame.shape[0],
        "列数": frame.shape[1],
        "列名": [list(column) if isinstance(column, tuple) else column for column in frame.columns],
        "HTML_SHA256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
    }
    return (
        f"<details>\n<summary>{label}（{frame.shape[0]} 行 × {frame.shape[1]} 列，完整返回）</summary>\n\n"
        + html
        + f"\n\n</details>\n\n[完整原生 HTML](docs/assets/readme/strategy/{stem}.html)\n"
    )


def main():
    os.chdir(ROOT)
    directory = ROOT / "docs/assets/readme/strategy"
    directory.mkdir(parents=True, exist_ok=True)
    namespace = {}
    metadata = {}
    for label, source in (
        ("数据准备", DATA_CODE),
        ("组合筛选", SELECTION_CODE),
        ("筛选工作簿原生样式", SELECTION_STYLE_CODE),
        ("规则评估", RULE_CODE),
        ("置入置出", SWAP_CODE),
        ("策略迭代报告", SWAP_OUT_CODE),
        ("自动决策树", TREE_AUTO_CODE),
        ("手工分裂", TREE_MANUAL_CODE),
    ):
        print(f"执行：{label}", flush=True)
        exec(compile(textwrap.dedent(source), f"strategy.py::{label}", "exec"), namespace)

    parts = [
        "### 特征筛选、规则评估与人工决策树\n",
        "以下示例共用相同的三项特征和 `FPD` 标签；建树与筛选只拟合训练集，"
        "规则评估保留多 DPD 及金额口径。数据准备代码如下，后续代码按顺序运行。\n",
        "<details>\n<summary>本节公共数据准备</summary>\n\n" + _code(DATA_CODE) + "\n</details>\n",
        "#### 多个筛选器组合与完整过程报告\n",
        "依次应用缺失率、集中度、IV 和相关性筛选。`get_selection_report_df()` 返回包含各子步骤的完整决策，"
        "`collect_selection_report()` 直接收集已拟合步骤的汇总、逐特征决策、指标明细和迭代事件；"
        "下面保留每张表的全部行列，未将多阶段结果改写为手工摘要。\n",
        _code(SELECTION_CODE + "\n" + SELECTION_STYLE_CODE),
    ]
    selection_report = namespace["selection_report"]
    selection_xlsx = Path(namespace["selection_paths"]["路径"]["xlsx"]).relative_to(ROOT).as_posix()
    for frame, stem, label in (
        (namespace["selection_details"], "selection-root", "组合器完整决策"),
        (selection_report.summary, "selection-summary", "完整阶段汇总"),
        (selection_report.metrics, "selection-metrics", "完整指标明细"),
        (selection_report.history, "selection-history", "完整迭代事件"),
    ):
        parts.append(_native_table(frame, stem, label, directory, metadata))
    parts.extend([
        "[下载带格式的完整展示工作簿](docs/assets/readme/strategy/组合筛选展示.xlsx) · "
        f"[下载组合筛选原生完整工作簿]({selection_xlsx})\n",
        "#### Rule 组合、先验规则与规则集分析\n",
        "通过 `&`、`|`、`~` 将衡枢风险分、近六个月多头、手机号近期多头和白名单组合为可执行规则，"
        "并用 `商品类别` 规则评估存量策略。"
        "`amount=\"放款金额\"` 使用金额口径；保留方法原始列名与多级表头，不将金额数值改写为人数。"
        "`ruleset_analysis` 展示逐规则与整体规则效果。\n",
        _code(RULE_CODE),
        _native_table(namespace["rule_report"], "rule-report", "Rule.report 完整结果", directory, metadata),
        _native_table(namespace["ruleset_report"], "ruleset-report", "ruleset_analysis 完整结果", directory, metadata),
        "[组合规则工作簿](docs/assets/readme/strategy/组合规则完整评估.xlsx) · "
        "[规则集工作簿](docs/assets/readme/strategy/规则集完整评估.xlsx)\n",
        "#### 规则置入与置出\n",
        "衡枢鉴真分老客版在本例按高值偏高风险使用，高值规则用于拒绝，低值规则用于白名单。"
        "置入风险按历史训练样本各分箱的真实坏样本率估计，不将高风险分当作高信用分。"
        "验证样本用于策略置换。生产基础拒绝、置出与置入按方法"
        "原生流程依次分析；`sample_survival_rate=0.70` 和 `out_in_uplift=2.0` 是本例的情景输入，"
        "后者上浮置入客群的预测风险。返回字典中的两张表均完整展示。\n",
        _code(SWAP_CODE),
        _native_table(namespace["swap"]["swap_pipeline"], "swap-pipeline", "swap_pipeline 完整结果", directory, metadata),
        _native_table(namespace["swap"]["swap_result"], "swap-result", "swap_result 完整结果", directory, metadata),
        "[规则置换完整工作簿](docs/assets/readme/strategy/规则置换完整分析.xlsx)\n",
        "#### 策略迭代报告：swap_out_report\n",
        "直接生成包含样本描述、相关性、变量分箱、业务影响、规则效果、金额口径和分月稳定性的 Excel。"
        "该方法原生返回 `ExcelWriter`，完整工作簿即报告产物。\n",
        _code(SWAP_OUT_CODE),
        "[下载策略迭代完整报告](docs/assets/readme/strategy/策略迭代完整报告.xlsx)\n",
        "#### 自动训练决策树，再手工指定分裂\n",
        "先在训练集自动拟合深度为 2 的树，并在验证集按 `MOB1` 的 7 / 3 / 0 天逾期阈值评估所有节点。"
        "下图由 `plot_tree_matplotlib` 直接保存，使用原生尺寸、配色和指标布局。\n",
        _code(TREE_AUTO_CODE),
        "![ManualTreeExtractor 自动训练原图](docs/assets/readme/strategy/manual-tree-auto.png)\n",
        _native_table(namespace["automatic_rules"], "manual-tree-auto-rules", "自动训练后的完整节点规则", directory, metadata),
        "再把根节点改为 `衡枢鉴真分老客版 <= 0.10`，把左节点改为 `近六个月非银多头机构数 <= 55`。"
        "`manual_split()` 修改可执行树结构，并重新生成规则效果表与原生树图；"
        "表格继续使用相同验证集和金额口径，保留方法返回的全部节点行。"
        "当前版本对人工创建节点的 GINI 使用 1.0000 占位，下面保留原图；"
        "判断节点风险时应读取原生坏样本率和规则评估表，不能把该占位值理解为重算的 GINI。\n",
        _code(TREE_MANUAL_CODE),
        "![ManualTreeExtractor 人工分裂原图](docs/assets/readme/strategy/manual-tree-intervention.png)\n",
        _native_table(namespace["manual_rules"], "manual-tree-intervention-rules", "人工分裂后的完整节点规则", directory, metadata),
        "[自动树完整规则工作簿](docs/assets/readme/strategy/自动决策树完整规则.xlsx) · "
        "[手工树完整规则工作簿](docs/assets/readme/strategy/手工决策树完整规则.xlsx)\n",
    ])
    section = ROOT / ".audit_tmp/readme-complete/sections/strategy.md"
    section.parent.mkdir(parents=True, exist_ok=True)
    section.write_text("\n".join(parts), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{section}", flush=True)


if __name__ == "__main__":
    main()
