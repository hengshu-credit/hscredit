"""为长章节标题设置目录短名，沿用 Sphinx 生成的正文和锚点。"""

import re

from docutils import nodes

SHORT_TITLES = {
    "showcase": {
        "为什么选 hscredit？": "四类风控报告",
        "核心功能盘点": "模块概览",
        "数据加载、标签与训练集划分": "加载与样本划分",
        "WOE、目标编码与 Pipeline": "编码与 Pipeline",
        "分箱摘要、客群比较与二维分箱表": "分箱分析",
        "分箱图与跨期趋势": "分箱可视化",
        "手工拆分与多数据集策略报告": "决策树规则报告",
        "常用图表": "诊断图与特征报告",
        "auto_feature_analysis：多标签特征报告": "多标签特征报告",
        "写入数据、多图、格式与透视表": "图表与透视表",
        "DataFrame 与 Series 的并行 apply": "并行 apply",
        "在 Jupyter 中重载模块": "Jupyter 模块重载",
    },
    "articles/tuning-guide": {
        "先选对入口": "调参入口",
        "已有模型：用 .tune() 获取最佳模型": "模型 tune()",
        "独立调参器：理解 fit 的返回值": "独立调参器",
        "搜索空间：先学三种声明，再学条件空间": "搜索空间",
        "自定义评估指标：普通函数只需加名称和方向": "自定义指标函数",
        "需要封装状态：继承 BaseMetric": "BaseMetric 指标类",
        "自定义 loss 训练，固定 metric 选择模型": "训练损失与选模指标",
        "从零实现一个自定义训练 loss": "自定义训练损失",
        "金额加权：区分训练权重和评估权重": "训练与评估权重",
        "sklearn Pipeline：预处理在每折内部拟合": "折内预处理",
        "多目标：保留所有不能同时被超越的候选": "多目标与 Pareto 前沿",
        "时间交叉验证：按放款时间模拟向未来预测": "时间交叉验证",
        "结果复盘、OOF 和复训": "OOF 与复训",
        "保存与恢复：完整对象、SQLite 和推理模型": "保存与恢复",
        "本次整理后的使用约定": "调参约定",
    },
    "articles/manual_tree_rule_mining": {
        "数据准备：统一字段、保留业务口径": "样本与字段",
        "数据驱动基线：先让数据说话": "基线决策树",
        "批量候选规则：把候选池先铺开": "候选规则",
        "人工决策树：把经验装回树里": "手工拆分",
        "把规则树画给业务看": "规则树可视化",
        "剪枝与节点删除：给策略做减法": "剪枝与删除节点",
        "多口径验证：同一棵树，换着角度盘问": "多口径验证",
        "从叶子到规则：挑出真正能上线的那几条": "叶节点规则筛选",
        "规则集落地：让一条规则变成能上线的策略": "规则集应用",
        "复用既有模型：让老树也能被人工调": "复用已有树模型",
        "一棵规则树到底值不值得上线？": "上线评估",
        "一条完整的策略工作流": "策略工作流",
    },
    "articles/losses-guide": {
        "先区分三个用途": "损失与指标用途",
        "准备训练集、验证集和测试集": "数据集划分",
        "每种损失对应什么指标": "损失与指标对照",
        "HSCredit 模型：最少配置的用法": "模型配置",
        "CV、GridSearchCV、ModelTuner 与 Optuna": "交叉验证与调参",
        "样本权重、放款金额和样本参数对齐": "样本权重对齐",
        "自定义指标、自定义损失与其他框架": "自定义损失与指标",
    },
    "articles/model-workflow": {
        "常用路径：单模型、单次验证": "单模型验证",
        "结果核对、训练过程与金融指标报告": "训练结果与报告",
        "保留每折内容、继续搜索、复用最佳模型": "保留折结果与续搜",
        "两种调参入口都能查看完整搜索过程": "查看搜索过程",
    },
}


def shorten_section_links(app, doctree):
    doc = app.env.docname
    if doc not in SHORT_TITLES:
        return
    for link in app.env.tocs[doc].findall(nodes.reference):
        if not link.get("anchorname"):
            continue
        title = link.astext()
        topic = re.sub(r"^\d+\.\s+", "", title)
        short = SHORT_TITLES[doc].get(topic, topic)
        if title != short:
            link.children.clear()
            link += nodes.Text(short)


def setup(app):
    # 在 Sphinx 收集完本页目录后处理显示文字，保留 refuri 和 anchorname。
    app.connect("doctree-read", shorten_section_links, priority=600)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
