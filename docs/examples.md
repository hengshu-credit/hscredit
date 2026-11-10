# 独立 Notebook 教程

每节下方的完整代码块都包含自己的 import、数据加载和实际调用，可直接复制到新的 Notebook 运行，不依赖前文变量或其他 Notebook。数据使用 `hscredit_dataset(columns=HSCREDIT_EXAMPLE_COLUMNS)` 全量按列加载，不做行采样。固定字段与选择口径见[演示字段集](examples-dataset.md)。原有其他 Notebook 和报告继续保留。

| Notebook | 内容 |
| --- | --- |
| {download}`01_binning <../examples/01_binning.ipynb>` | 数值与类别分箱、多标签统计、全方法基准对比、分箱图、人工切点与二维分箱 |
| {download}`02_encoders <../examples/02_encoders.ipynb>` | 九种编码器的编码、逆编码能力、保存恢复与表达式特征衍生 |
| {download}`03_selectors <../examples/03_selectors.ipynb>` | 23 个独立筛选器、summary、指标图、顺序与交集组合、完整筛选报告 |
| {download}`04_models <../examples/04_models.ipynb>` | 分类与风险调整金额回归、训练/测试/OOT、逐模型调参、续训、校准、评分、报告、SHAP 与部署 |
| {download}`05_scorecard <../examples/05_scorecard.ipynb>` | 分箱评分卡、超参数搜索、取整评分、报告、JSON 与 Python/SQL/Java 部署代码 |
| {download}`06_net <../examples/06_net.ipynb>` | 五种表格网络的分类/回归、预训练、微调、优化器、损失、调参、报告与推理制品 |
| {download}`07_rules <../examples/07_rules.ipynb>` | 规则运算、分类器、规则流程、多标签效果、策略置换、报告与规则制品 |
| {download}`08_rule_mining <../examples/08_rule_mining.ipynb>` | 单/多特征与多标签挖掘、人工二叉树、多叉树、节点调整、三数据集报告 |
| {download}`09_excel <../examples/09_excel.ipynb>` | 格式、多级表头、图片、原生透视表、已有工作簿追加与另存 |

## 运行

在已安装 `hscredit` 及对应可选依赖的 Python 内核中选择“重启内核并运行全部”。开发环境可使用 `pip install -e ".[all]"`，也可按[安装指南](installation.md)选择依赖组。TabPFN 的微调与 X-only 预训练需要 Python 3.10+ 和 `tabpfn>=9.2`；首次使用官方权重可能需要联网。

从 `examples` 目录启动，代码以 `import sys` 和 `sys.path.append('../')` 开始。每个段落单独导入所需接口；不设置环境变量、自动寻找仓库或定义演示包装函数。写文件的段落使用 `mkdtemp` 创建自己的输出目录，并显示位置；需要保留到指定位置时直接修改该段的 `output`。

也可以使用已有执行器验证单本：

```bash
python scripts/validate_examples.py --pattern 01_binning.ipynb --timeout 600
```

该执行器不回写源 Notebook；发布演示时应在正确内核下保存执行后的表格与图表。

## 共用口径

- 分类主标签为 `MOB6 > 30`；多标签报告展示各自明确的 MOB 与阈值。
- 回归标签为 `loan_amt`，当 `MOB6 > 60` 时取 `-loan_amt`。金额、MOB、标签和客户编号不进入模型特征。
- OOT 使用最新自然月，开发集排除 OOT 客户，再按客户随机拆分 train/test；调参验证集从 train 内部划出。
- 所有段落保留完整订单数据。用树数、训练轮数、搜索次数和必要的字段选择控制运行量；这些参数直接写在相应模型构造和训练调用中，短训练结果不代表充分收敛后的性能。
- TabPFN 两个独立段落同样保留全量输入，不自动缩样。全量上下文、预训练和微调需要充足内存与合适的 GPU；当前 32 GiB CPU 环境未执行这两段，不能将此前小样本验证当作全量验证。
- 经典与 Boosting 的现有 hscredit 封装使用分类后端。回归章节使用对应原生回归器，交由 hscredit 调参和生成报告；逻辑回归对应位置另列线性回归基线。
- 逆编码按编码器能力展示：OneHot、Ordinal 可还原已知类别；Cardinality 的合并类别不可恢复；其他有损编码器明确返回不支持逆编码。
- 评分卡核对 Python 部署、JSON 规则与完整模型的分数。SQL/Java 代码会生成，但各目标系统仍需在其执行环境中验收。

## 报告参数与合并导出

[公开 API Notebook](../examples/29_api_reference.ipynb) 顺序演示四类报告的文件导出，并把模型、特征、策略和树报告写入同一 writer。`save` 接收文件路径或 `ExcelWriter`，`sheet` 接收名称或 openpyxl `Worksheet`，最后统一保存。该 Notebook 使用本地 `hscredit_yyp.xlsx`，输出默认写入系统临时目录，可用 `HSCREDIT_EXAMPLE_OUTPUT` 指定位置。

详细报告章节见 [11_report.ipynb](../examples/11_report.ipynb)，树挖掘与多口径合并见 [26_multiway_tree.ipynb](../examples/26_multiway_tree.ipynb)。这些顺序教程应从头运行，与上表中逐节自包含的示例区分。
