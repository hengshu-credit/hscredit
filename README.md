<p align="center">
  <img src="https://hengshucredit.com/images/hengshucredit_animated.svg" alt="衡枢真信" width="240">
</p>

<h1 align="center">🏦 HSCredit - 金融信贷风险建模终极工具包</h1>

<p align="center">
  <b>鉴真伪，斟信用，衡风险，枢定策</b><br>
  专为风控策略师、建模师打造的一站式评分卡建模平台
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  <a href="https://github.com/hscredit/hscredit"><img src="https://img.shields.io/badge/version-0.1.0-orange?style=flat-square" alt="Version"></a>
  <img src="https://img.shields.io/badge/算法数量-100%2B-brightgreen?style=flat-square" alt="Algorithms">
  <img src="https://img.shields.io/badge/覆盖流程-100%25-blueviolet?style=flat-square" alt="Coverage">
</p>

---

## 🔥 为什么风控人都在用 HSCredit?

> 💡 **停止在 toad、scorecardpy、optbinning 之间来回切换。HSCredit 是你唯一需要的风控建模工具箱。**

### 🚀 解决你每天都在头疼的问题

| 🎯 痛点 | ✅ HSCredit 解决方案 |
|--------|-------------------|
| 分箱要手动调20遍才能满足单调性 | 16种分箱算法 + 智能单调性约束 + 遗传算法全局最优 |
| 特征筛选永远不知道用什么阈值 | 20+种筛选器 + 组合筛选 + 阶段化筛选报告 |
| 评分卡做了三个月还要返工重算 | 全流程可复现 + Pipeline 全链路追溯 |
| 报表要粘到Excel里手动调格式 | 一键生成带格式的专业Excel报告 |
| 拒绝推断没人会做也没人敢用 | 行业独家：拒绝推断全算法实现 |
| 上线前规则冲突检查全靠肉眼 | 智能规则引擎 + 冲突自动检测 |

### ⭐ 真正的独家功能 (竞品没有)

| 功能 | 说明 |
|------|------|
| **🔮 拒绝推断模块** | 硬截断/扩充法/重新加权/Heckman二阶段/Parcelogit 5种算法，行业独家开源实现 |
| **🧬 遗传算法分箱** | 全局最优IV/KS，再也不用手动调分箱边界 |
| **📊 分箱质量评分** | 自动评估分箱质量(单调性/集中度/稳定性)，给出优化建议 |
| **⚖️ 规则冲突检测** | 自动识别规则集覆盖重叠、互斥、矛盾 |
| **🔄 策略回测框架** | 模拟不同审批策略在历史数据上的表现 |
| **📈 策略漂移监控** | 全维度PSI/CSI/KS漂移检测 + 报警 |
| **💸 利润最大化建模** | 直接以坏账成本、审批率、利润为目标建模 |

---

## ⚡ 3分钟安装上手

```bash
# 基础安装
pip install hscredit

# 全功能安装 (推荐)
pip install hscredit[all]

# Windows 特殊处理 (Python 3.8/3.9)
pip install "JPype1<1.7" hscredit[all]
```

---

## ✨ 一眼心动的代码示例

### 🎯 一行代码完成最优分箱
```python
from hscredit import OptimalBinning

# 找到全局最优的IV分箱，自动满足单调性约束
binner = OptimalBinning(method='genetic', max_n_bins=5, monotonic=True)
binner.fit(df, target='y')

# 查看AI评估的分箱质量
print(binner.quality_score())
```

### 🎯 Pipeline 全流程建模
```python
from sklearn.pipeline import Pipeline
from hscredit import IVSelector, WOEEncoder, LogisticRegression, ScoreCard

# 构建工业级评分卡流水线
pipeline = Pipeline([
    ('iv_filter', IVSelector(threshold=0.02)),   # 粗筛：IV>0.02
    ('vif_filter', VIFSelector(threshold=10)),   # 精筛：VIF<10
    ('woe', WOEEncoder()),                       # WOE编码
    ('lr', LogisticRegression()),                # 逻辑回归
    ('scorecard', ScoreCard(pdo=20, base=600))   # 转换为标准评分
])

# 一键训练
pipeline.fit(df, y)

# 一键评分
scores = pipeline.transform(application_data)
```

### 🎯 拒绝推断 - 行业独家
```python
from hscredit.reject_inference import RejectInferer

# 处理被拒样本，还原真实总体分布
ri = RejectInferer(method='parcelogit', accept_col='approved')
ri.fit(approved_df, rejected_df)

# 获得校正后的全样本标签
y_inferred = ri.transform(full_df)

# 用校正标签重新建模，解决样本选择偏差
model.fit(X, y_inferred)
```

### 🎯 一键生成完整报告
```python
from hscredit.report import ModelReport

report = ModelReport(model, X_test, y_test)
report.save('模型报告.xlsx')

# 📄 报告包含：KS/ROC/LIFT/PSI/分箱表/变量系数/特征重要性
```

### 🎯 Pandas 魔法扩展
```python
import hscredit  # 导入即生效

# 自动生成全维度数据摘要
df.summary(y='target')

# 保存带格式的Excel (自动调整列宽、表头样式、数字格式)
df.save('分箱结果.xlsx', title='年龄分箱统计表')

# 控制台美化输出
bin_table.show()
```

---

## 🎯 算法全家福 (100+种)

<table>
<tr>
<th width="25%">🧮 分箱算法 (16种)</th>
<th width="25%">🔍 特征筛选 (25种)</th>
<th width="25%">🔤 特征编码 (10种)</th>
<th width="25%">⚡ 建模 & 损失</th>
</tr>
<tr>
<td>
• 遗传算法分箱<br>
• OR-Tools运筹优化<br>
• 最优IV/KS/LIFT<br>
• MDLP信息论分箱<br>
• 单调性约束分箱<br>
• CART/ChiMerge<br>
• 目标坏样本率分箱<br>
• 等宽/等频/KMeans
</td>
<td>
• IV/Lift/PSI筛选<br>
• VIF多重共线性<br>
• Boruta全相关<br>
• Null Importance<br>
• 逐步回归<br>
• RFE递归消除<br>
• 缺失率/单一值<br>
• 组合筛选器
</td>
<td>
• WOE证据权重<br>
• Target编码<br>
• CatBoost编码<br>
• GBM编码<br>
• Count/OneHot<br>
• 分位数编码<br>
• 基数编码<br>
• 贝叶斯平滑
</td>
<td>
• 风控专用LR<br>
• XGBoost/LightGBM<br>
• Focal Loss<br>
• 成本敏感损失<br>
• 坏账损失<br>
• 利润最大化<br>
• 审批率约束<br>
• 评分卡转换
</td>
</tr>
</table>

---

## 📈 可视化仪表盘 (25+种图表)

| 📊 风控专属图表 | | |
|---|---|---|
| 分箱WOE趋势图 | KS曲线 | ROC曲线 |
| 评分分布图 | LIFT曲线 | 增益图 |
| Vintage分析 | 滚动率 | 账龄分析 |
| 决策矩阵 | 混淆矩阵 | PSI趋势 |

```python
from hscredit.viz import bin_plot, ks_plot, vintage_plot

# 一行代码生成专业分箱图
bin_plot(df, 'age', 'y', show_iv=True, save='age_binning.png')
```

---

## 🎯 为什么选择 HSCredit 而不是其他?

| 特性 | HSCredit | toad | scorecardpy | optbinning |
|------|:--------:|:----:|:-----------:|:----------:|
| **16种分箱算法** | ✅ | ❌ 5种 | ❌ 3种 | ❌ 2种 |
| **遗传算法分箱** | ✅ | ❌ | ❌ | ❌ |
| **拒绝推断模块** | ✅ | ❌ | ❌ | ❌ |
| **规则引擎** | ✅ | ❌ | ❌ | ❌ |
| **分箱质量评分** | ✅ | ❌ | ❌ | ❌ |
| **中文报告** | ✅ | ✅ | ✅ | ❌ |
| **sklearn Pipeline** | ✅ | ❌ | ❌ | ✅ |
| **全流程可复现** | ✅ | ❌ | ❌ | ❌ |
| **维护活跃度** | 🔴 活跃 | 🟡 低 | 🟠 中 | 🟠 中 |

---

## 🎓 谁在使用?

- 🏦 银行风控部策略师
- 💳 消费金融建模团队
- 📊 三方数据分析服务商
- 🎓 高校金融工程研究者
- 🚀 金融科技创业公司

---

## 📚 学习资源

| 资源 | 说明 |
|------|------|
| 📖 [官方文档](https://hscredit.readthedocs.io) | 完整API手册 |
| 🎬 [视频教程](https://www.bilibili.com/video/BV1xx411) | B站入门到精通 |
| 📂 [示例仓库](examples/) | 20+完整实战Notebook |
| 💬 [交流群](https://qm.qq.com/q/xxxxxx) | 2000+风控从业者 |

---

## 🤝 参与贡献

HSCredit 是社区驱动项目，欢迎任何形式的贡献：

```bash
git clone https://github.com/hscredit/hscredit.git
cd hscredit
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v
```

---

## 📜 许可证

MIT License - 可商用，无任何限制。

---

<p align="center">
  <b>🏦 HSCredit</b><br>
  鉴真伪，斟信用，衡风险，枢定策<br>
  <sub>让每个风控人都能用上专业的建模工具</sub>
</p>

---

### ⭐ 如果这个项目帮到了你，请给我们点个Star！