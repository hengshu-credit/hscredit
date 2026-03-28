# hscredit Examples

完整示例目录，全部基于真实信贷数据 `hscredit.xlsx`（12448条样本，83个特征，目标变量FPD15）。

## 数据说明

| 字段类型 | 字段 | 说明 |
|----------|------|------|
| 目标变量 | `FPD15` | 首期逾期15天坏样本标记 |
| 目标变量 | `SFPD15` | 首两期逾期15天坏样本标记 |
| 逾期天数 | `MOB1` | 平滑FPD15 |
| 逾期天数 | `MOB1` | 平滑FPD15 |
| 时间字段 | `loan_date` | 贷款日期 |
| 外部评分 | `bj_qy24` / `mobtech80` / `bairong_v1` / `xiaoniu_c3` / `dxm_v6pro` / `bcf_fpv1` | 第三方征信/风控评分 |
| 行为特征 | `lender_count_*` / `loan_count_*` / `overdue_*` 等 | 网贷行为、逾期记录、申请查询 |

## 快速开始

```bash
pip install hscredit
jupyter notebook examples/
```

## 示例目录

| 文件 | 主题 | 主要内容 |
|------|------|----------|
| [01_binning.ipynb](01_binning.ipynb) | 分箱 | 13种分箱算法对比、单调约束(外部评分)、批量分箱、自定义切点、WOE转换 |
| [02_encoders.ipynb](02_encoders.ipynb) | 编码 | WOE/Target/Count/Quantile 编码器、Pipeline集成、各编码IV对比 |
| [03_selectors.ipynb](03_selectors.ipynb) | 特征筛选 | 缺失率/方差/IV/相关性/PSI/特征重要性筛选、CompositeFeatureSelector |
| [04_models.ipynb](04_models.ipynb) | 建模 | LogisticRegression(统计量)、ScoreCard、损失函数、概率转评分、系数图 |
| [05_eda.ipynb](05_eda.ipynb) | EDA | 数据概览、缺失分析、坏率趋势、批量IV、WOE、相关性、VIF、PSI、Vintage |
| [06_metrics.ipynb](06_metrics.ipynb) | 指标 | KS/AUC/Gini/IV/PSI/CSI/Lift@N、KS分箱表、规则lift分析 |
| [07_viz.ipynb](07_viz.ipynb) | 可视化 | bin_plot/ks_plot/corr_plot/roc/pr/lift/score_plots/阈值分析/特征趋势 |
| [08_stability.ipynb](08_stability.ipynb) | 稳定性 | 特征PSI、时序PSI追踪、评分PSI、ScoreDriftDetector、ModelCalibrator |
| [09_rules.ipynb](09_rules.ipynb) | 规则引擎 | 业务规则创建评估、ruleset_report、表达式优化、单/多特征规则挖掘 |
| [10_complete_workflow.ipynb](10_complete_workflow.ipynb) | 完整流程 | EDA→筛选→分箱→WOE→LR→评分卡→策略分析→稳定性监控 全链路 |
