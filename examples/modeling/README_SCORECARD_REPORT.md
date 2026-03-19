# 评分卡建模报告 - hscredit版

## 概述

本报告基于 `hscredit` 库实现了完整的评分卡建模流程，参考 `scorecardpipeline` 的 `scorecard_samples.ipynb` 样例，提供了从数据加载到模型报告生成的完整端到端流程。

## 核心特性

### 1. 完整的建模流程
- ✅ 数据加载与探索
- ✅ 特征工程（衍生特征生成）
- ✅ 特征筛选（缺失值、方差、单一值、相关性、IV值、逐步回归）
- ✅ 最优分箱（Chi2算法）
- ✅ 逻辑回归模型训练
- ✅ 评分卡生成与分数转换
- ✅ PMML模型导出

### 2. 模型评估与可视化
- ✅ KS曲线（训练集/测试集）
- ✅ 分数分布直方图
- ✅ 特征相关性热力图
- ✅ 特征分箱分析图
- ✅ KS、AUC等评估指标

### 3. Excel报告生成
- ✅ 汇总信息（模型基本信息、样本分布）
- ✅ 逻辑回归拟合结果（模型系数、特征重要性）
- ✅ 评分卡结果（刻度、分数表、分数统计）
- ✅ 模型评估可视化（KS曲线、分数分布、相关性）
- ✅ 特征分箱分析（分箱统计表+可视化）

## 使用方法

### 1. 运行notebook

```bash
# 在Jupyter Notebook中打开
jupyter notebook scorecard_modeling_report.ipynb

# 或使用Jupyter Lab
jupyter lab scorecard_modeling_report.ipynb
```

### 2. 输出文件

运行完成后，会在 `model_report/` 目录下生成以下文件：

```
model_report/
├── scorecard_report.xlsx      # Excel报告
├── scorecard.pmml            # PMML模型文件
├── ks_plot.png              # KS曲线
├── score_hist.png           # 分数分布图
├── correlation_heatmap.png  # 相关性热力图
└── bin_plots/              # 特征分箱图
    ├── feat_ratio_1_bin.png
    ├── feat_ratio_2_bin.png
    └── ...
```

## 核心代码示例

### 数据加载与预处理

```python
from hscredit.utils import init_setting
from hscredit.core.binning import OptimalBinning
from hscredit.core.selectors import (
    NullSelector, VarianceSelector, ModeSelector,
    CorrSelector, IVSelector, StepwiseSelector
)
from hscredit.core.models import LogisticRegression, ScoreCard
from hscredit.core.metrics import KS, AUC
from hscredit.core.viz import bin_plot, corr_plot, ks_plot, hist_plot
from hscredit.report import feature_bin_stats, ExcelWriter

# 初始化环境
init_setting(seed=42)

# 数据加载
df = pd.read_excel('data.xlsx')
df['target'] = (df['MOB1'] > 15).astype(int)

# 数据集划分
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
```

### 特征筛选

```python
# 1. 缺失值筛选
null_selector = NullSelector(threshold=0.3)
features = null_selector.fit_transform(train[feature_cols], train['target'])

# 2. 方差筛选
variance_selector = VarianceSelector(threshold=0.01)
features = variance_selector.fit_transform(train[features], train['target'])

# 3. 单一值筛选
mode_selector = ModeSelector(threshold=0.95)
features = mode_selector.fit_transform(train[features], train['target'])

# 4. 相关性筛选
corr_selector = CorrSelector(threshold=0.8)
features = corr_selector.fit_transform(train[features], train['target'])

# 5. IV值筛选
iv_selector = IVSelector(threshold=0.02)
features = iv_selector.fit_transform(train[features], train['target'])
```

### 最优分箱

```python
# 创建分箱器
binner = OptimalBinning(
    max_n_bins=5,
    min_bin_size=0.05,
    binning_type='chi2'
)

# 拟合并转换
binner.fit(train[features], train['target'])
train_binned = binner.transform(train[features])
test_binned = binner.transform(test[features])
```

### 逐步回归特征筛选

```python
# 创建逐步回归选择器
stepwise = StepwiseSelector(
    direction='forward',
    max_features=15,
    binner=binner
)

# 拟合并选择特征
stepwise.fit(train[features], train['target'])
final_features = stepwise.selected_features_
```

### 模型训练

```python
# 训练逻辑回归模型
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(train_binned[final_features], train['target'])

# 创建评分卡模型
scorecard = ScoreCard(
    binner=binner,
    lr_model=lr_model,
    pdo=50,
    base_score=600,
    base_odds=1/19
)

# 计算分数
train['score'] = scorecard.predict_score(train[final_features])
test['score'] = scorecard.predict_score(test[final_features])
```

### PMML导出

```python
# 导出PMML文件
scorecard.export_pmml(
    'model_report/scorecard.pmml',
    feature_names=final_features,
    target_name='target'
)
```

### Excel报告生成

```python
# 创建Excel写入器
writer = ExcelWriter()

# 获取或创建sheet
worksheet = writer.get_sheet_by_name('汇总信息')

# 插入标题
writer.insert_value2sheet(worksheet, (2, 2), value='模型基本信息', style='header')

# 插入DataFrame
writer.insert_df2sheet(worksheet, model_info_df, (3, 2))

# 插入图片
writer.insert_pic2sheet(worksheet, 'model_report/ks_plot.png', (10, 2), figsize=(800, 350))

# 保存文件
writer.save('model_report/scorecard_report.xlsx')
```

## 与scorecardpipeline的对应关系

| 功能模块 | scorecardpipeline | hscredit |
|---------|------------------|----------|
| 分箱 | `toad.transform.Combiner` | `hscredit.core.binning.OptimalBinning` |
| 特征筛选 | `sp.FeatureSelection` | `hscredit.core.selectors.*Selector` |
| 逻辑回归 | `sp.ITLubberLogisticRegression` | `hscredit.core.models.LogisticRegression` |
| 评分卡 | `toad.ScoreCard` | `hscredit.core.models.ScoreCard` |
| 特征分箱统计 | `sp.feature_bin_stats` | `hscredit.report.feature_bin_stats` |
| Excel报告 | `sp.ExcelWriter` | `hscredit.report.ExcelWriter` |
| 可视化 | `sp.bin_plot`, `sp.ks_plot` | `hscredit.core.viz.bin_plot`, `hscredit.core.viz.ks_plot` |

## 注意事项

1. **数据格式**：确保数据为DataFrame格式，标签列为0/1二分类
2. **分箱参数**：根据业务需求调整 `max_n_bins` 和 `min_bin_size`
3. **评分卡参数**：`pdo`、`base_score`、`base_odds` 根据业务需求设置
4. **PMML导出**：需要安装 `sklearn2pmml` 和 Java 环境
5. **报告生成**：确保 `model_report/` 目录有写入权限

## 依赖库

```txt
numpy
pandas
scikit-learn
openpyxl
matplotlib
seaborn
plotly  # 可选，用于交互式可视化
pypmml  # 可选，用于PMML预测验证
sklearn2pmml  # 可选，用于PMML导出
```

## 技术支持

如有问题，请参考：
- hscredit文档：`/hscredit/examples/`
- scorecardpipeline文档：`scorecardpipeline/examples/scorecard_samples.ipynb`

## 更新日志

### v1.0 (2026-03-18)
- 初始版本发布
- 实现完整的评分卡建模流程
- 支持Excel报告生成
- 支持PMML导出
