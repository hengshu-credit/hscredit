# hscredit 示例代码

本目录包含 hscredit 包的各种使用示例和教程，按功能模块分类组织。

## 快速导航

### 新手入门
- [01. 项目概述](tutorials/01_project_overview.ipynb) - 了解 hscredit 的整体架构和核心功能

### 核心功能

#### 02. 分箱 (Binning)
分箱是信贷风控建模的关键步骤，支持数值型和类别型特征的分箱。

- [02_binning_separate_methods.ipynb](binning/02_binning_separate_methods.ipynb) - 分箱方法独立使用
- [03_optimal_binning_unified.ipynb](binning/03_optimal_binning_unified.ipynb) - 最优分箱统一接口
- [04_feature_bin_stats.ipynb](binning/04_feature_bin_stats.ipynb) - 特征分箱统计
- [05_categorical_binning.ipynb](binning/05_categorical_binning.ipynb) - 类别型特征分箱
- [06_bin_table_display.ipynb](binning/06_bin_table_display.ipynb) - 分箱表格展示

#### 03. 特征编码 (Encoding)
支持多种编码方式，包括传统的编码方法和基于模型的编码。

- [07_encoders.ipynb](encoding/07_encoders.ipynb) - 编码器演示
- [08_gbm_encoder.ipynb](encoding/08_gbm_encoder.ipynb) - GBM编码器教程

#### 04. 特征选择 (Feature Selection)
提供多种特征筛选方法，包括相关性筛选、VIF筛选、IV筛选等。

- [09_feature_selectors.ipynb](feature_selection/09_feature_selectors.ipynb) - 特征筛选器演示

#### 05. 建模 (Modeling)
支持逻辑回归、评分卡等传统风控模型的构建。

- [10_logistic_regression.ipynb](modeling/10_logistic_regression.ipynb) - 逻辑回归演示
- [11_scorecard.ipynb](modeling/11_scorecard.ipynb) - 评分卡教程

#### 06. 报告生成 (Reports)
支持生成精美的Excel报告和自动化分析报告。

- [12_excel_writer.ipynb](reports/12_excel_writer.ipynb) - Excel写入器使用
- [13_feature_analysis_report.ipynb](reports/13_feature_analysis_report.ipynb) - 自动特征分析报告

#### 07. 策略分析 (Strategy Analysis)
支持规则挖掘、Swap分析等策略分析功能。

- [14_rule_usage.ipynb](strategy_analysis/14_rule_usage.ipynb) - 规则使用演示
- [15_swap_analysis.ipynb](strategy_analysis/15_swap_analysis.ipynb) - Swap分析演示

## 目录结构

```
examples/
├── tutorials/          # 教程和项目概述
├── binning/            # 分箱相关示例
├── encoding/           # 编码相关示例
├── feature_selection/  # 特征选择示例
├── modeling/           # 建模相关示例
├── reports/            # 报告生成示例
├── strategy_analysis/  # 策略分析示例
└── utils/              # 工具和数据文件
```

## 学习路径

### 建模人员路径
01. 项目概述 → 02. 分箱方法 → 03. 特征编码 → 04. 特征选择 → 05. 建模 → 06. 报告生成

### 策略人员路径
01. 项目概述 → 02. 分箱方法 → 04. 特征选择 → 07. 策略分析 → 06. 报告生成

## 运行示例

所有示例都是 Jupyter Notebook 格式，可以直接在 Jupyter 中打开运行。

```bash
# 安装 Jupyter（如果尚未安装）
pip install jupyter

# 启动 Jupyter
jupyter notebook

# 打开 examples/tutorials/01_project_overview.ipynb 开始学习
```

## 注意事项

1. 运行示例前请确保已安装所有依赖包（参见项目根目录的 `requirements.txt`）
2. 部分示例需要数据文件，数据文件位于 `utils/hscredit.xlsx`
3. 如需导入 hscredit 包，请确保 Python 路径设置正确
4. 所有 notebook 中的数据路径已统一为相对路径

## 更多信息

- [项目主文档](../README.md)
- [目录结构详细说明](../DIRECTORY_STRUCTURE.md)
- [API 文档](../docs/)
