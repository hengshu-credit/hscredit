# hscredit 示例代码说明

## 目录结构

examples/ 目录按功能模块分类组织，每个子目录对应一个核心功能：

### tutorials/
教程和项目概述文档，适合新手入门
- `01_project_overview.ipynb`: 项目整体介绍

### binning/
分箱相关示例（编号 02-06）
- `02_binning_separate_methods.ipynb`: 分箱方法的独立使用
- `03_optimal_binning_unified.ipynb`: 统一的最优分箱接口
- `04_feature_bin_stats.ipynb`: 特征分箱统计
- `05_categorical_binning.ipynb`: 类别型特征分箱
- `06_bin_table_display.ipynb`: 分箱表格展示

### encoding/
编码相关示例（编号 07-08）
- `07_encoders.ipynb`: 编码器演示
- `08_gbm_encoder.ipynb`: GBM编码器教程

### feature_selection/
特征选择示例（编号 09）
- `09_feature_selectors.ipynb`: 特征筛选器演示

### modeling/
建模相关示例（编号 10-11）
- `10_logistic_regression.ipynb`: 逻辑回归演示
- `11_scorecard.ipynb`: 评分卡教程

### reports/
报告生成示例（编号 12-13）
- `12_excel_writer.ipynb`: Excel写入器使用
- `13_feature_analysis_report.ipynb`: 自动特征分析报告

### strategy_analysis/
策略分析示例（编号 14-15）
- `14_rule_usage.ipynb`: 规则使用演示
- `15_swap_analysis.ipynb`: Swap分析演示

### utils/
工具和数据文件
- `hscredit.xlsx`: 示例数据文件
- `load_data.py`: 数据加载工具
- `basic_usage.py`: 基本用法示例
- `plot_weights_demo.md`: 权重绘图说明

## 文件命名规范

所有示例 notebook 统一采用以下命名规范：
- **编号_功能名称.ipynb**
- 编号从 01 开始，按功能顺序递增
- 名称使用下划线分隔，简洁明了
- 全部使用小写字母

示例：
- `01_project_overview.ipynb`
- `02_binning_separate_methods.ipynb`
- `03_optimal_binning_unified.ipynb`

## 学习路径建议

1. **新手入门**: 从 `tutorials/01_project_overview.ipynb` 开始
2. **分箱功能**: 学习 `binning/` 目录中的示例（02-06）
3. **特征工程**: 依次学习 `encoding/`（07-08）和 `feature_selection/`（09）
4. **模型构建**: 研究 `modeling/` 目录中的教程（10-11）
5. **报告生成**: 查看 `reports/` 中的示例（12-13）
6. **策略分析**: 了解 `strategy_analysis/` 中的方法（14-15）

## 数据文件说明

所有示例使用的数据文件统一放在 `utils/` 目录下：
- `hscredit.xlsx`: 主要示例数据文件
- 所有 notebook 中使用相对路径引用此文件：`../utils/hscredit.xlsx`

## 先验知识

hscredit是一个完整的金融信贷风险建模工具包，支持评分卡建模、策略分析、规则挖掘等功能。项目从 scorecardpipeline 迁移而来,旨在成为公司级开源项目,hscredit要去除对第三方风控库(toad、optbinning、scorecardpy)的依赖,大部分功能全部自主实现；hscredit继承scorecardpipeline的优势，即对用户可见的api简单化、优秀的可视化风格、快捷的excel写入器和快速产出样式精美的金融信贷场景用到的各类分析报告；

针对金融信贷场景中的评分，通常值域非[0, 1]的都是分数越大风险越低，而值域在[0, 1]的都是分数越小风险越低；

## 用户需求

对于建模人员，最大的需求是预处理方便，各种特征工程和筛选全部集成好很容易调用，模型调参有现成框架，直接运行慢慢找参数即可，模型loss可自定义，最终向老板或者客户汇报时能有精美的报告图和表。

对于策略人员，最大的需求是能快速从数据集中找到有效的特征，各种方式分析特征有效性，提取规则，分析规则有效性，快速产出对外的报告（汇报需要，精美的图文表甚至ppt），完整的分析报告有强逻辑性和连贯性。

## 更多信息

详细的目录结构说明请参考项目根目录下的 `DIRECTORY_STRUCTURE.md` 文件。
