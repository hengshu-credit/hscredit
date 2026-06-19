# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HSCredit（衡枢真信）是一个面向金融信贷场景的 Python 3.9+ 信用风险建模工具包，提供**评分卡建模全流程**覆盖：分箱、编码、特征筛选、建模、评估、可视化、规则挖掘与报告生成。所有输出（列名、报告、错误信息）均为**中文**。

## 验证数据集约定

修改代码后，使用 `examples/hscredit_yyp.xlsx`（真实放款数据）验证功能是否正常：

| 场景 | 验证参数 |
|------|----------|
| 常规评分卡建模 | `x` = 衡枢鉴真分老客版 字段，`y` = FPD 列 |
| 多特征输入 | `x` = [衡枢鉴真分老客版, 近六个月非银多头机构数, 青云24] |
| 多标签规则/逾期分析 | `overdue=['MOB1']`，`dpds=[7, 3, 0]` |
| 金额相关计算 | `amount='放款金额'` |
| 日期/账龄分析 | `date_col='放款时间'` |
| 类别特征分析 | 使用 `商品类别` 列 |

## 常用命令

```bash
# 安装（开发模式）
pip install -e ".[dev]"

# 运行所有测试
pytest tests/ -v --tb=short

# 运行特定测试模块
pytest tests/test_binning/ -v

# 运行单个测试函数
pytest tests/test_binning/test_binning.py::test_binning_basic -v

# 跳过慢速/集成测试
pytest tests/ -m "not slow and not integration"

# 测试覆盖率
pytest tests/ --cov=hscredit --cov-report=html --cov-report=term

# 格式化、lint、类型检查（Makefile）
make check

# 单独运行
black hscredit tests
flake8 hscredit tests
mypy hscredit --ignore-missing-imports

# Makefile 常用目标
make format    # black
make lint      # flake8
make type-check  # mypy
make coverage  # 覆盖率报告
make clean     # 清理 __pycache__ 等
make docs      # 构建 sphinx 文档
make jupyter   # 启动 Jupyter Notebook
make quickstart  # 安装 + 验证环境

# 直接运行 Jupyter
cd examples && jupyter notebook
# 或指定端口
jupyter notebook --port=8888
```

## 架构设计

### 统一基类继承体系（sklearn 兼容）

所有核心组件均继承自 sklearn `BaseEstimator` + `TransformerMixin`/`ClassifierMixin` + `ABC`，兼容 sklearn Pipeline：

- `BaseBinning` → [hscredit/core/binning/](hscredit/core/binning/) 中 17 种分箱算法 + `OptimalBinning2D` 二维交互分箱
- `BaseEncoder` → [hscredit/core/encoders/](hscredit/core/encoders/) 中 9 种编码器
- `BaseFeatureSelector` → [hscredit/core/selectors/](hscredit/core/selectors/) 中 22 种筛选器
- `BaseRiskModel` → [hscredit/core/models/](hscredit/core/models/) 中 boosting 和经典模型

### 双 API 调用风格

所有有监督组件支持两种调用方式：

```python
# sklearn 风格 — X 和 y 分别传入
binner.fit(X_train, y_train)

# scorecardpipeline 风格 — 通过 target 参数指定目标列名
binner = OptimalBinning(target='target', ...)
binner.fit(df)  # 自动从 df 中提取目标列

# 混合风格 — y 参数优先于 df 中的 target 列
binner.fit(df, y=ext_y)
```

### 工厂模式：OptimalBinning

`OptimalBinning` 是所有 17 种分箱方法的统一入口：

```python
best_method = OptimalBinning.auto_select_method(X, y, 'feature_name')
binner = OptimalBinning(method=best_method)
```

### 核心模块组织

```
hscredit/
├── core/
│   ├── binning/       # 17 种分箱算法 + BaseBinning + OptimalBinning 工厂
│   ├── encoders/      # 9 种编码器（WOE/Target/Count/OneHot/...）
│   ├── selectors/     # 22 种特征筛选器
│   ├── models/
│   │   ├── boosting/      # XGBoost / LightGBM / CatBoost / NGBoost
│   │   ├── classical/     # LogisticRegression / RandomForest 等
│   │   ├── scorecard/     # ScoreCard / ScoreTransformer / ScoreDrift
│   │   ├── losses/        # 14 种自定义损失函数 + 框架适配器
│   │   ├── evaluation/    # ModelReport / Calibration / Interpretability
│   │   ├── rules/         # RuleSet / RulesClassifier
│   │   └── tuning/        # Optuna 超参数调优
│   ├── metrics/       # 分类 / 稳定性 / 特征 / 金融指标
│   ├── viz/           # 50+ 种可视化图表
│   ├── eda/           # 数据探索（相关性/特征/账龄/群体稳定性/策略分析）
│   ├── rules/          # 规则引擎（Rule 表达式解析/优化）
│   ├── financial/      # 金融计算（FV/PV/PMT/NPER/IRR/NPV）
│   └── feature_engineering/  # NumExprDerive 表达式衍生
├── report/             # 报告生成器
│   └── mining/         # 规则挖掘器（SingleFeature/MultiFeature/MultiLabel/Tree）
├── excel/              # ExcelWriter 上下文管理器
└── utils/              # Pandas 扩展 / IO / 日志 / 随机种子
```

### 关键设计模式

**SelectionReportCollector**：聚合 Pipeline 中多个筛选器的报告，生成统一中文摘要。每个筛选器实现 `get_selection_report()` 返回含中文键的字典（`输入特征数`, `选中特征数`, `特征列表`, `筛选方法`）。

**Loss Adapter 模式**：`BaseLoss` 子类通过 `XGBoostLossAdapter`、`LightGBMLossAdapter` 等适配到各框架格式。

**Pandas 扩展**：`import hscredit` 自动注册 `df.summary()`、`df.save()`、`df.show()`：

```python
import hscredit
df.summary(y='target')           # 全维度数据摘要
df.save('结果.xlsx', title='标题')  # 带格式导出 Excel
bin_table.show()                # 格式化表格输出
```

**规则挖掘 Pipeline**：

```python
from hscredit.report import SingleFeatureRuleMiner, MultiFeatureRuleMiner, TreeRuleExtractor

miner = SingleFeatureRuleMiner(target='ISBAD', method='best_iv', max_n_bins=5)
rules = miner.get_top_rules(top_n=10, metric='lift')
```

**OverduePredictor**：按账龄加权预测逾期率：

```python
from hscredit.report import OverduePredictor
predictor = OverduePredictor(feature='score', target='IS_OVERDUE')
report = predictor.fit(train_df).get_report()
```

### 异常体系

所有自定义异常继承自 `HSCreditError`。使用具体类型：`ValidationError`、`InputValidationError`、`InputTypeError`、`FeatureNotFoundError`、`StateError`、`NotFittedError`、`DependencyError`、`SerializationError`。辅助函数 `raise_not_fitted()`、`raise_feature_not_found()`、`raise_missing_columns()` 可用。

### 可选依赖

XGBoost、LightGBM、CatBoost、PyTorch/TabNet、PMML、Optuna、SHAP 均为可选依赖：

```bash
pip install hscredit[xgboost]      # XGBoost
pip install hscredit[lightgbm]     # LightGBM
pip install hscredit[catboost]     # CatBoost
pip install hscredit[net]           # PyTorch, TabNet
pip install hscredit[tune]          # Optuna 超参调优
pip install hscredit[explain]      # SHAP
pip install hscredit[all]          # 所有可选依赖
```

## 代码开发规范

### 开发前必读

1. **修改前**：完整研究项目结构和相关功能的实现方式，所有改动必须与现有实现风格保持一致且逻辑连贯
2. **修改后**：使用 `examples/hscredit_yyp.xlsx` 验证所有模块功能点（模块导入、方法使用、输出内容、输出格式）不受本次改动影响
3. **分箱**：所有分箱使用 `hscredit.core.binning` 中的分箱器，所有分箱指标计算使用 `compute_bin_stats`
4. **规则**：所有规则使用 `hscredit.core.rules.rule` 中的 `Rule` 类实现，报告指标计算使用 `Rule.report`

### API 设计约定

| 维度 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `OptimalBinning`, `WOEEncoder` |
| 函数/方法 | snake_case | `fit_transform`, `get_bin_table` |
| 常量 | UPPER_SNAKE_CASE | `VALID_METHODS`, `_AVAILABLE` |
| 模块 docstring | 中文，描述模块职责 | `"""统一分箱接口."""` |
| 类 docstring | `**参数**` → `**属性**` → `**参考样例**` 三段式 | 见 `BaseBinning` |
| 输出 DataFrame 列名 | **中文** | `分箱`, `样本总数`, `坏样本率` |

### 关键文件路径

- 分箱基类：[hscredit/core/binning/base.py](hscredit/core/binning/base.py)
- 分箱指标计算：[hscredit/core/metrics/_binning.py](hscredit/core/metrics/_binning.py)
- 规则引擎：[hscredit/core/rules/rule.py](hscredit/core/rules/rule.py)
- 报告模块：[hscredit/report/](hscredit/report/)
- 可视化：[hscredit/core/viz/](hscredit/core/viz/)
- 测试目录：[tests/](tests/)

## 测试注意事项

- 测试标记：`@pytest.mark.slow`、`@pytest.mark.integration`、`@pytest.mark.unit`
- 部分测试文件在 [tests/conftest.py](tests/conftest.py) 的 `collect_ignore` 中被排除，因为它们依赖本地 xlsx 文件或包含脚本风格逻辑
- 测试目录结构：`test_binning/`、`test_encoding/`、`test_feature_selection/`、`test_modeling/`、`test_models/`、`test_reports/`、`test_rules/`、`test_utils/`、`test_visualization/`

## 代码风格

- 代码格式化：Black（行长度 120，无限制）
- Python 版本：目标 Python 3.9+ 兼容
- 所有用户-facing 输出使用**中文**
