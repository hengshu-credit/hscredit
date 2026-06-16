# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HSCredit（衡枢真信）是一个面向金融信贷场景的 Python 3.8+ 信用风险建模工具包，提供**评分卡建模全流程**覆盖：分箱、编码、特征筛选、建模、评估、可视化、规则挖掘与报告生成。所有输出（列名、报告、错误信息）均为**中文**。

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

# 格式化、lint、类型检查、测试（通过 Makefile）
make check

# 或单独运行：
black hscredit tests
flake8 hscredit tests
mypy hscredit --ignore-missing-imports

# 通过 Makefile
make format    # black
make lint      # flake8
make type-check  # mypy
make coverage  # 覆盖率报告
make clean     # 清理 __pycache__ 等
make docs      # 构建 sphinx 文档
make jupyter   # 启动 Jupyter Notebook
make quickstart  # 安装 + 验证环境
```

## 架构设计

### 统一基类继承体系（sklearn 兼容）

所有核心组件均继承自 sklearn `BaseEstimator` + `TransformerMixin`/`ClassifierMixin` + `ABC`：

- `BaseBinning` → [hscredit/core/binning/](hscredit/core/binning/) 中 17 种分箱算法 + InteractionBinning 二维交互分箱
- `BaseEncoder` → [hscredit/core/encoders/](hscredit/core/encoders/) 中 9 种编码器
- `BaseFeatureSelector` → [hscredit/core/selectors/](hscredit/core/selectors/) 中 22+ 种筛选器
- `BaseRiskModel` → [hscredit/core/models/](hscredit/core/models/) 中 boosting 和经典模型

所有组件均兼容 sklearn Pipeline，实现 `fit(X, y)` / `transform(X)` 接口。

### 双 API 调用风格

所有有监督组件支持两种调用方式：

```python
# sklearn 风格 — X (DataFrame/array) 和 y (array) 分别传入
binner.fit(X_train, y_train)

# scorecardpipeline 风格 — 通过 target 参数指定目标列名
binner = OptimalBinning(target='target', ...)
binner.fit(df)  # 自动从 df 中提取目标列

# 混合风格 — y 参数优先于 df 中的 target 列
binner.fit(df, y=ext_y)
```

### 工厂模式：OptimalBinning

`OptimalBinning` 是所有 17 种分箱方法的统一入口，根据 `method` 参数委托给具体分箱类：

```python
# 自动选择最优方法
best_method = OptimalBinning.auto_select_method(X, y, 'feature_name')
binner = OptimalBinning(method=best_method)
```

### 核心模块组织

```
hscredit/
├── core/
│   ├── binning/       # 17 种分箱算法 + BaseBinning + OptimalBinning 工厂
│   ├── encoders/      # 9 种编码器（WOE/Target/Count/OneHot/...）
│   ├── selectors/     # 22+ 种特征筛选器 + CompositeFeatureSelector
│   ├── models/
│   │   ├── boosting/      # XGBoost / LightGBM / CatBoost / NGBoost 包装器
│   │   ├── classical/     # LogisticRegression / RandomForest 等
│   │   ├── scorecard/     # ScoreCard / ScoreTransformer / ScoreDrift
│   │   ├── losses/        # 14 种自定义损失函数 + 框架适配器
│   │   ├── evaluation/    # ModelReport / Calibration / Interpretability
│   │   ├── rules/         # RuleSet / RulesClassifier
│   │   └── tuning/        # Optuna 超参数调优
│   ├── metrics/       # 分类 / 稳定性 / 特征 / 金融指标
│   ├── viz/          # 50+ 种可视化图表
│   ├── eda/          # 数据探索（相关性/特征/账龄/群体稳定性/策略分析）
│   ├── rules/        # 规则引擎（Rule 表达式解析/优化）
│   ├── financial/    # 金融计算（FV/PV/PMT/NPER/IRR/NPV）
│   └── feature_engineering/  # NumExprDerive 表达式衍生
├── report/           # 报告生成器（特征分析/规则挖掘/置换分析/逾期预测）
├── excel/            # ExcelWriter 上下文管理器
└── utils/            # Pandas 扩展 / IO / 日志 / 随机种子
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

miner = SingleFeatureRuleMiner(target='ISBAD', method='optimal_iv', max_n_bins=5)
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
pip install hscredit[lightgbm]      # LightGBM
pip install hscredit[catboost]     # CatBoost
pip install hscredit[deep-learning] # PyTorch, TabNet
pip install hscredit[tune]          # Optuna 超参调优
pip install hscredit[explain]       # SHAP
pip install hscredit[all]           # 所有可选依赖
```

## 代码开发规范

### 开发前必读

1. **修改前**：完整研究项目结构和相关功能的实现方式，所有改动必须与现有实现风格保持一致且逻辑连贯
2. **修改后**：校验所有模块功能点（模块导入、方法使用、输出内容、输出格式）不受本次改动影响
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

- 代码格式化：Black（行长度不限制）
- Python 版本：目标 Python 3.8+ 兼容，支持最新 Python 3.14
- 所有用户-facing 输出使用**中文**

## 未来规划（参考 docs/ROADMAP.md）

| 优先级 | 计划功能 | 状态 |
|--------|----------|------|
| 🔴 P0 | **特征工程模块扩充**（TimeFeatureGenerator / CrossFeatureGenerator / MissingValueImputer） | 待开发 |
| 🔴 P0 | **拒绝推断（Reject Inference）**（三大竞品均无，独家差异化） | 待开发 |
| 🟠 P1 | 分箱质量评分 + batch_to_excel + BestPSIBinning | 待开发 |
| 🟠 P1 | 规则运营工具（覆盖率仿真/跨期追踪/冲突检测） | 待开发 |
| 🟡 P2 | **二维交互分箱（OptimalBinning2D）** | ✅ 已实现 |
| 🟡 P2 | SHAP 报告集成 + 反事实解释（CounterfactualExplainer） | 待开发 |

## 已知待修复问题

根据最近提交和 git 状态，以下文件存在待处理内容：

- `examples/01_binning.ipynb` — 修改待提交
- `examples/08_rules.ipynb` — 修改待提交
- `examples/11_report.ipynb` — 修改待提交
- `examples/21_rule_swap_analysis_v2.ipynb` — 新文件待处理

最近修复的问题（参考提交）：
- `_get_bin_labels` 相关修复
- 分箱约束回归测试
- 用户自定义切分点（strict_user_splits）相关功能
- **规则置换分析（rule_swap_analysis_v2）**：修复了多个问题：
  - **场景流程显示**：各场景正确显示流程（全量→拒绝/置出→剩余→通过→置入→置换）
  - **OUT-IN double uplift bug**：修复了 uplift 被重复应用的问题。OUT-IN 行只显示预测坏样本数（无 uplift），只在 ALL-IN 阶段应用一次 uplift
  - **剩余样本行**：只要有 rules_base 就显示剩余样本行
  - **IN-IN 通过行**：只有有 rules_out 时才显示 IN-IN 通过行
  - **Pandas index 对齐**：使用 .loc[] 而非 .values 确保索引正确对齐