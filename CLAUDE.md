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

# 完整检查（格式化 + lint + 类型检查 + 测试）
make check

# 单独运行
black hscredit tests              # 格式化
flake8 hscredit tests             # lint
mypy hscredit --ignore-missing-imports  # 类型检查

# 构建与发布验证
python -m build
python -m twine check dist/*
```

## 架构设计

### 统一基类继承体系（sklearn 兼容）

所有核心组件均继承自 sklearn `BaseEstimator` + `TransformerMixin`/`ClassifierMixin` + `ABC`，兼容 sklearn Pipeline：

- `BaseBinning` → [hscredit/core/binning/base.py](hscredit/core/binning/base.py) — 18 种分箱算法 + `OptimalBinning2D` 二维交互分箱
- `BaseEncoder` → [hscredit/core/encoders/base.py](hscredit/core/encoders/base.py) — 10 种编码器
- `BaseFeatureSelector` → [hscredit/core/selectors/base.py](hscredit/core/selectors/base.py) — 22 种筛选器 + `CompositeFeatureSelector` + `SelectionReportCollector`
- `BaseRiskModel` → [hscredit/core/models/base.py](hscredit/core/models/base.py) — boosting 和经典模型

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

`OptimalBinning` 是所有 18 种分箱方法的统一入口，支持预分箱 + 二次分箱的两阶段流程：

```python
best_method = OptimalBinning.auto_select_method(X, y, 'feature_name')
binner = OptimalBinning(method=best_method)
```

### 懒加载策略

Boosting 模型（XGBoost/LightGBM/CatBoost/NGBoost）和调参工具（Optuna）通过 `__getattr__` 懒加载，避免 `import hscredit` 时加载重依赖。两层 `__getattr__`：顶层 `hscredit/__init__.py` 委托给 `core/models/__init__.py`，后者按需导入 `boosting/` 或 `tuning/` 子包。

### 核心模块组织

```
hscredit/
├── core/
│   ├── binning/       # 18 种分箱算法 + BaseBinning + OptimalBinning 工厂 + OptimalBinning2D
│   ├── encoders/      # 10 种编码器（WOE/Target/Count/OneHot/Ordinal/Quantile/CatBoost/GBM/Cardinality）
│   ├── selectors/     # 22 种筛选器 + CompositeFeatureSelector + SelectionReportCollector
│   ├── models/
│   │   ├── boosting/      # XGBoost / LightGBM / CatBoost / NGBoost（懒加载）
│   │   ├── classical/     # LogisticRegression / RandomForest / ExtraTrees / GradientBoosting
│   │   ├── scorecard/     # ScoreCard / RoundScoreCard / ScoreTransformer / ScoreDriftCalibrator
│   │   ├── losses/        # 15 种自定义损失函数 + 3 种评估指标 + 5 种框架适配器
│   │   ├── evaluation/    # ModelReport / Calibration / Interpretability
│   │   ├── rules/         # RuleSet / RulesClassifier / LogicOperator
│   │   └── tuning/        # Optuna 超参数调优（懒加载）
│   ├── metrics/       # 分类 / 稳定性 / 特征 / 金融 / 回归指标 + 分箱统计
│   ├── viz/           # 8 个子模块: binning/model/risk/variable/score/strategy/tree/style
│   ├── eda/           # 10 个分析子模块: overview/target/feature/relationship/correlation/stability/population/strategy/vintage/report
│   ├── rules/         # Rule 表达式解析 + ExprOptimizer
│   ├── financial/     # 基础金融计算 (FV/PV/PMT/NPER/IPMT/PPMT/RATE) + 高级 (NPV/IRR/MIRR)
│   └── feature_engineering/  # NumExprDerive 表达式衍生
├── report/             # 报告生成器
│   └── mining/         # 规则挖掘器（Single/Multi/MultiLabel/Tree/ManualTree）
├── excel/              # ExcelWriter 上下文管理器 + dataframe2excel
└── utils/              # Pandas 扩展 / IO / 日志 / 随机种子 / 输入校验
```

### 关键设计模式

**SelectionReportCollector**：聚合 Pipeline 中多个筛选器的报告，生成统一中文摘要。每个筛选器实现 `get_selection_report()` 返回含中文键的字典（`输入特征数`, `选中特征数`, `特征列表`, `筛选方法`）。

**Loss Adapter 模式**：`BaseLoss` 子类通过 `to_xgboost()`, `to_lightgbm()`, `to_catboost()` 便捷方法或 `XGBoostLossAdapter`, `LightGBMLossAdapter` 等适配器转换为各框架格式。

**Pandas 扩展**：`import hscredit` 自动注册 `df.summary()`, `df.save()`, `df.show()`, `df.eda_info()`, `df.missing_analysis()`, `series.save()`。

**评分卡继承链**：`ScoreCard` → `StandardScoreTransformer` → `BaseScoreTransformer`。评分公式：`Score = A - B × ln(odds)`，其中 `B = pdo / ln(rate)`，`A = base_score + B × ln(actual_odds)`。

**规则引擎**：`Rule` 类使用 pandas eval 语法，支持 `&`（与）、`|`（或）、`~`（非）、`^`（异或）运算符组合。AST 解析提取列名，`ExprOptimizer` 简化和美化表达式。

### 异常体系

所有自定义异常继承自 `HSCreditError`（9 种）：`ValidationError`, `InputValidationError`, `InputTypeError`, `FeatureNotFoundError`, `StateError`, `NotFittedError`, `DependencyError`, `SerializationError`。辅助函数 `raise_not_fitted()`, `raise_feature_not_found()`, `raise_missing_columns()` 可用。

### 可选依赖

```bash
pip install hscredit[boost]       # XGBoost / LightGBM / CatBoost / NGBoost
pip install hscredit[net]         # PyTorch / TabNet
pip install hscredit[tune]        # Optuna 超参调优
pip install hscredit[explain]     # SHAP
pip install hscredit[pmml]        # PMML 导出
pip install hscredit[all]         # 所有可选依赖
```

## 代码开发规范

### 开发前必读

1. **修改前**：完整研究项目结构和相关功能的实现方式，所有改动必须与现有实现风格保持一致且逻辑连贯
2. **修改后**：使用 `examples/hscredit_yyp.xlsx` 验证所有模块功能点（模块导入、方法使用、输出内容、输出格式）不受本次改动影响
3. **分箱**：所有分箱使用 `hscredit.core.binning` 中的分箱器，所有分箱指标计算使用 `compute_bin_stats`
4. **规则**：所有规则使用 `hscredit.core.rules.rule` 中的 `Rule` 类实现，报告指标计算使用 `Rule.report`
5. **中文输出**：所有 DataFrame 列名、错误信息、报告内容使用中文

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
- 评分卡模型：[hscredit/core/models/scorecard/scorecard.py](hscredit/core/models/scorecard/scorecard.py)
- 损失函数基类：[hscredit/core/models/losses/base.py](hscredit/core/models/losses/base.py)
- 报告模块：[hscredit/report/](hscredit/report/)
- 可视化：[hscredit/core/viz/](hscredit/core/viz/)
- 测试目录：[tests/](tests/)

## 测试注意事项

- 测试标记：`@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.unit`
- 部分测试文件在 [tests/conftest.py](tests/conftest.py) 的 `collect_ignore` 中被排除（依赖本地 xlsx 或包含脚本风格逻辑）
- 测试目录存在重复命名：`test_models/` vs `test_modeling/`, `test_report/` vs `test_reports/`
- 测试子目录均缺少 `__init__.py`（不影响 pytest 收集，但结构不统一）
- 覆盖率不足的模块：EDA, financial, 大部分 encoders, viz 函数, overdue_predictor, swap_analysis

## 代码风格

- 代码格式化：Black（行长度 120）
- Python 版本：目标 Python 3.9+ 兼容
- 所有用户可见输出使用**中文**

## 已知问题

1. `hscredit.info()` 中 "待实现模块" 文本过期：列出了 `core.encoding`（不存在）和 `core.metrics`（已实现）
2. `core/eda/__init__.py` 的 `__all__` 导出了两个私有函数 `_build_overdue_labels` 和 `_create_binary_target`
3. `init_setting()` 在 `import hscredit` 时全局调用 `warnings.filterwarnings("ignore")`，会抑制所有警告
4. EDA 模块中存在大量 `except Exception:` 裸异常捕获，会静默吞掉错误
5. `BalancedFocalLoss` 导出不一致：在 models `__init__.py` 中用 try/except 包裹但在多处 `__all__` 中列出
6. 部分损失函数（`RankingAUCProxyLoss`, `KSFocusedLoss`, `TopKBadCaptureLoss`, `AmountWeightedLoss`, `ExpectedValueLoss`）在 `core/__init__.py` 显式导入但不在 `models/__all__` 中
