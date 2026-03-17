# hscredit - 金融信贷风险策略和模型开发库

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**一个完整的金融信贷风险建模工具包**

[快速开始](#快速开始) | [功能特性](#功能特性) | [文档](#文档) | [示例](#示例)

</div>

---

## 📖 简介

`hscredit` 是一个专业的金融信贷风险策略和模型开发库,从 `scorecardpipeline` 迁移而来,旨在成为公司级开源项目。核心特点是去除对第三方风控库(toad、optbinning、scorecardpy)的依赖,自主实现核心功能,提供更易用的API和完善的文档。

### 主要功能

- ✅ **评分卡建模**: 完整的评分卡建模流程,支持自动分箱、WOE转换、评分卡生成
- ✅ **策略分析**: 风控策略分析、规则效果评估
- ✅ **规则挖掘**: 基于决策树的规则自动提取
- ✅ **特征筛选**: 多种特征筛选方法,支持IV、相关性、VIF等
- ✅ **自动分箱**: 多种分箱算法,包括决策树、卡方、最优分箱等
- ✅ **自定义损失函数**: 支持XGBoost、LightGBM、CatBoost、TabNet的自定义损失和评估指标
- ✅ **Excel报告**: 自动生成专业的模型报告
- ✅ **PMML导出**: 支持评分卡导出为PMML格式
- ✅ **超参数搜索**: 支持Pipeline式的超参数优化

---

## 🚀 快速开始

### 安装

```bash
pip install hscredit
```

### 基本使用

```python
import hscredit as hsc
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 加载示例数据
data = hsc.datasets.load_german_credit()
X, y = data.drop('target', axis=1), data['target']

# 构建评分卡建模Pipeline
pipeline = Pipeline([
    ('binning', hsc.OptimalBinning(method='tree', max_n_bins=5)),
    ('encoding', hsc.WOEEncoder()),
    ('selection', hsc.FeatureSelector(iv_threshold=0.02)),
    ('model', LogisticRegression())
])

# 训练模型
pipeline.fit(X, y)

# 预测
y_pred = pipeline.predict_proba(X)[:, 1]

# 创建评分卡
scorecard = hsc.ScoreCard(
    pdo=60,
    base_score=750,
    combiner=pipeline.named_steps['binning'],
    encoder=pipeline.named_steps['encoding'],
    model=pipeline.named_steps['model']
)

# 计算评分
scores = scorecard.transform(X)

# 生成Excel报告
report = hsc.ExcelReport('model_report.xlsx')
report.add_model_summary(scorecard, X, y)
report.save()
```

---

## ✨ 功能特性

### 1. 多种分箱算法

```python
from hscredit import OptimalBinning

# 决策树分箱
binner_tree = OptimalBinning(method='tree', max_n_bins=5)

# 卡方分箱
binner_chi = OptimalBinning(method='chi', max_n_bins=5)

# 最优分箱(带单调性约束)
binner_opt = OptimalBinning(
    method='optimal',
    monotonic_trend='ascending',
    solver='cp'
)

binner.fit(X['age'], y)
X_binned = binner.transform(X['age'])
```

### 2. 智能特征筛选

```python
from hscredit import FeatureSelector

selector = FeatureSelector(
    iv_threshold=0.02,          # IV值筛选
    corr_threshold=0.7,         # 相关性筛选
    vif_threshold=10,           # VIF筛选
    missing_threshold=0.95      # 缺失率筛选
)

selector.fit(X, y)
X_selected = selector.transform(X)

# 查看筛选结果
print(selector.selected_features_)
print(selector.removed_features_)
```

### 3. 逐步回归

```python
from hscredit import StepwiseSelector

stepwise = StepwiseSelector(
    direction='both',
    criterion='aic',
    p_enter=0.05,
    p_remove=0.1
)

stepwise.fit(X_woe, y)
X_final = stepwise.transform(X_woe)
```

### 4. 评分卡生成

```python
from hscredit import ScoreCard

scorecard = ScoreCard(
    pdo=60,
    rate=2,
    base_odds=35,
    base_score=750,
    combiner=binner,
    encoder=encoder
)

scorecard.fit(X_woe, y)

# 获取评分卡表
score_table = scorecard.score_table_

# 导出PMML
scorecard.to_pmml('scorecard.pmml')
```

### 5. 规则挖掘

```python
from hscredit import RuleExtractor

extractor = RuleExtractor(
    max_depth=3,
    min_samples_leaf=100,
    max_rules=10
)

extractor.fit(X, y)
rules = extractor.get_rules()

# 评估规则
evaluator = hsc.RuleEvaluator()
report = evaluator.evaluate(rules, X_test, y_test)
```

### 6. 指标计算

```python
from hscredit.metrics import KS, AUC, PSI, IV

# KS计算
ks_value = KS(y_score, y_true)
ks_table = KS_bucket(y_score, y_true, bucket=10)

# AUC计算
auc_value = AUC(y_score, y_true)

# PSI计算
psi_value = PSI(train_score, test_score)

# IV计算
iv_df = IV(X, y, return_dataframe=True)
```

### 7. 自定义损失函数

```python
from hscredit.core.models import (
    FocalLoss, WeightedBCELoss, CostSensitiveLoss,
    BadDebtLoss, ProfitMaxLoss,
    KSMetric, GiniMetric, PSIMetric,
    XGBoostLossAdapter, LightGBMLossAdapter
)
import xgboost as xgb
import lightgbm as lgb

# 示例1: 使用Focal Loss处理不平衡数据
focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
focal_adapter = XGBoostLossAdapter(focal_loss)

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'objective': 'binary:logistic', 'eval_metric': 'auc'}
bst = xgb.train(
    params, dtrain,
    obj=focal_adapter.objective(),  # 自定义损失
    num_boost_round=100
)

# 示例2: 成本敏感学习
# 假设漏抓坏客户损失10000元,误拒好客户损失100元
cost_loss = CostSensitiveLoss(fn_cost=100, fp_cost=1)
cost_adapter = LightGBMLossAdapter(cost_loss)

train_data = lgb.Dataset(X_train, label=y_train)
bst = lgb.train(
    {'objective': 'binary', 'metric': 'auc'},
    train_data,
    fobj=cost_adapter.objective(),
    num_boost_round=100
)

# 示例3: 坏账率优化
bad_debt_loss = BadDebtLoss(
    target_approval_rate=0.3,  # 目标通过率30%
    bad_debt_weight=1.0
)

# 示例4: 利润最大化
profit_loss = ProfitMaxLoss(
    interest_income=100,   # 每笔贷款利息收益100元
    bad_debt_loss=1000     # 每笔坏账损失1000元
)

# 示例5: 使用自定义评估指标
ks_metric = KSMetric()
bst = lgb.train(
    {'objective': 'binary'},
    train_data,
    feval=focal_adapter.metric(ks_metric),  # KS作为评估指标
    num_boost_round=100
)
```

---

## 📚 文档

- [API参考文档](https://hscredit.readthedocs.io/api/)
- [使用教程](https://hscredit.readthedocs.io/tutorials/)
- [示例代码](examples/)
- [迁移指南](docs/migration.md)

---

## 📁 项目结构

```
hscredit/
├── core/                   # 核心算法
│   ├── binning/           # 分箱算法
│   ├── encoding/          # 编码转换
│   ├── selection/         # 特征筛选
│   └── metrics/           # 指标计算
├── model/                 # 模型模块
│   ├── linear/            # 线性模型
│   ├── scorecard/         # 评分卡
│   └── losses/            # 自定义损失函数
│       ├── focal_loss.py          # Focal Loss
│       ├── weighted_loss.py       # 加权损失
│       ├── risk_loss.py           # 风控业务损失
│       ├── custom_metrics.py      # 自定义指标
│       └── adapters.py            # 框架适配器
├── analysis/              # 分析模块
│   ├── strategy/          # 策略分析
│   └── rules/             # 规则挖掘
├── report/                # 报告模块
│   ├── excel/             # Excel报告
│   └── plot/              # 可视化
├── utils/                 # 工具模块
└── examples/              # 示例代码
```

---

## 🔧 开发指南

### 环境设置

```bash
# 克隆仓库
git clone https://github.com/hscredit/hscredit.git
cd hscredit

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -e ".[dev]"
```

### 环境验证

```bash
# 快速验证环境
python scripts/validate_environment.py

# 或使用Makefile
make validate
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行带覆盖率的测试
pytest tests/ --cov=hscredit --cov-report=html

# 运行特定测试
pytest tests/test_binning.py -v
```

### 使用Jupyter Notebook验证

```bash
# 安装Jupyter（如果未安装）
pip install jupyter notebook ipykernel

# 注册kernel
python -m ipykernel install --user --name=hscredit

# 启动Jupyter
make jupyter
# 或
cd examples && jupyter notebook

# 打开并运行以下notebook进行验证：
# 1. 00_project_overview.ipynb - 项目总览
# 2. 01_excel_writer_validation.ipynb - Excel模块验证
```

### 代码规范

```bash
# 格式化代码
black hscredit/

# 排序import
isort hscredit/

# 代码检查
flake8 hscredit/

# 类型检查
mypy hscredit/
```

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---

## 🙏 致谢

本项目参考了以下优秀的开源项目:

- [toad](https://github.com/amphibian-dev/toad) - 风控建模工具
- [optbinning](https://github.com/guillermo-navas-palencia/optbinning) - 最优分箱
- [scorecardpy](https://github.com/ShichenXie/scorecardpy) - 评分卡建模
- [sklearn](https://scikit-learn.org/) - 机器学习库

---

## 📮 联系方式

- 项目主页: https://github.com/hscredit/hscredit
- 问题反馈: https://github.com/hscredit/hscredit/issues
- 邮箱: hscredit@example.com

---

<div align="center">
Made with ❤️ by hscredit team
</div>
