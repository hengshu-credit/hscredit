# hscredit 快速参考指南

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        hscredit 架构分层                             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: 报告层 (Report)                                           │
│    ├─ ExcelWriter      - 样式化Excel写入器                          │
│    ├─ FeatureAnalyzer  - 特征分析器                                 │
│    ├─ ruleset_report   - 规则集报告                                 │
│    └─ swap_analysis    - 置换风险分析                               │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: 核心层 (Core)                                             │
│    ├─ binning/         - 15+ 分箱算法                               │
│    ├─ encoders/        - 8种 特征编码                               │
│    ├─ selectors/       - 20+ 特征筛选方法                           │
│    ├─ models/          - 模型/损失函数                              │
│    │   ├─ losses/      - 自定义损失函数                             │
│    │   ├─ LogisticRegression                                       │
│    │   └─ ScoreCard    - 评分卡模型                                 │
│    ├─ metrics/         - KS/AUC/PSI/IV/Gini                         │
│    ├─ viz/             - 可视化图表                                 │
│    ├─ feature_engineering/  - 特征工程                              │
│    ├─ rules/           - 规则引擎                                   │
│    ├─ financial/       - 金融计算                                   │
│    ├─ tuning/          - 超参数调优 [新增]                          │
│    └─ explainability/  - SHAP解释 [新增]                            │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: 工具层 (Utils)                                            │
│    ├─ datasets         - 数据集工具                                 │
│    ├─ io              - IO工具                                      │
│    ├─ logger          - 日志工具                                    │
│    └─ describe        - 数据描述                                    │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: 基础层                                                    │
│    ├─ numpy/pandas    - 数据处理                                    │
│    ├─ sklearn         - 机器学习基础                                │
│    ├─ scipy           - 科学计算                                    │
│    └─ matplotlib      - 可视化基础                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## 参考库对比

| 功能 | toad | optbinning | scorecardpy | skorecard | hscredit |
|------|------|------------|-------------|-----------|----------|
| **分箱** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| **编码** | WOE | - | WOE | WOE | 8种编码 |
| **筛选** | ⭐⭐ | - | ⭐ | - | ⭐⭐⭐ |
| **评分卡** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **可视化** | ⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| **调参** | - | - | - | - | ⭐⭐⭐ [新增] |
| **SHAP** | - | - | - | - | ⭐⭐⭐ [新增] |
| **报告** | - | - | - | - | ⭐⭐⭐ [新增] |

## 核心优势

1. **功能完整**: 覆盖建模全流程，无需切换工具
2. **自主可控**: 去除第三方依赖，核心算法自研
3. **API简洁**: 继承scorecardpipeline的简洁风格
4. **可视化精美**: 专业的金融风控可视化风格
5. **报告专业**: Excel/PPT报告一键生成
6. **业务导向**: 深度适配金融信贷场景

## 快速开始示例

### 示例1: 基础分箱分析
```python
import hscredit as hscr
import pandas as pd

# 加载数据
X, y = hscr.germancredit()

# 分箱
binner = hscr.OptimalBinning(method='optimal_iv', max_n_bins=5)
binner.fit(X[['age']], y)

# 查看分箱表
table = binner.get_bin_table('age')
print(table)

# 可视化
hscr.bin_plot(binner, 'age')
```

### 示例2: 特征筛选
```python
# 组合筛选
selector = hscr.CompositeFeatureSelector([
    ('null', hscr.NullSelector(threshold=0.95)),
    ('mode', hscr.ModeSelector(threshold=0.95)),
    ('iv', hscr.IVSelector(threshold=0.02)),
    ('vif', hscr.VIFSelector(threshold=10)),
])

X_selected = selector.fit_transform(X, y)
print(selector.get_report())  # 中文报告
```

### 示例3: 完整建模流程
```python
from sklearn.model_selection import train_test_split

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 分箱+WOE
binner = hscr.OptimalBinning(method='optimal_iv')
encoder = hscr.WOEEncoder()

X_train_woe = encoder.fit_transform(binner.fit_transform(X_train, y_train), y_train)
X_test_woe = encoder.transform(binner.transform(X_test))

# 模型训练
model = hscr.LogisticRegression()
model.fit(X_train_woe, y_train)

# 评估
y_pred = model.predict_proba(X_test_woe)[:, 1]
print(f"KS: {hscr.KS(y_test, y_pred):.4f}")
print(f"AUC: {hscr.AUC(y_test, y_pred):.4f}")
```

## 目录结构

```
hscredit/
├── hscredit/              # 主包
│   ├── core/              # 核心模块
│   │   ├── binning/       # 分箱算法 (15+种)
│   │   ├── encoders/      # 特征编码 (8种)
│   │   ├── selectors/     # 特征筛选 (20+种)
│   │   ├── models/        # 模型与损失函数
│   │   ├── metrics/       # 评估指标
│   │   ├── viz/           # 可视化
│   │   ├── feature_engineering/  # 特征工程
│   │   ├── rules/         # 规则引擎
│   │   ├── financial/     # 金融计算
│   │   ├── tuning/        # 【新增】调参
│   │   └── explainability/ # 【新增】解释性
│   ├── report/            # 报告模块
│   └── utils/             # 工具模块
├── tests/                 # 测试
├── examples/              # 示例
└── docs/                  # 文档
```

## 开发路线图

```
Phase 1 (4-6周): 核心功能补全
├── Optuna调参模块 [2周]
├── SHAP解释性 [2周]
└── 评分卡完善 [2周]

Phase 2 (3-4周): 功能增强
├── 特征工程扩展 [1周]
├── PPT报告生成 [1周]
├── 模型持久化 [1周]
└── 测试覆盖提升 [1周]

Phase 3 (2-3周): 优化完善
├── 性能优化 [1周]
├── 文档完善 [1周]
└── 示例丰富 [1周]

Phase 4 (2周): 开源准备
├── CI/CD配置
├── 发布流程
└── 社区文档
```

## 参考库设计借鉴

### 从 toad 学习
- Combiner 统一分箱接口设计
- 简洁的KS计算 API
- 数据质量检测报告风格

### 从 optbinning 学习
- 严格的数学优化方法
- 丰富的约束条件支持
- 分箱质量评估指标

### 从 scorecardpy 学习
- woebin/woebin_ply 简洁API
- 评分卡转换逻辑
- 模型评估指标体系

### 从 skorecard 学习
- sklearn风格API设计
- Pipeline友好设计
- 详细的文档风格

### 从驻场代码学习
- ModelTuner调参可视化
- 贝叶斯调参实现
- 建模文档自动生成
- 循环删变量逻辑

## 依赖清单

### 核心依赖
```
numpy>=1.19.0
pandas>=1.2.0
scipy>=1.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

### 可选依赖
```
optuna>=3.0.0          # 调参
shap>=0.40.0           # SHAP解释
xgboost>=1.4.0         # XGBoost
lightgbm>=3.2.0        # LightGBM
catboost>=1.0.0        # CatBoost
python-pptx>=0.6.0     # PPT报告
sklearn2pmml>=0.82.0   # PMML导出
```
