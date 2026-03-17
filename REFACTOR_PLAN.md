# hscredit 项目重构方案

## 一、重构目标

1. **消除代码重复**：移除 `optimal_binning.py` 中的重复类定义
2. **完善API导出**：确保所有公共类和函数在顶层 `__init__.py` 中正确导出
3. **优化项目结构**：分离资源文件、规范化目录命名、拆分大文件
4. **提高可维护性**：统一命名规范、完善模块依赖关系
5. **增强用户体验**：提供清晰一致的API接口

## 二、新项目结构设计

```
hscredit/
├── __init__.py                    # 主包入口，导出所有公共API
├── core/                          # 核心算法模块
│   ├── __init__.py               
│   ├── binning/                   # 分箱算法模块
│   │   ├── __init__.py
│   │   ├── base.py               # 分箱基类
│   │   ├── uniform_binning.py    # 等宽分箱
│   │   ├── quantile_binning.py   # 等频分箱
│   │   ├── tree_binning.py       # 决策树分箱
│   │   ├── cart_binning.py       # CART分箱
│   │   ├── chi_merge_binning.py  # 卡方分箱
│   │   ├── optimal_ks_binning.py # 最优KS分箱
│   │   ├── optimal_iv_binning.py # 最优IV分箱
│   │   ├── optimal_binning.py    # **统一分箱接口（仅保留统一接口逻辑）**
│   │   ├── mdlp_binning.py       # MDLP分箱
│   │   ├── kmeans_binning.py     # KMeans分箱
│   │   ├── monotonic_binning.py  # 单调性约束分箱
│   │   ├── genetic_binning.py    # 遗传算法分箱
│   │   ├── smooth_binning.py     # 平滑分箱
│   │   ├── kernel_density_binning.py # 核密度分箱
│   │   ├── best_lift_binning.py  # Best Lift分箱
│   │   └── target_bad_rate_binning.py # 目标坏样本率分箱
│   ├── encoders/                  # 特征编码器模块
│   │   ├── __init__.py
│   │   ├── base.py               # 编码器基类
│   │   ├── woe_encoder.py        # WOE编码
│   │   ├── target_encoder.py     # 目标编码
│   │   ├── count_encoder.py      # 计数编码
│   │   ├── one_hot_encoder.py    # 独热编码
│   │   ├── ordinal_encoder.py    # 序数编码
│   │   ├── quantile_encoder.py   # 分位数编码
│   │   ├── catboost_encoder.py   # CatBoost编码
│   │   └── gbm_encoder.py        # GBM编码器
│   ├── selectors/                 # 特征筛选模块
│   │   ├── __init__.py
│   │   ├── base.py               # 筛选器基类
│   │   ├── composite.py          # **新增：组合筛选器**
│   │   ├── report_collector.py   # **新增：筛选报告收集器**
│   │   ├── variance_selector.py
│   │   ├── null_selector.py
│   │   ├── mode_selector.py
│   │   ├── corr_selector.py
│   │   ├── vif_selector.py
│   │   ├── iv_selector.py
│   │   ├── lift_selector.py
│   │   ├── psi_selector.py
│   │   ├── cardinality_selector.py
│   │   ├── type_selector.py
│   │   ├── regex_selector.py
│   │   ├── importance_selector.py
│   │   ├── null_importance_selector.py
│   │   ├── rfe_selector.py
│   │   ├── sequential_selector.py
│   │   ├── stepwise_selector.py  # **重命名：step_wise_selector.py → stepwise_selector.py**
│   │   ├── boruta_selector.py
│   │   ├── mutual_info_selector.py
│   │   ├── chi2_selector.py
│   │   └── f_test_selector.py
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── scorecard.py
│   │   └── losses/               # 损失函数子模块
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── adapters.py
│   │       ├── custom_metrics.py
│   │       ├── focal_loss.py
│   │       ├── risk_loss.py
│   │       └── weighted_loss.py
│   ├── metrics/                   # 指标计算模块
│   │   ├── __init__.py
│   │   ├── binning_metrics.py
│   │   ├── classification.py
│   │   ├── importance.py
│   │   ├── regression.py
│   │   └── stability.py
│   ├── viz/                       # 可视化模块
│   │   ├── __init__.py
│   │   ├── binning_plots.py
│   │   └── model_plots.py
│   ├── feature_engineering/       # 特征工程模块
│   │   ├── __init__.py
│   │   └── derive.py             # **重命名：更明确的命名**
│   ├── rules/                     # 规则引擎模块
│   │   ├── __init__.py
│   │   └── rule.py
│   └── financial/                 # 金融计算模块
│       ├── __init__.py
│       └── calculations.py       # **重命名：更明确的命名**
├── report/                        # 报告生成模块
│   ├── __init__.py
│   ├── excel/                     # Excel报告
│   │   ├── __init__.py
│   │   ├── writer.py             # Excel写入器
│   │   └── styles.py             # **新增：样式定义（从writer.py拆分）**
│   ├── feature_analyzer.py       # 特征分箱统计分析
│   ├── feature_report.py         # 特征分析报告
│   ├── ruleset_report.py         # 规则集报告
│   └── swap_analysis_report.py   # 置换分析报告
├── resources/                     # **新增：资源文件目录**
│   ├── fonts/                    # 字体文件
│   │   └── SimHei.ttf
│   └── templates/                # 模板文件
│       └── template.xlsx
├── utils/                         # 工具函数模块
│   ├── __init__.py
│   ├── bin_table_display.py
│   ├── datasets.py
│   ├── describe.py
│   ├── io.py
│   ├── logger.py
│   ├── misc.py
│   └── random.py
├── tests/                         # 测试文件（保持不变）
│   ├── __init__.py
│   ├── test_*.py
│   └── ...
├── examples/                      # 示例文件（清理测试文件）
│   ├── README.md
│   ├── basic_usage.ipynb
│   └── ...
├── setup.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── Makefile
```

## 三、具体重构步骤

### 步骤1：创建resources目录并移动资源文件

**目的**：分离资源文件与代码，提高项目组织的清晰度

**操作**：
1. 创建 `hscredit/resources/` 目录
2. 创建 `hscredit/resources/fonts/` 和 `hscredit/resources/templates/` 子目录
3. 移动 `utils/font.ttf` → `resources/fonts/SimHei.ttf`
4. 移动 `report/excel/template.xlsx` → `resources/templates/template.xlsx`
5. 更新所有引用这些资源文件的代码

### 步骤2：修复optimal_binning.py中的重复类定义

**问题**：`optimal_binning.py` 中重复定义了 `UniformBinning`, `QuantileBinning`, `TreeBinning`, `ChiMergeBinning` 等类

**解决方案**：
1. 删除 `optimal_binning.py` 中重复的类定义
2. 从对应的独立文件导入这些类
3. 保留 `OptimalBinning` 统一接口类的逻辑

### 步骤3：完善顶层__init__.py的API导出

**问题**：部分已实现功能未在顶层导出

**新增导出**：
- `LogisticRegression` 和 `ScoreCard` 模型类
- `GBMEncoder` 编码器
- `KMeansBinning`, `CartBinning`, `GeneticBinning`, `SmoothBinning`, `KernelDensityBinning`, `BestLiftBinning`, `TargetBadRateBinning` 分箱类
- `plot_weights` 可视化函数
- `core.metrics` 中的常用指标函数（`KS`, `AUC`, `PSI`, `IV`, `Gini` 等）

### 步骤4：整理examples和tests目录结构

**问题**：`examples/` 目录中混入了测试文件

**操作**：
1. 识别 `examples/` 目录中的 `test_*.py` 文件
2. 将测试文件移动到 `tests/` 目录
3. 在 `examples/` 目录中添加 `README.md` 说明示例用途
4. 确保示例文件命名清晰（如 `basic_binning_example.py`）

### 步骤5：统一文件和类命名规范

**问题**：文件名与类名不一致，如 `step_wise_selector.py` vs `StepwiseSelector`

**操作**：
1. 重命名 `step_wise_selector.py` → `stepwise_selector.py`
2. 更新所有导入该文件的位置
3. 确保其他文件命名遵循 `lowercase_with_underscores.py` 规范

### 步骤6：拆分超大文件

**问题**：部分文件过大，影响可维护性

**操作**：
1. 拆分 `report/excel/writer.py`：
   - 提取样式定义到 `styles.py`
   - 保留核心写入逻辑在 `writer.py`
2. 拆分 `core/selectors/base.py`：
   - 将 `CompositeFeatureSelector` 移至 `composite.py`
   - 将 `SelectionReportCollector` 移至 `report_collector.py`
   - 保留基类定义在 `base.py`

### 步骤7：修复模块导入关系和依赖

**问题**：条件导入可能隐藏依赖问题，相对导入层级过深

**操作**：
1. 将条件导入改为明确的可选依赖检查：
   ```python
   # 原代码
   try:
       from .report.feature_report import auto_feature_analysis_report
   except ImportError:
       pass
   
   # 修改为
   from .report.feature_report import auto_feature_analysis_report  # 明确导入
   ```
2. 简化相对导入，必要时使用绝对导入
3. 添加完整的类型注解，提高代码可读性

### 步骤8：更新setup.py和pyproject.toml

**操作**：
1. 在 `setup.py` 的 `package_data` 中包含 `resources/` 目录
2. 更新 `pyproject.toml` 中的包配置
3. 确保资源文件在安装时正确打包

## 四、重构后的API使用示例

```python
import hscredit

# 查看版本和可用功能
hscredit.info()

# 分箱模块
from hscredit import OptimalBinning, ChiMergeBinning
binner = OptimalBinning(method='optimal_iv', max_n_bins=5, monotonic='descending')
binner.fit(X, y)
X_woe = binner.transform(X, metric='woe')

# 编码器模块
from hscredit import WOEEncoder, GBMEncoder
encoder = WOEEncoder(cols=['feature1', 'feature2'])
X_woe = encoder.fit_transform(X, y)

# 特征筛选模块
from hscredit import IVSelector, VIFSelector, CompositeFeatureSelector
selector = IVSelector(threshold=0.02)
selector.fit(X, y)
X_selected = selector.transform(X)

# 模型模块
from hscredit import LogisticRegression, ScoreCard
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.summary())

# 指标计算
from hscredit import KS, AUC, PSI
ks_value = KS(y_true, y_pred)
auc_value = AUC(y_true, y_pred)

# 可视化
from hscredit import bin_plot, ks_plot
bin_plot(binner.get_bin_table('feature'))
ks_plot(y_true, y_pred)

# 报告生成
from hscredit import ExcelWriter, feature_bin_stats
writer = ExcelWriter('report.xlsx')
writer.write_df(df, sheet_name='data')
writer.save()
```

## 五、重构优先级

### 高优先级（立即执行）
1. ✅ 创建resources目录并移动资源文件
2. ✅ 修复optimal_binning.py中的重复类定义
3. ✅ 完善顶层__init__.py的API导出
4. ✅ 整理examples和tests目录结构

### 中优先级（近期执行）
5. 统一文件和类命名规范
6. 拆分超大文件（writer.py, base.py）
7. 修复模块导入关系

### 低优先级（长期改进）
8. 完善类型注解
9. 增加单元测试覆盖率
10. 编写详细的使用文档

## 六、预期效果

重构完成后，hscredit将具备：
- ✅ 清晰的项目结构和模块划分
- ✅ 一致的命名规范
- ✅ 完整的API导出
- ✅ 零代码重复
- ✅ 良好的可维护性
- ✅ 优秀的用户体验
