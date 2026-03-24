# hscredit 具体修复建议和实现计划

## 一、现有代码问题清单与修复方案

### 1.1 核心问题

#### 问题1: 部分模块仅声明未实现
**位置**: `hscredit/core/feature_engineering/__init__.py`

```python
# 当前只有 NumExprDerive
from .expression import NumExprDerive
__all__ = ['NumExprDerive']
```

**修复方案**:
```python
# hscredit/core/feature_engineering/__init__.py
from .expression import NumExprDerive
from .auto_features import AutoFeatureGenerator
from .time_features import TimeFeatureGenerator
from .cross_features import CrossFeatureGenerator
from .polynomial import PolynomialFeatureGenerator

__all__ = [
    'NumExprDerive',
    'AutoFeatureGenerator',
    'TimeFeatureGenerator',
    'CrossFeatureGenerator',
    'PolynomialFeatureGenerator',
]
```

---

#### 问题2: ScoreCard类未完整实现
**位置**: `hscredit/core/models/scorecard.py`

**缺失功能**:
- 概率到分数的线性转换
- 分箱WOE表生成
- 评分卡表生成
- 部署代码生成

**修复方案**:
```python
class ScoreCard(BaseEstimator):
    """完整评分卡模型.
    
    功能:
    1. WOE编码 + 逻辑回归
    2. 概率到分数转换
    3. 评分卡表生成
    4. 部署代码导出
    """
    
    def __init__(self, 
                 binner=None,
                 encoder=None,
                 classifier=None,
                 PDO=20, 
                 base_score=600, 
                 base_odds=50,
                 rate=2):
        self.binner = binner or OptimalBinning(method='optimal_iv')
        self.encoder = encoder or WOEEncoder()
        self.classifier = classifier or LogisticRegression()
        self.PDO = PDO
        self.base_score = base_score
        self.base_odds = base_odds
        self.rate = rate
        
    def fit(self, X, y):
        # 1. 分箱
        self.binner.fit(X, y)
        X_binned = self.binner.transform(X)
        
        # 2. WOE编码
        self.encoder.fit(X_binned, y)
        X_woe = self.encoder.transform(X_binned)
        
        # 3. 逻辑回归
        self.classifier.fit(X_woe, y)
        
        # 4. 生成分箱表
        self.binning_tables_ = self._generate_binning_tables(X, y)
        
        # 5. 生成评分卡表
        self.scorecard_table_ = self._generate_scorecard_table()
        
        return self
    
    def predict_score(self, X):
        """预测分数 (300-1000)"""
        proba = self.predict_proba(X)[:, 1]
        return self._proba_to_score(proba)
    
    def _proba_to_score(self, proba):
        """概率转分数"""
        B = self.PDO / np.log(self.rate)
        A = self.base_score + B * np.log(self.base_odds)
        odds = proba / (1 - proba)
        score = A - B * np.log(odds)
        return np.clip(score, 300, 1000)
    
    def _generate_scorecard_table(self):
        """生成评分卡表 (变量/分箱/分数)"""
        # 返回DataFrame: [变量名, 分箱, WOE, 系数, 分数]
        pass
    
    def export_deployment_code(self, filepath, language='python'):
        """导出部署代码"""
        if language == 'python':
            self._export_python_code(filepath)
        elif language == 'java':
            self._export_java_code(filepath)
        elif language == 'sql':
            self._export_sql_code(filepath)
```

---

#### 问题3: 缺失Optuna调参模块
**位置**: 需要新建 `hscredit/core/tuning/`

**实现方案**:
```python
# hscredit/core/tuning/__init__.py
from .optuna_tuner import OptunaTuner
from .param_space import LGBMParamSpace, XGBParamSpace, CatBoostParamSpace

__all__ = [
    'OptunaTuner',
    'LGBMParamSpace',
    'XGBParamSpace', 
    'CatBoostParamSpace',
]
```

```python
# hscredit/core/tuning/optuna_tuner.py
import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history

class OptunaTuner:
    """Optuna超参数调优器.
    
    Example:
        >>> from hscredit.core.tuning import OptunaTuner, LGBMParamSpace
        >>> tuner = OptunaTuner(
        ...     model_type='lgbm',
        ...     param_space=LGBMParamSpace(),
        ...     metric='ks',
        ...     direction='maximize'
        ... )
        >>> tuner.fit(X_train, y_train, X_val, y_val)
        >>> best_params = tuner.best_params_
        >>> tuner.plot_optimization_history()
        >>> tuner.plot_param_importances()
    """
    
    def __init__(self, model_type, param_space, metric='ks', 
                 direction='maximize', n_trials=100, timeout=None,
                 early_stopping_rounds=50):
        self.model_type = model_type
        self.param_space = param_space
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.early_stopping_rounds = early_stopping_rounds
        self.study_ = None
        
    def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):
        """执行调参"""
        def objective(trial):
            params = self.param_space.get_params(trial)
            model = self._create_model(params)
            
            if self.model_type in ['lgbm', 'xgboost']:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[optuna.integration.LightGBMPruningCallback(trial, self.metric)]
                )
            else:
                model.fit(X_train, y_train)
                
            y_pred = model.predict_proba(X_val)[:, 1]
            return self._calculate_metric(y_val, y_pred)
        
        self.study_ = optuna.create_study(direction=self.direction)
        self.study_.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params_ = self.study_.best_params
        return self
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        return plot_optimization_history(self.study_)
    
    def plot_param_importances(self):
        """绘制参数重要性"""
        return plot_param_importances(self.study_)
    
    def plot_contour(self, params=None):
        """绘制等高线图"""
        from optuna.visualization import plot_contour
        return plot_contour(self.study_, params=params)
```

---

#### 问题4: 缺失SHAP解释性模块
**位置**: 需要新建 `hscredit/core/explainability/`

**实现方案**:
```python
# hscredit/core/explainability/__init__.py
from .shap_explainer import SHAPExplainer
from .feature_importance import FeatureImportanceAggregator

__all__ = ['SHAPExplainer', 'FeatureImportanceAggregator']
```

```python
# hscredit/core/explainability/shap_explainer.py
import shap

class SHAPExplainer:
    """SHAP解释器封装.
    
    Example:
        >>> from hscredit.core.explainability import SHAPExplainer
        >>> explainer = SHAPExplainer(model, model_type='lgbm')
        >>> explainer.fit(X_train)
        >>> shap_values = explainer.explain(X_test)
        >>> explainer.summary_plot()
        >>> explainer.waterfall_plot(idx=0)
    """
    
    def __init__(self, model, model_type='lgbm', feature_names=None):
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        self.explainer_ = None
        self.shap_values_ = None
        
    def fit(self, X_background):
        """初始化解释器"""
        if self.model_type in ['lgbm', 'xgboost', 'catboost']:
            self.explainer_ = shap.TreeExplainer(self.model)
        else:
            self.explainer_ = shap.KernelExplainer(self.model.predict_proba, X_background)
        return self
    
    def explain(self, X):
        """计算SHAP值"""
        self.shap_values_ = self.explainer_.shap_values(X)
        return self.shap_values_
    
    def summary_plot(self, plot_type='dot', max_display=20):
        """全局特征重要性图"""
        return shap.summary_plot(
            self.shap_values_, 
            self.feature_names,
            plot_type=plot_type,
            max_display=max_display
        )
    
    def waterfall_plot(self, idx=0):
        """单个样本解释瀑布图"""
        return shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values_[idx],
                base_values=self.explainer_.expected_value,
                feature_names=self.feature_names
            )
        )
    
    def force_plot(self, idx=0):
        """单个样本力图"""
        return shap.force_plot(
            self.explainer_.expected_value,
            self.shap_values_[idx],
            self.feature_names
        )
    
    def dependence_plot(self, feature, interaction_feature=None):
        """特征依赖图"""
        return shap.dependence_plot(
            feature,
            self.shap_values_,
            interaction_index=interaction_feature
        )
    
    def get_feature_importance(self):
        """获取全局特征重要性"""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.shap_values_).mean(axis=0)
        }).sort_values('importance', ascending=False)
```

---

### 1.2 代码质量问题

#### 问题5: 类型注解不完整
**修复方案**: 为所有公共API添加完整类型注解

```python
# 示例: BaseBinning基类添加类型注解
from typing import Union, Optional, List, Dict, Any
import pandas as pd
import numpy as np

class BaseBinning:
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            sample_weight: Optional[np.ndarray] = None) -> 'BaseBinning':
        ...
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...
    
    def get_bin_table(self, feature: str) -> pd.DataFrame:
        ...
```

#### 问题7: 错误处理不完善
**修复方案**: 定义自定义异常体系

```python
# hscredit/exceptions.py
class HSCreditError(Exception):
    """hscredit基础异常"""
    pass

class BinningError(HSCreditError):
    """分箱相关异常"""
    pass

class EncodingError(HSCreditError):
    """编码相关异常"""
    pass

class SelectionError(HSCreditError):
    """特征筛选相关异常"""
    pass

class ModelingError(HSCreditError):
    """建模相关异常"""
    pass
```

---

## 二、具体实现任务清单

### 2.1 新增模块开发

| 优先级 | 模块 | 文件 | 工作量 | 依赖 |
|--------|------|------|--------|------|
| P0 | 调参模块 | core/tuning/ | 3天 | optuna |
| P0 | SHAP解释 | core/explainability/ | 2天 | shap |
| P0 | 评分卡完善 | core/models/scorecard.py | 2天 | - |
| P1 | 特征工程 | core/feature_engineering/ | 2天 | - |
| P1 | PPT报告 | report/ppt/ | 2天 | python-pptx |
| P1 | 模型持久化 | core/persistence/ | 2天 | joblib |
| P2 | AutoML | core/automl/ | 3天 | 所有模块 |

### 2.2 现有模块修复

| 优先级 | 模块 | 问题 | 工作量 |
|--------|------|------|--------|
| P0 | core/__init__.py | 完善导出 | 0.5天 |
| P0 | core/binning/ | 类型注解 | 1天 |
| P0 | core/selectors/ | 类型注解 | 1天 |
| P0 | core/encoders/ | 类型注解 | 1天 |
| P1 | utils/ | 完善工具函数 | 1天 |
| P1 | report/ | 完善报告功能 | 2天 |
| P2 | tests/ | 增加测试覆盖 | 3天 |

### 2.3 文档和示例

| 优先级 | 任务 | 工作量 |
|--------|------|--------|
| P0 | API文档 | 2天 |
| P0 | 使用教程 | 3天 |
| P1 | 示例notebook | 2天 |
| P1 | 开发文档 | 1天 |
| P2 | 视频教程 | 待定 |

---

## 三、关键代码示例

### 3.1 完整建模流程示例

```python
"""完整建模流程示例 - 展示hscredit最佳实践"""

import pandas as pd
from sklearn.model_selection import train_test_split

import hscredit as hscr

# 1. 数据加载
X, y = hscr.germancredit()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 2. 数据探索
print(hscr.feature_describe(X_train))

# 3. 特征筛选 (组合多种筛选方法)
selector = hscr.CompositeFeatureSelector([
    ('null', hscr.NullSelector(threshold=0.95)),
    ('mode', hscr.ModeSelector(threshold=0.95)),
    ('cardinality', hscr.CardinalitySelector(threshold=100)),
    ('iv', hscr.IVSelector(threshold=0.02)),
    ('corr', hscr.CorrSelector(threshold=0.8)),
    ('vif', hscr.VIFSelector(threshold=10)),
])
X_train_sel = selector.fit_transform(X_train, y_train)
X_test_sel = selector.transform(X_test)
print(selector.get_report())  # 中文筛选报告

# 4. 分箱与编码
binner = hscr.OptimalBinning(method='optimal_iv', max_n_bins=5)
encoder = hscr.WOEEncoder()

X_train_binned = binner.fit_transform(X_train_sel, y_train)
X_test_binned = binner.transform(X_test_sel)

X_train_woe = encoder.fit_transform(X_train_binned, y_train)
X_test_woe = encoder.transform(X_test_binned)

# 5. 模型训练
from hscredit.core.models import LogisticRegression

model = LogisticRegression(calculate_stats=True)
model.fit(X_train_woe, y_train)
print(model.summary())  # 模型统计摘要

# 6. 模型调参 (使用Optuna)
from hscredit.core.tuning import OptunaTuner, LGBMParamSpace

tuner = OptunaTuner(
    model_type='lgbm',
    param_space=LGBMParamSpace(),
    metric='ks',
    n_trials=50
)
tuner.fit(X_train_sel, y_train, X_test_sel, y_test)
best_params = tuner.best_params_

# 7. SHAP解释
from hscredit.core.explainability import SHAPExplainer

explainer = SHAPExplainer(model, model_type='linear', feature_names=X_train_woe.columns)
explainer.fit(X_train_woe)
shap_values = explainer.explain(X_test_woe)
explainer.summary_plot()

# 8. 模型评估
from hscredit import KS, AUC, PSI, Gini

y_pred = model.predict_proba(X_test_woe)[:, 1]
print(f"KS: {KS(y_test, y_pred):.4f}")
print(f"AUC: {AUC(y_test, y_pred):.4f}")
print(f"Gini: {Gini(y_test, y_pred):.4f}")

# 9. 生成报告
from hscredit.report import auto_feature_analysis_report

report = auto_feature_analysis_report(
    X_train, y_train, 
    binner=binner,
    output_path='feature_report.xlsx'
)

# 10. 评分卡转换 (如果是评分卡模型)
from hscredit.core.models import ScoreCard

scorecard = ScoreCard(PDO=20, base_score=600)
scorecard.fit(X_train_sel, y_train)
scores = scorecard.predict_score(X_test_sel)

# 导出评分卡表
print(scorecard.scorecard_table_)

# 导出部署代码
scorecard.export_deployment_code('scorecard_deploy.py', language='python')
```

### 3.2 策略分析流程示例

```python
"""策略分析流程示例 - 面向策略人员"""

import hscredit as hscr
from hscredit.report import FeatureAnalyzer

# 1. 数据加载
X, y = hscr.load_data('strategy_data.csv')

# 2. 快速特征分析
analyzer = FeatureAnalyzer(X, y)

# 2.1 IV值分析
iv_table = analyzer.calculate_iv()
print("高IV特征:", iv_table[iv_table.iv > 0.1].feature.tolist())

# 2.2 PSI分析
psi_table = analyzer.calculate_psi(X_train=X, X_test=X_oot)
print("稳定特征:", psi_table[psi_table.psi < 0.1].feature.tolist())

# 2.3 分箱分析
for feature in ['age', 'income', 'debt_ratio']:
    bin_table = analyzer.analyze_binning(feature, method='optimal_iv')
    print(f"\n{feature}分箱:")
    print(bin_table)
    
    # 可视化
    fig = analyzer.plot_binning(feature)
    fig.savefig(f'{feature}_binning.png')

# 3. 规则提取
from hscredit.core.rules import Rule, ruleset_report

rules = [
    Rule("age < 25 & income < 5000", name="young_low_income"),
    Rule("debt_ratio > 0.8 & credit_score < 600", name="high_debt_bad_credit"),
]

# 规则效果分析
for rule in rules:
    coverage, precision, lift = rule.evaluate(X, y)
    print(f"{rule.name}: 覆盖率={coverage:.2%}, 命中率={precision:.2%}, 提升度={lift:.2f}")

# 规则集报告
ruleset_report(X, y, rules, output='ruleset_report.xlsx')

# 4. 规则置换风险分析
from hscredit.report import SwapAnalyzer

swap_analyzer = SwapAnalyzer()
result = swap_analyzer.analyze(X, y, rules)
print(result.summary())
result.to_excel('swap_analysis.xlsx')
```

---

## 四、依赖管理

### 4.1 核心依赖
```
numpy>=1.19.0
pandas>=1.2.0
scipy>=1.5.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

### 4.2 可选依赖
```
# 调参
optuna>=3.0.0

# SHAP解释
shap>=0.40.0

# 树模型
xgboost>=1.4.0
lightgbm>=3.2.0
catboost>=1.0.0

# 深度学习
torch>=1.8.0
pytorch-tabnet>=3.1

# PPT报告
python-pptx>=0.6.0

# PMML导出
sklearn2pmml>=0.82.0
```

---

## 五、测试策略

### 5.1 单元测试
```python
# tests/unit/test_binning.py
import pytest
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning

class TestOptimalBinning:
    def test_basic_binning(self):
        X = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        
        binner = OptimalBinning(method='optimal_iv', max_n_bins=3)
        binner.fit(X, y)
        
        assert len(binner.get_bin_table('feature')) == 3
        
    def test_monotonic_constraint(self):
        # 测试单调性约束
        pass
        
    def test_user_splits(self):
        # 测试用户指定切分点
        pass
```

### 5.2 集成测试
```python
# tests/integration/test_full_pipeline.py
def test_full_scorecard_pipeline():
    """测试完整评分卡流程"""
    from hscredit import germancredit
    from sklearn.model_selection import train_test_split
    
    X, y = germancredit()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # 完整流程
    # ... (省略具体代码)
    
    # 验证结果
    assert ks > 0.4
    assert auc > 0.7
```

### 5.3 性能测试
```python
# tests/performance/test_binning_speed.py
def test_large_dataset_binning():
    """测试大数据集分箱性能"""
    import time
    
    n_samples = 1000000
    X = pd.Series(np.random.randn(n_samples))
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    start = time.time()
    binner = OptimalBinning(method='tree')
    binner.fit(X, y)
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # 5秒内完成
```

---

## 六、性能优化建议

### 6.1 分箱算法优化
- 使用Numba加速关键计算
- 大数据集采用采样分箱
- 并行分箱处理

### 6.2 特征筛选优化
- 稀疏矩阵优化
- 增量计算IV/PSI
- 分布式筛选支持

### 6.3 模型训练优化
- 早停机制
- GPU加速支持
- 模型缓存

---

## 七、文档计划

### 7.1 API文档
使用sphinx自动生成API文档

### 7.2 使用教程
1. 快速入门 (10分钟上手)
2. 分箱指南
3. 特征筛选指南
4. 建模指南
5. 报告生成指南

### 7.3 示例
- 评分卡建模完整示例
- 策略分析完整示例
- 自定义损失函数示例
- 调参优化示例

---

## 八、发布计划

### 8.1 版本规划
- v0.2.0: 核心功能补全 (调参+SHAP+评分卡)
- v0.3.0: 功能增强 (特征工程+PPT报告+持久化)
- v0.4.0: 优化完善 (性能+文档+测试)
- v1.0.0: 正式版发布

### 8.2 发布检查清单
- [ ] 所有测试通过
- [ ] 文档完整
- [ ] 示例可运行
- [ ] 版本号更新
- [ ] CHANGELOG更新
- [ ] 打包测试
- [ ] PyPI发布
