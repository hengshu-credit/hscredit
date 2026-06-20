# 快速开始

下面给出从数据初筛到报告交付的典型工作流。所有组件都兼容 sklearn `Pipeline`，
所有输出列名与报告均为中文。

## 1. 数据和变量初筛

```python
import hscredit
import hscredit.core.eda as eda

# 数据质量、缺失、字段类型、样本分布
summary = eda.data_info(df)

# 批量评估变量 IV 和坏率趋势
iv_result = eda.batch_iv_analysis(df, features=["age", "income"], target="fpd30")
trend = eda.bad_rate_trend(df, target_col="fpd30", date_col="apply_month")

# pandas 扩展方法，适合快速查看数据摘要
df.summary(y="fpd30")
```

## 2. 分箱、编码和变量筛选

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.selectors import IVSelector, VIFSelector, CompositeFeatureSelector

binner = OptimalBinning(method="best_iv", max_n_bins=5, target="fpd30")
binner.fit(train_df)
train_bins = binner.transform(train_df)

encoder = WOEEncoder(target="fpd30")
encoder.fit(train_bins)
train_woe = encoder.transform(train_bins)

selector = CompositeFeatureSelector([
    ("iv", IVSelector(threshold=0.02)),
    ("vif", VIFSelector(threshold=10.0)),
])
selector.fit(X_train, y_train)
X_selected = selector.transform(X_train)
```

`OptimalBinning` 是 17 种分箱方法的统一入口，可自动选择最优方法：

```python
best_method = OptimalBinning.auto_select_method(X, y, "feature_name")
binner = OptimalBinning(method=best_method)
```

## 3. 评分卡建模

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.models import ScoreCard

binner = OptimalBinning(method="best_iv", max_n_bins=5)
binner.fit(X_train, y_train)
X_train_woe = binner.transform(X_train, metric="woe")

scorecard = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750, binner=binner)
scorecard.fit(X_train_woe, y_train)
scores = scorecard.predict(X_test)
```

## 4. 机器学习风控模型

```python
from hscredit.core.models import XGBoostRiskModel, LightGBMRiskModel, CatBoostRiskModel

models = {
    "xgboost": XGBoostRiskModel(max_depth=4, n_estimators=200),
    "lightgbm": LightGBMRiskModel(num_leaves=31, n_estimators=200),
    "catboost": CatBoostRiskModel(depth=5, iterations=200),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, model.evaluate(X_test, y_test))
```

## 5. 策略规则挖掘

```python
from hscredit.report.mining import (
    SingleFeatureRuleMiner,
    MultiFeatureRuleMiner,
    TreeRuleExtractor,
)

single_miner = SingleFeatureRuleMiner(target="fpd30", method="best_iv", max_n_bins=5)
single_miner.fit(train_df)
single_rules = single_miner.get_top_rules(top_n=10, metric="lift")

cross_miner = MultiFeatureRuleMiner(target="fpd30", method="chi", max_n_bins=4)
cross_miner.fit(train_df)
cross_rules = cross_miner.get_cross_rules("age", "income", top_n=10)

extractor = TreeRuleExtractor(algorithm="rf", max_depth=5)
extractor.fit(X_train, y_train)
tree_rules = extractor.extract_rules(top_n=20, metric="confidence")
```

## 6. 模型报告和 Excel 交付

```python
from hscredit.report import auto_model_report

report_path = auto_model_report(
    model,
    X_test,
    y_test,
    save_path="模型评估报告.xlsx",
)
```

借助 pandas 扩展方法可一键导出与展示：

```python
import hscredit  # 注册 df.summary() / df.save() / bin_table.show()

bin_table.save("分箱结果.xlsx", title="年龄分箱")
bin_table.show()
```

## 规则表达式

所有规则统一用 `Rule` 类表达，支持任意层级嵌套与与/或/非逻辑：

```python
from hscredit.core.rules import Rule

rule = Rule("age >= 30") & Rule("income < 5000")
report = rule.report(df, target="fpd30")   # 命中率、坏率、Lift 等指标
```
