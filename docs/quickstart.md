# 快速开始

仓库提供了可直接运行的完整示例：

```bash
python examples/00_quickstart.py
```

下面的 Python 代码块按顺序执行即可完成同一条基础流程。示例只使用基础依赖。

## 1. 准备数据并按时间切分

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import hscredit
from hscredit.core.model_selection import time_train_test_split

X_array, y_array = make_classification(
    n_samples=400,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    weights=[0.72, 0.28],
    random_state=42,
)
df = pd.DataFrame(X_array, columns=["age", "income", "debt_ratio", "apply_count"])
df["apply_date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
df["customer_id"] = [f"C{i // 2:04d}" for i in range(len(df))]
df["target"] = y_array

train_df, test_df = time_train_test_split(df, "apply_date", test_size=0.25)
features = ["age", "income", "debt_ratio", "apply_count"]
X_train, y_train = train_df[features], train_df["target"]
X_test, y_test = test_df[features], test_df["target"]
```

## 2. 数据探索

```python
import hscredit.core.eda as eda

summary = eda.data_info(train_df)
iv_result = eda.batch_iv_analysis(train_df, features=features, target="target")
trend = eda.bad_rate_trend(train_df, target_col="target", date_col="apply_date")
```

## 3. 分箱与变量筛选

```python
from hscredit.core.binning import OptimalBinning
from hscredit.core.selectors import IVSelector, VIFSelector, CompositeFeatureSelector

binner = OptimalBinning(method="best_iv", max_n_bins=5)
binner.fit(X_train, y_train)
X_train_woe = binner.transform(X_train, metric="woe")

selector = CompositeFeatureSelector([
    ("iv", IVSelector(threshold=0.0)),
    ("vif", VIFSelector(threshold=20.0)),
])
X_selected = selector.fit_transform(X_train_woe, y_train)
```

## 4. 评分卡建模

```python
from hscredit.core.models import ScoreCard

scorecard = ScoreCard(
    pdo=60,
    rate=2,
    base_odds=35,
    base_score=750,
    binner=binner,
)
scorecard.fit(X_train_woe, y_train)
scores = scorecard.predict(X_test)
```

## 5. 机器学习模型与概率校准

```python
from hscredit.core.models import RandomForestRiskModel
from hscredit.core.models.evaluation import ProbabilityCalibrator

model = RandomForestRiskModel(n_estimators=30, random_state=42)
model.fit(X_train, y_train)
metrics = model.evaluate(X_test, y_test)

calibrator = ProbabilityCalibrator(
    model=model,
    method="platt",
    calib_ratio=None,
).fit(X_test, y_test)
calibrated_proba = calibrator.predict_proba(X_test)
calibration_report = calibrator.report(X_test, y_test)
```

## 6. 策略规则挖掘

```python
from hscredit.report.mining import SingleFeatureRuleMiner, TreeRuleExtractor

single_miner = SingleFeatureRuleMiner(
    target="target",
    method="best_iv",
    max_n_bins=4,
    min_samples=5,
).fit(train_df[features + ["target"]])
single_rules = single_miner.get_top_rules(top_n=5, metric="lift")

extractor = TreeRuleExtractor(max_depth=3, min_samples_leaf=10)
extractor.fit(X_train, y_train)
tree_rules = extractor.extract_rules()
```

## 7. 模型报告和制品保存

```python
from hscredit.report import auto_model_report

report = auto_model_report(
    model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    excel_path="模型评估报告.xlsx",
    verbose=False,
    with_plots=False,
)

model.save_artifact("risk_model.joblib")
restored_model = RandomForestRiskModel.load_artifact("risk_model.joblib")
```

## 8. 规则表达式

```python
from hscredit.core.rules import Rule

rule = Rule("age >= 0") & Rule("income < 1")
rule_report = rule.report(test_df, target="target")
```
