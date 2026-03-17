# 快速开始

本教程帮助你快速上手hscredit，展示最常用的功能。

## 5分钟入门

### 1. 创建Excel报告

```python
import pandas as pd
from hscredit.report.excel import ExcelWriter

# 创建测试数据
df = pd.DataFrame({
    '特征': ['年龄', '收入', '负债率'],
    'IV值': [0.15, 0.23, 0.31],
    '分箱数': [5, 6, 4]
})

# 创建Excel写入器
writer = ExcelWriter(theme_color='3f1dba')

# 获取工作表
ws = writer.get_sheet_by_name("Sheet")

# 写入标题
writer.insert_value2sheet(ws, "B2", "特征筛选报告", style="header")

# 写入数据
writer.insert_df2sheet(ws, df, (4, 2), fill=True)

# 保存文件
writer.save("feature_selection_report.xlsx")

print("✅ 报告已生成: feature_selection_report.xlsx")
```

### 2. 使用自定义损失函数

```python
import numpy as np
from hscredit.core.models import FocalLoss

# 创建测试数据
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0.1, 0.9, 0.8, 0.2, 0.7])

# 使用Focal Loss处理不平衡数据
loss_fn = FocalLoss(alpha=0.75, gamma=2.0)
loss_value = loss_fn(y_true, y_pred)

print(f"Focal Loss: {loss_value:.4f}")
```

### 3. 在XGBoost中使用

```python
import xgboost as xgb
from hscredit.core.models import FocalLoss, XGBoostLossAdapter

# 创建损失函数和适配器
focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
adapter = XGBoostLossAdapter(focal_loss)

# 训练模型
model = xgb.XGBClassifier(
    objective=adapter.objective,
    eval_metric=adapter.metric,
    n_estimators=100
)

model.fit(X_train, y_train)
```

## 完整示例

### Excel报告生成

```python
from hscredit.report.excel import ExcelWriter
import pandas as pd
import numpy as np

# 创建多个数据表
df_model = pd.DataFrame({
    '模型': ['LR', 'XGBoost', 'LightGBM'],
    'AUC': [0.75, 0.82, 0.81],
    'KS': [0.45, 0.52, 0.51],
    'PSI': [0.02, 0.03, 0.02]
})

df_features = pd.DataFrame({
    '特征': ['age', 'income', 'debt_ratio', 'credit_history'],
    'IV': [0.15, 0.23, 0.31, 0.18],
    '重要性': [0.12, 0.18, 0.25, 0.14]
})

# 创建报告
writer = ExcelWriter()

# 第一页：模型评估
ws1 = writer.get_sheet_by_name("模型评估")
writer.insert_value2sheet(ws1, "B2", "模型评估报告", style="header")
writer.insert_df2sheet(ws1, df_model, (4, 2), fill=True)

# 第二页：特征分析
ws2 = writer.create_sheet("特征分析")
writer.insert_value2sheet(ws2, "B2", "特征重要性分析", style="header")
writer.insert_df2sheet(ws2, df_features, (4, 2), fill=True)

# 保存
writer.save("model_report.xlsx")
```

### 损失函数对比

```python
from hscredit.core.models import (
    FocalLoss,
    WeightedBCELoss,
    CostSensitiveLoss,
    BadDebtLoss
)
import numpy as np

# 测试数据
y_true = np.array([0, 0, 1, 1, 1, 0])
y_pred = np.array([0.1, 0.3, 0.8, 0.9, 0.7, 0.2])

# 对比不同损失函数
losses = {
    'Focal Loss': FocalLoss(alpha=0.75, gamma=2.0),
    'Weighted BCE': WeightedBCELoss(pos_weight=2.0),
    'Cost Sensitive': CostSensitiveLoss(cost_fp=1.0, cost_fn=5.0),
    'Bad Debt Loss': BadDebtLoss(bad_debt_rate=0.15)
}

print("损失函数对比:")
print("-" * 40)
for name, loss_fn in losses.items():
    loss_value = loss_fn(y_true, y_pred)
    print(f"{name:20s}: {loss_value:.4f}")
```

## 下一步学习

- {doc}`user_guide/excel_writer` - Excel报告详细教程
- {doc}`user_guide/losses` - 自定义损失函数详细教程
- {doc}`examples/index` - 更多示例代码
- {doc}`api/report` - API参考文档

## 常用功能速查

### Excel写入

```python
from hscredit.report.excel import ExcelWriter

writer = ExcelWriter()
ws = writer.get_sheet_by_name("Sheet")

# 写入DataFrame
writer.insert_df2sheet(ws, df, (1, 1), fill=True)

# 写入单个值
writer.insert_value2sheet(ws, "A1", "标题", style="header")

# 保存
writer.save("output.xlsx")
```

### 损失函数

```python
from hscredit.core.models import FocalLoss

# 创建损失函数
loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

# 计算
loss = loss_fn(y_true, y_pred)

# 梯度（用于优化）
grad, hess = loss_fn.gradient_hessian(y_true, y_pred)
```

### 框架适配

```python
from hscredit.core.models import (
    FocalLoss,
    XGBoostLossAdapter,
    LightGBMLossAdapter
)

# XGBoost
focal_loss = FocalLoss()
xgb_adapter = XGBoostLossAdapter(focal_loss)

# LightGBM
lgb_adapter = LightGBMLossAdapter(focal_loss)
```
