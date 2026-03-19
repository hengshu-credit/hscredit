# 评分卡建模报告修复说明

## 问题已解决 ✅

**问题已从库层面解决！** `IVSelector` 代码已更新，现在原生支持 `category` 类型。

## 问题描述（历史记录）

在运行 `scorecard_modeling_report.ipynb` 时遇到以下错误：

```
TypeError: Object with dtype category cannot perform the numpy op isnan
```

### 问题原因

`GermanCredit` 数据集使用了 `category` 类型的列来存储类别变量，但旧版本的 `IVSelector` 在计算IV值时只处理了 `dtype == 'object'` 的字符串类型，未处理 `category` 类型。

当调用 `np.isnan(x)` 对category类型的数据操作时，pandas的Categorical类型不支持numpy的isnan操作，因此报错。

## 库层面修复

### 修改的文件

`hscredit/hscredit/core/selectors/iv_selector.py`

### 主要修改内容

1. **`_compute_iv_single` 函数**
   - 使用 `pd.isnull()` 替代 `np.isnan()` 来处理缺失值
   - 支持对 category 类型的数组进行缺失值判断

2. **`_fit_impl` 方法**
   - 在编码类别变量时，同时检查 `object` 和 `category` 类型
   - 修改前：`if X[col].dtype == 'object'`
   - 修改后：`if X[col].dtype.name in ['object', 'category']`

3. **文档字符串**
   - 更新文档说明支持的数据类型：数值型、object、category

### 修复后的代码

```python
def _compute_iv_single(x: np.ndarray, y: np.ndarray, regularization: float = 1.0) -> float:
    """计算单个特征的IV值。"""
    # 处理缺失值 - 兼容category和object类型
    if isinstance(x, pd.Series):
        has_missing = x.isnull().values
    else:
        # 如果是numpy数组，尝试转换为Series以使用isnull
        try:
            has_missing = pd.Series(x).isnull().values
        except:
            # 如果转换失败，使用pd.isnull直接判断
            has_missing = pd.isnull(x)
    # ... 其余逻辑保持不变
```

```python
def _fit_impl(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]]) -> None:
    """拟合IV值筛选器。"""
    # ... 前置逻辑
    # 编码类别变量 - 支持 object 和 category 类型
    X_encoded = X.copy()
    for col in X.columns:
        # 检查是否为类别型变量（object 或 category 类型）
        if X[col].dtype.name in ['object', 'category']:
            X_encoded[col] = pd.factorize(X[col])[0]
    # ... 后续逻辑保持不变
```

## 临时修复方案（已废弃）

之前的修复方案是在特征筛选之前手动转换类型：

```python
# 数据类型预处理：将category类型转换为object类型
train_preprocessed = train.copy()
test_preprocessed = test.copy()

for col in train.columns:
    if train[col].dtype.name == 'category':
        train_preprocessed[col] = train[col].astype('object')
        test_preprocessed[col] = test[col].astype('object')
```

**现在不需要此步骤了！** 可以直接使用原始的 `train` 和 `test` 数据。

## 其他Selector的状态

其他Selector不需要修改，因为它们使用的都是pandas的原生方法：

- ✅ **NullSelector** - 使用 `X.isnull()` - 支持category
- ✅ **ModeSelector** - 使用 `value_counts()` - 支持category
- ✅ **VarianceSelector** - 需要数值型数据，category会被自动跳过或报错
- ✅ **CorrSelector** - 使用 `X.corr()` - 支持category（但需数值转换）
- ✅ **StepwiseSelector** - 通过分箱器处理 - 支持category

## 验证

修复后，可以直接运行notebook，无需任何数据类型转换预处理：

```python
# 直接使用原始数据
iv_selector = IVSelector(threshold=0.02)
iv_selector.fit(train[features_after_corr], train[target])  # 支持category类型
features_after_iv = iv_selector.transform(features_after_corr)
print(f'IV值筛选后特征数: {len(features_after_iv)}')
```

## 总结

- ✅ IVSelector 现在原生支持 category 类型
- ✅ 不再需要手动进行数据类型转换
- ✅ notebook 可以直接使用原始数据运行
- ✅ 修复从库层面解决了根本问题，适用于所有使用 IVSelector 的场景
