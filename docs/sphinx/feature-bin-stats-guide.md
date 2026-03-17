# Feature Bin Stats 功能使用指南

`feature_bin_stats` 是一个强大的特征分箱统计分析工具，参考 scorecardpipeline (scp) 实现，支持单特征/多特征、单目标/多逾期标签组合分析。

## 功能特点

- **单特征分析**: 快速查看单个特征的分箱效果
- **多特征批量分析**: 支持同时分析多个特征
- **多逾期标签组合**: 支持不同 MOB 和 DPD 组合的对比分析
- **多级表头展示**: 多目标分析时自动合并为多级表头
- **自定义分箱规则**: 支持传入自定义切分点
- **IV值汇总**: 快速评估多个特征的预测能力

## API 介绍

### `feature_bin_stats` 函数

```python
feature_bin_stats(
    data: pd.DataFrame,                    # 数据集
    feature: Union[str, List[str]],        # 特征名称或列表
    target: Optional[str] = None,          # 二分类目标变量
    overdue: Optional[Union[str, List[str]]] = None,  # 逾期天数字段
    dpd: Optional[Union[int, List[int]]] = None,      # 逾期定义天数
    rules: Optional[Union[List, Dict]] = None,        # 自定义分箱规则
    method: str = 'mdlp',                  # 分箱方法
    desc: Optional[Union[str, Dict]] = None,          # 特征描述
    binner: Optional[BaseBinning] = None,  # 预训练分箱器
    max_n_bins: int = 5,                   # 最大分箱数
    min_bin_size: float = 0.05,            # 最小样本占比
    missing_separate: bool = True,         # 缺失值单独分箱
    return_cols: Optional[List[str]] = None,          # 指定返回列
    return_rules: bool = False,            # 是否返回分箱规则
    del_grey: bool = False,                # 是否删除灰样本
    verbose: int = 0                       # 详细程度
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]
```

### `FeatureAnalyzer` 类

```python
analyzer = FeatureAnalyzer(
    method: str = 'mdlp',
    max_n_bins: int = 5,
    min_bin_size: float = 0.05,
    missing_separate: bool = True
)

# 批量分析
analyzer.analyze(data, features, overdue, dpd, ...)

# 多维度对比
analyzer.compare_targets(data, feature, overdue_list, dpd_list, ...)

# IV值汇总
analyzer.get_iv_summary(data, features, target)
```

## 使用示例

### 示例 1: 单特征单目标分析

```python
from hscredit import feature_bin_stats, FeatureAnalyzer

# 基础分箱分析
table = feature_bin_stats(
    data=data,
    feature='credit_score',
    target='target',  # 二分类目标变量
    method='mdlp',
    desc='信用评分',
    max_n_bins=5,
    return_cols=['样本总数', '坏样本率', 'LIFT值', '指标IV值']
)
print(table)
```

输出:
```
    指标名称  指标含义   分箱                                     分箱标签  样本总数      坏样本率     指标IV值     LIFT值
0  credit_score  信用评分 -1.0                                       缺失   100      0.180000  0.427259  0.927835
1  credit_score  信用评分  0.0               (-inf, 420.97]   326      0.500000  0.427259  2.577320
2  credit_score  信用评分  1.0   (420.97, 518.07]   939      0.293930  0.427259  1.515102
3  credit_score  信用评分  2.0    (518.07, 613.31]  1641      0.189519  0.427259  0.976900
4  credit_score  信用评分  3.0                 (613.31, +inf)  1994      0.101304  0.427259  0.522185
```

### 示例 2: 多逾期标签组合分析

```python
# 同时分析 MOB1 DPD0+、MOB1 DPD7+、MOB3 DPD0+、MOB3 DPD7+
table = feature_bin_stats(
    data=data,
    feature='credit_score',
    overdue=['MOB1', 'MOB3'],  # 多个逾期字段
    dpd=[0, 7],                # 多个逾期定义
    method='mdlp',
    desc='信用评分',
    max_n_bins=5,
    return_cols=['坏样本率', 'LIFT值', '分档KS值']
)
print(table)
```

输出多级表头:
```
    分箱详情                                                     MOB1_0+                                             MOB1_7+                                             MOB3_0+                                             MOB3_7+
    指标名称  指标含义   分箱         分箱标签    坏样本率     LIFT值   分档KS值    坏样本率     LIFT值   分档KS值    坏样本率     LIFT值   分档KS值    坏样本率     LIFT值   分档KS值
0  credit_score  信用评分 -1.0       缺失  0.180000  0.927835  0.00179  0.130000  0.879567  0.00283  0.190000  1.214834  0.00509  0.150000  1.243781  0.00554
...
```

### 示例 3: 多特征批量分析

```python
# 同时分析多个特征
table = feature_bin_stats(
    data=data,
    feature=['score', 'age', 'income'],
    overdue='MOB1',
    dpd=0,
    method='mdlp',
    desc={'score': '信用评分', 'age': '年龄', 'income': '收入'},
    max_n_bins=5
)
print(table)
```

### 示例 4: 使用自定义分箱规则

```python
# 使用自定义切分点
table = feature_bin_stats(
    data=data,
    feature='credit_score',
    overdue='MOB1',
    dpd=7,
    rules=[500, 600, 700],  # 自定义切分点
    desc='信用评分（自定义分箱）',
    return_cols=['样本总数', '坏样本率', 'LIFT值']
)
```

### 示例 5: 删除灰样本

```python
# 删除逾期天数 (0, dpd] 的灰样本，只保留好样本和明确逾期样本
table = feature_bin_stats(
    data=data,
    feature='credit_score',
    overdue='MOB1',
    dpd=7,
    method='mdlp',
    desc='信用评分',
    del_grey=True,  # 删除灰样本
    max_n_bins=5
)
```

### 示例 6: IV值汇总分析

```python
from hscredit import FeatureAnalyzer

analyzer = FeatureAnalyzer(method='mdlp', max_n_bins=5)

# 批量计算IV值
iv_summary = analyzer.get_iv_summary(
    data=data,
    features=['score', 'age', 'income', 'debt_ratio'],
    target='target'
)
print(iv_summary)
```

输出:
```
     特征名称       IV值  分箱数    预测力
0       score  0.427259    5   强预测力
1         age  0.252338    5  中等预测力
2  debt_ratio  0.180000    4  中等预测力
3      income  0.000000    1   无预测力
```

### 示例 7: 返回分箱规则

```python
# 获取分箱统计表和分箱规则
table, rules = feature_bin_stats(
    data=data,
    feature='credit_score',
    overdue='MOB1',
    dpd=0,
    method='mdlp',
    desc='信用评分',
    max_n_bins=5,
    return_rules=True
)

print("分箱规则 (切分点):")
print(f"  credit_score: {rules['credit_score']}")
# 输出: credit_score: [420.97, 518.07, 613.31]
```

## 分箱方法选择

| 方法 | 说明 | 适用场景 |
|-----|------|---------|
| `mdlp` | MDLP算法，基于信息论 | 推荐，自动确定最优分箱数 |
| `optimal` | 最优分箱 | 追求最高IV值 |
| `quantile` | 等频分箱 | 保持每箱样本数相等 |
| `uniform` | 等距分箱 | 保持每箱范围相等 |
| `kmeans` | KMeans聚类分箱 | 基于聚类中心分箱 |

## 返回列说明

分箱统计表包含以下列:

| 列名 | 说明 |
|-----|------|
| 指标名称 | 特征名称 |
| 指标含义 | 特征描述 |
| 分箱 | 分箱索引 |
| 分箱标签 | 分箱区间标签 |
| 样本总数 | 每箱样本数 |
| 样本占比 | 每箱样本占比 |
| 好样本数 | 好样本数量 |
| 坏样本数 | 坏样本数量 |
| 好样本占比 | 好样本占比 |
| 坏样本占比 | 坏样本占比 |
| 坏样本率 | 每箱坏样本率 |
| 分档WOE值 | WOE值 |
| 分档IV值 | 每箱IV贡献 |
| 指标IV值 | 总IV值 |
| LIFT值 | Lift值 |
| 坏账改善 | 坏账改善率 |
| 累积LIFT值 | 累积Lift值 |
| 累积坏账改善 | 累积坏账改善率 |
| 分档KS值 | KS统计量 |

## IV值预测力评估标准

| IV值范围 | 预测力 |
|---------|-------|
| < 0.02 | 无预测力 |
| 0.02 - 0.1 | 弱预测力 |
| 0.1 - 0.3 | 中等预测力 |
| 0.3 - 0.5 | 强预测力 |
| > 0.5 | 超强预测力(需检查) |

## 注意事项

1. **目标变量类型**: 当使用 `target` 参数时，必须是二分类变量 (0/1)
2. **逾期分析**: 使用 `overdue` + `dpd` 参数时，会自动将逾期天数转换为二分类
3. **灰样本处理**: `del_grey=True` 会删除逾期天数在 (0, dpd] 区间的样本
4. **多级表头**: 多目标分析时会自动使用 MultiIndex 列名
5. **缺失值**: `missing_separate=True` 会将缺失值单独分为一箱

## 完整示例代码

参考 `examples/04_feature_bin_stats_demo.py`
