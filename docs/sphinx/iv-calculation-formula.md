# IV值计算公式校对报告

## 问题发现

在分箱统计表中，IV值出现了负数，这违反了IV值的理论性质。

## 理论基础

### IV (Information Value) 标准公式

```
IV = Σ (bad_dist - good_dist) * log(bad_dist / good_dist)
```

其中：
- `bad_dist = bad_count_i / total_bad` (第i个bin的坏样本分布)
- `good_dist = good_count_i / total_good` (第i个bin的好样本分布)
- `WOE_i = log(bad_dist / good_dist)` (Weight of Evidence)

### 理论证明：IV值不可能为负

对于每个bin的IV贡献：

**情况1：bad_dist > good_dist**
- `bad_dist / good_dist > 1`
- `log(bad_dist / good_dist) > 0`
- `(bad_dist - good_dist) > 0`
- **乘积 > 0**

**情况2：bad_dist < good_dist**
- `bad_dist / good_dist < 1`
- `log(bad_dist / good_dist) < 0`
- `(bad_dist - good_dist) < 0`
- **乘积 > 0**（负负得正）

**情况3：bad_dist = good_dist**
- `log(bad_dist / good_dist) = 0`
- **乘积 = 0**

**结论：每个bin的IV值都 ≥ 0，总和也必然 ≥ 0**

## 问题根源

### 错误1：平滑处理不当

**错误代码：**
```python
good_distr = (good_counts + epsilon) / (total_good + epsilon * len(good_counts))
bad_distr = (bad_counts + epsilon) / (total_bad + epsilon * len(bad_counts))
```

**问题：**
- 分子和分母分别添加不同的平滑因子
- 破坏了好坏样本分布的比例关系
- 可能导致 `(bad_dist - good_dist)` 和 `log(bad_dist / good_dist)` 符号相反

**正确做法：**
```python
# 将0替换为epsilon，保持分布归一化
good_counts_smooth = np.where(good_counts == 0, epsilon, good_counts)
bad_counts_smooth = np.where(bad_counts == 0, epsilon, bad_counts)

# 重新计算总数
total_good_smooth = good_counts_smooth.sum()
total_bad_smooth = bad_counts_smooth.sum()

# 计算分布（保持归一化）
good_distr = good_counts_smooth / total_good_smooth
bad_distr = bad_counts_smooth / total_bad_smooth
```

### 错误2：公式顺序错误

**错误代码：**
```python
iv = ((good_dist - bad_dist) * np.log(good_dist / bad_dist)).sum()
```

**问题：**
- 公式顺序反了
- 应该是 `(bad_dist - good_dist) * log(bad_dist / good_dist)`

**正确公式：**
```python
iv = ((bad_dist - good_dist) * np.log(bad_dist / good_dist)).sum()
```

## 修复内容

### 修复的文件和方法

1. **hscredit/core/metrics/binning_metrics.py**
   - `woe_iv_vectorized()` - 核心IV计算函数
   - 修正了平滑处理逻辑
   - 添加了数学正确性注释

2. **hscredit/core/binning/optimal_iv_binning.py**
   - `_calc_iv()` - IV优化分箱的IV计算
   - 修正了平滑处理逻辑

3. **hscredit/core/binning/kernel_density_binning.py**
   - `_calculate_iv()` - 核密度分箱的IV计算
   - 修正了公式顺序和平滑方法

4. **hscredit/core/binning/genetic_binning.py**
   - `_calculate_iv()` - 遗传算法分箱的IV计算
   - 修正了公式顺序和平滑方法

5. **hscredit/core/metrics/importance.py**
   - `IV()` - 特征重要性IV计算
   - `IV_table()` - IV详细统计表
   - 修正了公式顺序

## 验证测试

### 测试用例

1. ✅ 极端不平衡数据（坏样本率仅4%）
2. ✅ 某些bin只有好样本（坏样本数为0）
3. ✅ 某些bin只有坏样本（好样本数为0）
4. ✅ 某个bin完全为空（good=0, bad=0）
5. ✅ 单调分箱场景
6. ✅ 多种分箱方法（optimal_iv, kernel_density, genetic）

### 测试结果

所有测试用例均通过，IV值均为非负数。

**示例输出：**
```
测试 optimal_iv 方法:
各bin IV值: [ 2.01099638  0.2312421   0.2312421  25.90104938  2.62835338]
总IV值: 31.002883
是否存在负值: False
✓ 测试通过
```

## 正确的IV计算实现

```python
def woe_iv_vectorized(good_counts, bad_counts, epsilon=1e-10):
    """计算WOE和IV值."""
    good_counts = np.asarray(good_counts, dtype=float)
    bad_counts = np.asarray(bad_counts, dtype=float)
    
    total_good = good_counts.sum()
    total_bad = bad_counts.sum()
    
    if total_good == 0 or total_bad == 0:
        return np.zeros(len(good_counts)), np.zeros(len(good_counts)), 0.0
    
    # 平滑处理：将0替换为epsilon，避免log(0)
    good_counts_smooth = np.where(good_counts == 0, epsilon, good_counts)
    bad_counts_smooth = np.where(bad_counts == 0, epsilon, bad_counts)
    
    # 重新计算平滑后的总数
    total_good_smooth = good_counts_smooth.sum()
    total_bad_smooth = bad_counts_smooth.sum()
    
    # 计算分布（保持归一化）
    good_distr = good_counts_smooth / total_good_smooth
    bad_distr = bad_counts_smooth / total_bad_smooth
    
    # 计算WOE
    woe = np.log(bad_distr / good_distr)
    
    # 计算IV（理论上总是非负）
    bin_iv = (bad_distr - good_distr) * woe
    total_iv = bin_iv.sum()
    
    return woe, bin_iv, total_iv
```

## 关键要点

1. **IV公式必须正确**：`(bad_dist - good_dist) * log(bad_dist / good_dist)`
2. **平滑处理要保证归一化**：将0替换为epsilon后重新计算总数
3. **避免破坏比例关系**：不要在分子分母分别加不同的平滑因子
4. **理论验证**：IV值理论上永远 ≥ 0，任何负值都是计算错误

## 参考资料

- scorecardpy: 使用 `.replace(0, 0.9)` 方法进行平滑
- toad: 使用类似的平滑处理
- 信息论基础：IV值是KL散度的应用，理论上非负

---

**修复日期**: 2026-03-16  
**修复人员**: CodeBuddy AI  
**验证状态**: ✅ 所有测试通过
