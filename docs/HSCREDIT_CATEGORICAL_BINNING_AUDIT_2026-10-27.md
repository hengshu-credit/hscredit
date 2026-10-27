# 类别、数值枚举与未知类别专项审查

> 提交前复核已修复本文 C01-C04，并新增独立回归与真实数据验证。当前结论见 [提交前复核](HSCREDIT_PREPUSH_REVIEW_2026-10-28.md)。下文保留修复前的检查结果，不作为当前缺陷结论。

日期：2026-10-27（Asia/Shanghai）。以下为修复前源码检查和复现记录。

结论：**普通低、中基数枚举已有完整支持框架；数值枚举必须明确类别语义，未知类别也有清晰协议。但当前仍有类型转换、特殊值和规则导入方面的实质缺陷，不能笼统认定类别路径已经稳健。**

修复后验证入口：[Notebook](../examples/31_categorical_binning_audit.ipynb)；修复前聚合证据、源码 SHA256 和运行环境：[JSON](validation/categorical-binning-audit-2026-10-27.json)。

## 1. 当前究竟支持什么

| 输入/场景 | 当前行为 |
|---|---|
| `object`、pandas `string`、`category` | 按类别处理，包括中文字符串和数字形式的字符串 |
| `bool`、nullable `boolean` | 按类别处理 |
| 普通 `int/float`，包括数值枚举编码 | 默认按数值区间处理，不自动推断业务枚举含义 |
| 数值列显式转换为 `category` | 按类别处理；匹配依据是类别身份，不再按编码大小分区 |
| `cat_cutoff=整数` | 唯一值数不超过该阈值的数值列按类别处理 |
| `0 < cat_cutoff < 1` | 按唯一值数/非缺失样本数的比例判断是否作为类别 |
| `category_order` | 为已经识别为类别的字段指定顺序；单独传该参数不会改变数值列类型 |
| 自由文本 | 把整个字符串作为一个枚举值，不做分词、语义理解或相似文本归并 |

类型判断位置：[base.py:2858](../hscredit/core/binning/base.py#L2858)。`cat_cutoff` 是**数值转类别的识别阈值**，不是低频类别合并阈值，而且当前是全局配置。

17 种方法都通过共享适配器处理类别：

流程（Mermaid 源码）：

```text
flowchart LR
    A[训练类别及标签] --> B[确定类别顺序]
    B --> C[类别转有序数值编码]
    C --> D[所选分箱方法决定边界]
    D --> E[还原原始类别分组]
    E --> F[按类别身份匹配预测数据]
```

默认先按训练坏样本率升序排序，同率按首次出现顺序，再交给所选算法。因此数值枚举 10/20/30 的类别顺序可以是 20/30/10，取决于训练风险，而不是编码大小。`uniform/quantile` 在类别路径中也会先使用标签排序，不能把整个类别处理流程视为无监督。

这属于“先排序、再合并相邻类别”的分箱，并不是任意枚举分组组合的全局搜索。`max_n_bins=3` 也不表示保留每个类别独立一箱。

## 2. 数值枚举的关键陷阱

训练取值为 10、20、30；三个值各 50 行，坏数分别为 40、5、25。预测取值为 10、40：

| 配置 | 识别类型 | 预测箱号 |
|---|---|---|
| 默认数值 dtype | numerical | `[0, 2]`：40 按数值区间进入最高端箱 |
| 显式 `category` | categorical | `[2, -3]`：40 是未知类别 |
| `cat_cutoff=3` | categorical | `[2, -3]` |
| 仅指定 `category_order`，仍用数值 dtype | numerical | `[0, 2]`：顺序配置没有使该字段变为类别 |

因此，**数值枚举没有被识别为类别时，`handle_unknown='raise'` 也不是枚举范围校验器**。新的数值编码通常仍能落入 `[-inf,+inf)` 覆盖的数值区间。

有业务顺序的等级需要显式传 `category_order`。当前不会自动采用 `pd.CategoricalDtype(ordered=True)` 的顺序。本次 dtype 声明顺序为 C/A/B，实际默认排序为 A/B/C。顺序参数应传列表或元组；本机直接传 `numpy.ndarray` 被拒绝。

默认类别顺序按坏率升序。如果又强制要求 `monotonic='descending'`，需要提供与该方向相容的类别顺序，否则两项设置可能冲突。

## 3. 训练外类别的协议

| `handle_unknown` | 箱号/标签/WOE |
|---|---|
| `-3`（默认）或 `"value"` | 未知箱 `-3`，标签 `unknown`，WOE 为 `0.0` |
| `"raise"` | 遇到未见类别报错；特征任务内部异常可能带并行包装 |
| 已在训练结果中记录的整数箱号，例如 `0` | 归入该箱，并使用该箱的 WOE；不存在的箱号在拟合时被拒绝 |

缺失值默认是 `-1`，特殊值是 `-2`，与未知类别不同。缺失是否合并受 `missing_separate`/用户规则控制。未知箱通常没有训练观测行，但内部会保留其中性 WOE 映射。

WOE=0 表示该特征不给好坏相对赔率提供偏移，**不代表该客户风险为零**。当前没有 `most_frequent`、`worst` 或自动稀有类 `OTHER` 等未知策略；指定箱号也是全局参数，不是逐字段字典。

### 3.1 “未见过”以实际训练观测为准

即使 pandas dtype 预声明了 10/20/30/40，训练只出现 10/20/30，预测 40 仍会被认作未知类别。这是合理的训练观测协议。

但如果预测前强制套用只有 10/20/30 的 `CategoricalDtype`，pandas 会先把 40 转成缺失；分箱器收到的已经是 NaN，只能按 `-1` 处理，严格未知策略也无法恢复原值。这是上游信息丢失，不是未知策略失效。

### 3.2 匹配严格区分类型

共享比较器会把 NumPy 标量转换为原生标量，然后要求类型和值同时相等。`1`、`1.0`、`"1"`、`True` 在普通类别匹配中可以是不同类别。[比较器](../hscredit/core/binning/_categorical.py#L24)。

这能避免字符串编码和数字编码被随意合并，但也要求训练和预测采用相同 schema。默认不会去空格、统一大小写或把 `"NULL"`/`"nan"` 字符串当成缺失值。

## 4. 当前已复现的四项问题

### C01 · P1 · nullable Int64 含缺失时，已见整数会变成未知

位置：[_typed_mask](../hscredit/core/binning/_categorical.py#L59)。当前实现调用 `Series.map()` 逐值比较。在本次 pandas 2.3.3 环境中，带缺失的 `Int64` Series 虽然 `.tolist()` 给出整数，但 `.map()` 向回调传入的却是浮点数。

训练类别为整数 10/20/30，预测 `[10,40,NA]`：

| 预测表示 | 结果 |
|---|---|
| nullable `Int64` | `[-3,-3,-1]`，已见的 10 被误判未知 |
| `Int64` 再转 `category` | `[2,-3,-1]`，正确区分已见、未知、缺失 |
| `Int64` 再转 `object` | `[2,-3,-1]`，同样正确 |
| `float64` | `[-3,-3,-1]`，此时是实际输入类型与训练整数不一致 |

训练阶段也受影响：`Int64` 枚举含缺失并用 `cat_cutoff=3` 时，120 行样本中 **90 个非缺失训练值全部被转换到未知箱**，剩下 30 行是缺失，拟合没有报错。转成 `category/object` 后正常。

需要区分：真实的 int→float 类型漂移是严格匹配协议的边界；dtype 仍为 `Int64` 却因内部 map 改成浮点，是实现与 pandas 行为不兼容的问题。不能仅用总人数守恒来发现它。

### C02 · P1 · 数值特殊码与内部类别编码发生碰撞

位置：[类别编码](../hscredit/core/binning/_categorical.py#L111)、[适配器](../hscredit/core/binning/base.py#L647)及各数值算法的特殊值过滤。

类别先编码为 0、1、2……，但底层算法仍持有原始 `special_codes`。配置特殊码 0/1 时，它们会误命中普通类别的内部编码。

输入只有字符串 A/B/C/D，没有任何数值 0：

| TreeBinning 配置 | 当前结果 |
|---|---|
| `special_codes=None` | `[['A'],['B'],['C','D']]` |
| `special_codes=[0]` | `[['A','B'],['C'],['D']]` |
| `special_codes=[1]` | `[['A'],['B','C'],['D']]` |
| `special_codes=[-999]` | 与未配置特殊值一致 |

quantile、best_iv、chi 也复现了同类变化。一个训练中不存在的特殊值不应改变结果，因此这不是普通算法差异。最终统计仍可能守恒，但学习阶段实际使用了错误的样本掩码。

### C03 · P1 · 特殊值处理破坏了类型安全匹配

位置：[_special_mask](../hscredit/core/binning/base.py#L2208)、[最终保留箱策略](../hscredit/core/binning/base.py#L2243)。这里采用普通 `Series.eq()`，与类别匹配的类型安全规则不一致。

object 列中前五个值分别为 `int 1 / float 1.0 / bool True / str '1' / str 'A'`：

- 不配特殊值时，能分别进入 5 个箱。
- 配 `special_codes=[1]` 后，前三个值全部变成 `-2`，把 1.0 和 True 一并覆盖。

此外，混合类型值 1 和字符串 `'1'` 的显示标签均可为 `"1"`；索引能区分，纯文本标签不一定能区分，业务输入应尽量统一类型。

### C04 · P1 · 类别 `import_rules(...).fit(...)` 的 WOE 表与转换不一致

位置：[导入时初始化映射](../hscredit/core/binning/base.py#L3354)、[最终保留导入快照](../hscredit/core/binning/base.py#L2468)。

只导入类别规则时，代码先为各组生成 0.0 WOE 占位值；随后 fit 更新了分箱统计表，但快照逻辑把这些占位值当成应保留的已有映射。

| 入口 | 训练表 WOE | 已见类别实际转换 WOE |
|---|---|---|
| `user_splits + user_splits_fixed=True` 后 fit | `[1.519826,-2.063693,0.133531]` | 与表一致 |
| 仅 `import_rules(...).fit(...)` | 同上 | **`[0,0,0]`** |
| 正常模型完整 `export()/load()` | 携带已学习映射 | 正确恢复 |

应区分“用户明确提供的已有 WOE”和“自动生成的占位 WOE”，不能把两者都冻结。修复时也不能反向破坏完整模型导出/载入的映射保留协议。

## 5. 稀有类、高基数与自由文本

当前分箱可以合并类别，但没有独立的低频类别阈值/专用 OTHER 桶，也没有默认的类别坏率平滑。`min_bin_size` 限制的是合并后的箱，不是每个类别本身。因此，小样本类别的极端坏率仍可能影响排序。

每条文本都唯一的合成样本中，训练标签排序可产生完美分离：本次训练 WOE 的 AUC=1.0；预测换成全新的文本后，未知率=100%，WOE 全部为 0，单特征 AUC=0.5。这个反例说明了机制，不是对真实业务效果的评估。自由文本应先做适合业务的特征提取，不能期待类别分箱理解文本语义。

性能也需要关注：唯一类别提取使用逐项类型比较，每个类别又多次扫描整列生成掩码，主要开销接近 `O(N×K + K²)`。本机 `uniform`、1,000 行、单线程的小规模观察：

| 类别数 | 拟合时间 |
|---:|---:|
| 10 | 约 0.15 秒 |
| 100 | 约 1.31 秒 |
| 200 | 约 2.58 秒 |

这些不是百万行容量验收，不能线性外推为生产 SLA。高基数字段需要类型安全的哈希键、向量化统计和映射缓存。

类别排序使用标签，所以所有 fit 都应在训练集/训练折内完成，验证集和预测集只 transform。即使选的是 uniform/quantile，也不能在全量类别数据上先 fit 再划分训练验证集。

## 6. 当前建议的用法与修复优先级

对于业务含义确实为整数编码的枚举，可在训练和预测使用同一个入口：

```python
def prepare_categories(frame):
    result = frame.copy()
    result['职业编码'] = result['职业编码'].astype('Int64').astype('category')
    result['渠道'] = result['渠道'].astype('string')
    return result

binner = OptimalBinning(
    method='tree',
    max_n_bins=5,
    handle_unknown='value',   # 默认 -3；需要严格检查时用 'raise'
    n_jobs=1,
).fit(prepare_categories(X_train), y_train)

indices = binner.transform(prepare_categories(X_test), metric='indices')
woe = binner.transform(prepare_categories(X_test), metric='woe')
```

该示例不适用于必须保留前导零的字符串编码，或本来具有小数含义的类别；这些字段应保持自己的规范化规则。预测端不要直接套用只包含训练观测值的固定 CategoricalDtype，以免提前把新类别变为 NaN。

当前可以采用的原则：

1. 明确标注哪些列是类别，少依赖全局 `cat_cutoff` 猜测混合 DataFrame 的语义。
2. 有序类别同时提供显式 `category_order` 列表；该列表当前必须完整覆盖观测类别，不能预放训练未见的类别。
3. 区分未知与缺失，并观察 `indices == -3` 的比例；映射到已有箱之前确认实际箱号和风险含义。
4. 当前避免在类别路径直接使用会碰撞的数值特殊码；不要认为换成 `legacy` 就能修正类型、掩码或 WOE 问题。
5. 手工类别分组优先使用经本次验证的 `user_splits + fixed + fit`；模型复用使用完整 export/load，暂不依赖有 C04 问题的原始规则导入再拟合路径。

建议先修 C01–C04，再完善：逐字段类型 schema、逐字段未知策略、低频 OTHER 桶/平滑、有序 dtype 处理，以及高基数性能。类型安全性应全链路统一，不能简单把所有类别转字符串或改成普通 `==`，否则又会混淆原本应区分的类别。

## 7. 验证范围

解释器：`E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe`，Python 3.11.5；hscredit 0.1.3 当前仓库源码，pandas 2.3.3、NumPy 2.4.6、scikit-learn 1.6.1、OR-Tools 9.7.2996。没有修改共享环境。

- 真实数据 970 行，FPD 好/坏数为 834/136，商品类别 6 种。
- 17 方法 × 两种类别表示 = 34 组正常路径，每组验证三种未知策略，共 102 次拟合；逐箱原始行计数核验通过。
- 类别/契约范围既有测试 **350 通过、0 失败、0 跳过，15.99 秒**。它们没有覆盖本次四项反例；通过数量不能替代边界正确性。
- 专项 Notebook **8 个代码单元全部执行，0 个执行错误**；源码首尾 SHA256 一致。
- 未重新测量覆盖率，未做百万行容量测试，未修改本次发现的问题。

执行的既有测试为 `tests/test_binning/` 下的 `test_categorical_adapter.py`、`test_categorical_methods.py`、`test_categorical_rules.py`、`test_categorical_binning_complete.py`、`test_binning_contracts.py`，JUnit 记录为 `.audit_tmp/categorical-audit-existing.xml`。
