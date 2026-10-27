# 2026-10-28 提交前复核

复核跨度：2026-10-27 至 2026-10-28（Asia/Shanghai）。范围为提交前全部本地修改：
分箱算法与自定义目标、表格神经网络及调参、逾期标签与报告、手工树兼容、Notebook 和文档。

完整回归 **7,055 通过、0 失败、9 跳过**；最后遗传指标修复后，相关回归 **1,418 项全部通过**。
两组结果有重叠，不相加。当前证据、执行范围和最终库源码指纹保存在
[机器可读复核记录](validation/prepush-review-2026-10-28.json)。

## 本轮发现并修复的问题

| 问题 | 修复与独立验证 |
|---|---|
| nullable Int64 含缺失时，已见整数被内部 `Series.map` 转成浮点后识别为未知类别 | 从原生标量生成类型安全掩码；验证重复索引、已见/未见/缺失和真实商品类别 |
| 原始特殊码 0/1 碰撞类别编码 | 数值算法和后处理中的类别子集不再重复匹配原始特殊码；17 种方法对照、混合数值字段以及 threading/loky 均验证 |
| 特殊值 1 同时覆盖 1.0/True | 类别特殊码与用户组归属统一使用类型安全比较，并规范 NumPy 标量 |
| `import_rules(...).fit(...)` 把零 WOE 占位值冻结 | 仅导入规则时重新学习 WOE；显式导入的已有 WOE 仍保持不变；用原始好坏人数和对数公式核对 |
| 遗传 KS 在纯箱上错位或返回 NaN | 使用包含全部箱的共享统计内核；完全分离样本应为 1，混合纯箱反例应为 0.5 |
| 遗传 Gini 不符合公开接口的 `2 × AUC − 1` | 用箱内坏率作为风险分数；以所有好坏样本对逐对比较、同分半计入独立核算，无区分样本应为 0 |
| 遗传 IV 的零计数平滑与共享指标口径不同 | 复用 `compute_bin_stats` 的计数平滑和未舍入统计；独立计算平滑后的分布及 IV |
| 新增参数未进入 API 参数说明 | 补齐 OptimalBinning.objective、CPSATBinning.custom_objective 和 GeneticBinning 的 custom_objective/woe_clip；60 项公开 API 契约复测通过 |
| 新网络 API 目录层级过深，导航校验遗漏该页面 | 将说明段落改为 rubric，模型及方法恢复第 4/5 层；同步导航校验和测试 |
| 新审计证据被忽略、文档引用本地生成图片 | 只开放所引用聚合证据，静态图片随仓库保存；本地生成 HTML/Graphviz 制品改为明确的生成说明；修复文档格式和页面导航 |

原类别审计的 C01-C04 已修复，修复前 JSON 保留，不能把原审计中的问题描述继续当作当前结论。
[类别 Notebook](../examples/31_categorical_binning_audit.ipynb) 已增加修复后断言并重新执行，
[修复后聚合结果](validation/categorical-binning-recheck-2026-10-27.json) 单独保存。

KS 按两组经验累计分布的最大绝对差计算，定义见 [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)。
排序 Gini 的 `2 × AUC − 1` 关系见 [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。
OR-Tools/CP-SAT 的既有 `gini` 目标是箱纯度评分，已在文档中明确区分；本轮没有擅自改变其优化定义。

## 验证环境与结果

解释器：`E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe`，Python 3.11.5。
NumPy 2.4.6、pandas 2.3.3、scikit-learn 1.6.1、PyTorch 2.5.1+cu124、Optuna 4.9.0。
没有升级共享 Anaconda；构建使用临时隔离环境满足 pyproject 声明的构建后端要求。

| 验证范围 | 实际结果 |
|---|---|
| 首轮完整 `tests/` | 7,007 通过、3 个 API 文档失败、9 跳过；失败项随后修复 |
| C01-C04 独立复现 | 38 项中 35 失败、3 通过，随后修复 |
| 类别修复回归 | 395 通过 |
| 类别修复后的完整 `tests/`，未排除 slow/integration | 7,055 通过、0 失败、9 跳过，947.66 秒 |
| 遗传指标独立反例 | 3 项修复前均失败，修复后通过 |
| 最终全部分箱、并行 API 及 hsbin 分箱器/表格回归 | 1,418 通过、0 失败、0 跳过，126.71 秒 |
| 最终遗传分箱与报告真实数据核算 | 3 个目标 × 4 个字段，共 12 组通过；每组 970 行、136 个坏样本；逐箱人数、坏数和报告总量独立核对 |
| 6 份变更 Notebook | 61 个代码单元全部执行，0 执行错误；包括真实 TabPFN 权重推理、分类/回归、embedding 与保存恢复 |
| Black、关键 Flake8、Makefile 指定 mypy | 595 个 Black 检查文件通过；E9/F63/F7/F82/F601 通过；3 个基础设施模块类型检查通过 |
| Python 3.9 语法 | 89 个变更 Python 文件、8 份 Notebook 通过 AST 检查；实际运行版本仍为 3.11.5 |
| 文档 | Sphinx `-W --keep-going` 全文档构建通过、0 警告；导航与搜索产物校验通过；内部 superpowers 历史设计草稿不作为发布页面 |
| 构建 | sdist 与 wheel 均构建成功，twine check 均通过；wheel 中 280 个库 Python 文件与最终源码逐字节一致，未打包私有验证工作簿 |

完整回归之后只有三个库文件改变：`genetic_binning.py` 修复上述原生指标，
`cp_sat_binning.py` 与 `or_binning.py` 澄清 Gini 文档。其余 **277 / 280 个库文件指纹一致**。
最后修改后执行了 1,418 项相关回归和真实数据核算，没有再次运行完整 `tests/`。
期间主动终止的一轮测试不计作成功验证。

9 项跳过为未配置真实数据库的既有集成用例。完整回归有 643 条警告，最终分箱回归有 18 条警告，
未将测试通过描述为无警告；本轮未重新测量测试覆盖率。

## 复现

```powershell
$env:PYTHONUTF8 = '1'
$env:MPLBACKEND = 'Agg'
$reviewPython = '.\.audit_tmp\defect-repairs\runtime\Scripts\python.exe'
& $reviewPython -m pytest tests/ -q --tb=short --junitxml=.audit_tmp/prepush-review-verified.xml
& $reviewPython -m pytest tests/test_binning/ tests/test_parallel_api_contract.py tests/test_skills/test_hsbin_binners.py tests/test_skills/test_hsbin_tables.py -q --tb=short --junitxml=.audit_tmp/prepush-binning-final.xml
& $reviewPython -m sphinx -b html -W --keep-going docs .audit_tmp/prepush-review-docs
& $reviewPython scripts/validate_docs_build.py .audit_tmp/prepush-review-docs
& $reviewPython -m build --outdir .audit_tmp/prepush-dist-verified
& $reviewPython -m twine check .audit_tmp/prepush-dist-verified/*
```

Notebook 使用相同验收解释器的内核，执行副本和日志保留在 `.audit_tmp/prepush-notebooks/`。
为保留历史审计，执行时只把聚合证据输出目录重定向到临时目录，没有改变验证数据或断言。
