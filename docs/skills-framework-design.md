# hscredit Agent Skills 框架设计

> 状态：第一实现阶段已完成并验证；已交付 `hsbin` 与 `hsreport`。

## 1. 状态与范围

本文档定义 hscredit 首期 Agent Skills 框架。目标是在仓库根目录提供可被 Codex、其他支持 `SKILL.md` 的 Agent，以及普通 Chat 工具适配器发现和调用的技能，同时继续以现有 hscredit Python API 作为唯一计算实现。

总体框架规划八个可发现 Skill：

- `hscredit`：统一入口和路由 Skill。
- `hsbin`：分箱、分箱统计和分箱图。
- `hsreport`：完整 Excel 分析报告。
- `hsmodel`：模型、调参、损失函数、评分卡和规则模型。
- `hsrule`：表达式规则、规则流和规则分析。
- `hsselect`：特征筛选。
- `hsviz`：全部公开可视化能力，包括分箱图。
- `hsexcel`：Excel 组装和交付。

第一实现阶段只实现 `skills/hsbin` 与 `skills/hsreport`，以及这两个 Skill 运行所需的公共框架。`hscredit`、`hsmodel`、`hsrule`、`hsselect`、`hsviz` 和 `hsexcel` 创建为仅含 `.gitkeep` 的占位目录，并进入 `skills/ROADMAP.md`；这些目录不得包含 `SKILL.md`、Schema、运行脚本，也不得以可调用操作的形式暴露。其他 hscredit 能力同样进入规划清单。

## 2. 目标

1. Agent 能依据简短、可区分的 Skill 描述选择正确领域能力。
2. Agent 能把文件或进程内对象组装成统一请求，并调用现有 hscredit API。
3. 表格、图片、模型和 Excel 报告以文件产物交付，Chat 响应只携带摘要和受限预览。
4. 每个 Skill 可以从 GitHub 独立安装，不依赖相邻 Skill 目录。
5. 缺少可选依赖时默认在隔离环境自动安装；只有用户明确要求时才修改当前代码环境。
6. 所有公开调用经过操作白名单、输入校验、中文错误和可重复验证。

## 3. 非目标

- 不重写分箱、建模、规则、绘图或 Excel 业务逻辑。
- 不提供任意 Python、任意 import、任意 pip 包名或任意 shell 命令执行。
- 不把大表、训练样本或完整模型内容塞入 Agent 上下文。
- 不自动上传 Skill、提交代码、推送仓库或发布 PyPI 包。
- 不承诺所有 Agent 产品使用相同的本地发现目录；产品差异由安装适配层处理。
- 不在本期把未列入首期目录的 hscredit API 包装成通用反射调用。

## 4. 命名与安装单位

对外套件名使用 `hscredit`。每个子 Skill 保持短名称，以支持精确直接调用：

```text
$hscredit
$hsbin
$hsreport
$hsmodel
$hsrule
$hsselect
$hsviz
$hsexcel
```

`hscredit` 是轻量路由 Skill。它根据任务选择领域操作，但不复制领域说明。用户安装整套后既可从 `$hscredit` 开始，也可直接调用子 Skill。

在线安装以 GitHub 仓库路径为基本单位。支持一次安装全部八个目录，也支持只安装部分目录。公开仓库使用直接下载；私有仓库由宿主 Agent 使用已有 Git 凭证或授权令牌。

OpenAI 项目 Skills API 的发布单位是单个 Skill 目录或 ZIP。因此后续发布工具应按八个 Skill 分别创建或更新远端 Skill，并记录远端 Skill ID 与版本；`hscredit` 仍作为统一入口名称。远端发布工具不属于首期实现。

## 5. 目录结构

AI 可发现内容位于仓库根目录：

```text
skills/
├── CATALOG.md
├── ROADMAP.md
├── install.py
├── hscredit/
│   └── .gitkeep
├── hsbin/
├── hsreport/
├── hsmodel/
│   └── .gitkeep
├── hsrule/
│   └── .gitkeep
├── hsselect/
│   └── .gitkeep
├── hsviz/
│   └── .gitkeep
└── hsexcel/
    └── .gitkeep
```

其余六个目录只用于固定规划结构。对应能力实际实现时删除 `.gitkeep`，再加入完整 Skill 文件。

每个 Skill 目录自包含以下文件：

```text
<skill>/
├── SKILL.md
├── runtime.json
├── agents/
│   └── openai.yaml
├── references/
│   └── operations.md
├── schemas/
│   └── request.schema.json
└── scripts/
    └── run.py
```

约束：

- `skills/` 不含 `__init__.py`，避免发布通用顶层 Python 包。
- `SKILL.md` 包含发现描述、选择标准、最短调用流程和关键安全边界。
- `references/operations.md` 记录本 Skill 的操作、必要输入、产物和真实 hscredit API 映射。
- `schemas/request.schema.json` 约束公共信封和本 Skill 的操作参数。
- `runtime.json` 声明 hscredit 来源、兼容版本、操作所需 extras 和运行时协议版本。
- `scripts/run.py` 只使用 Python 标准库完成引导，再调用已安装的公共运行时。
- 每个 Skill 目录不得引用 `../` 下的运行时代码、Schema 或说明文件。

第一实现阶段的公共执行实现位于正常的 hscredit Python 命名空间：

```text
hscredit/skills_runtime/
├── __init__.py
├── __main__.py
├── artifacts.py
├── bootstrap.py
├── contracts.py
├── dependencies.py
├── errors.py
├── io.py
├── objects.py
├── registry.py
└── operations/
    ├── binning.py
    ├── reports.py
    └── visualization.py
```

公共运行时负责真实参数装配和 API 调用。根目录 Skill 负责发现、说明、请求 Schema 和隔离环境引导。

## 6. Skill 能力边界

### 6.1 hsbin

首期登记以下 hscredit 能力：

- `feature_bin_stats`
- `feature_efficiency_analysis`
- `benchmark_binning_methods`
- `feature_binning_summary`
- `feature_group_binning_summary`
- `OptimalBinning` 的创建、拟合、转换、分箱表和 artifact 生命周期
- `OptimalBinning2D` 的创建、拟合、转换、分箱表和 artifact 生命周期
- `bin_plot`
- `bin_trend_plot`
- `bin_overdues_plot`
- `bin_2d_plot`

函数操作使用原函数名作为 `operation`。类操作使用稳定动作名，例如 `optimal_binning_fit`、`optimal_binning_transform`、`optimal_binning_2d_fit` 和 `optimal_binning_2d_transform`。

### 6.2 hsreport

首期登记三个完整 Excel 报告入口：

- `auto_feature_analysis`
- `auto_model_report`
- `swap_out_report`

适配器把统一输出配置映射到现有 `excel_writer`、`excel_path`、`save` 和图片目录参数。报告仍由现有 hscredit 报告代码生成。

### 6.3 hsmodel

首期支持：

- `hscredit.core.models` 当前公开模型类的查询、创建、拟合、预测和 artifact 保存。
- `AutoTuner.create`、`ModelTuner` 和模型实例的 `tune` 工作流。
- 当前公开 losses、metrics 和框架 loss adapters 的创建。
- `ScoreCard`、`RoundScoreCard`、`ProbabilityScoreCard` 和评分转换器工作流。
- `RuleSet`、`RulesClassifier`、`LogicOperator`、`create_and_ruleset`、`create_or_ruleset` 和 `combine_rules`。

模型名、Loss 名、调参器和规则模型组件必须来自注册表白名单。构造参数允许使用 JSON 对象，但未知类名、未知入口和不可序列化参数必须拒绝。

### 6.4 hsrule

首期支持：

- `Rule`
- `RuleFlow`
- `get_columns_from_query`
- `optimize_expr`
- `beautify_expr`
- `get_expr_variables`
- `ruleset_analysis`
- `multi_label_rule_analysis`
- `rule_swap_analysis`

`swap_out_report` 属于 `hsreport`。`hsmodel` 中的规则能力仅指规则分类模型，表达式规则和策略规则分析属于 `hsrule`。

### 6.5 hsselect

首期支持所有当前公开 Selector：

- Type、Regex、Null、Mode、Cardinality、Variance
- Corr、VIF
- IV、Lift、PSI
- FeatureImportance、NullImportance、RFE、Sequential、Stepwise
- Boruta、MutualInfo、Chi2、FTest、StabilityAware、ScorecardFeatureSelection
- CompositeFeatureSelector

统一动作包括查询可用 Selector、创建、拟合、转换、拟合转换、读取 support、读取输出特征名、生成筛选报告，以及用 `SelectionReportCollector` 汇总流程报告。

### 6.6 hsviz

首期登记 `hscredit.core.viz` 当前公开绘图函数和样式入口，包括：

- 分箱、二维分箱、分箱趋势、逾期分箱、KS、PSI、相关性、直方图和分布图。
- ROC、PR、Lift、Gain、混淆矩阵、校准和模型权重图。
- 评分分布、评分分箱、审批率与坏率曲线。
- 规则置换、策略仿真、Vintage、特征重要性、漂移和分群图。
- 决策树可视化。
- 统一主题、配色和保存图像入口。

`hsbin` 与 `hsviz` 允许同时暴露分箱图。二者必须映射到同一 visualization adapter 和同一参数校验逻辑，确保默认值、显式 `anchor` 等覆盖参数和渲染结果一致。

### 6.7 hsexcel

首期支持：

- `ExcelWriter`
- `dataframe2excel`
- `DataFrame.save`
- `Series.save`
- 多 Sheet 工作簿组装计划
- `resolve_condition_color`
- `register_pivot_aggregation`

显式调用参数继续优先于 Writer 默认值。条件格式遵循单次 `condition_color`、Writer `condition_color`、副主题默认色的现有优先级。

### 6.8 hscredit

统一入口读取各 Skill 的公开操作索引并选择领域。它只能路由到已登记操作，不允许把用户文本转换成任意 import 路径或 Python 代码。

## 7. 请求契约

所有 Skill 使用版本化 JSON 信封：

```json
{
  "version": "1",
  "operation": "feature_bin_stats",
  "inputs": {
    "data": {
      "kind": "file",
      "path": "data/sample.xlsx",
      "sheet_name": "建模样本"
    }
  },
  "parameters": {
    "feature": ["年龄", "收入"],
    "target": "target"
  },
  "output": {
    "directory": "outputs",
    "name": "变量分箱统计",
    "overwrite": false
  },
  "environment": {
    "mode": "isolated",
    "reuse": true
  }
}
```

公共字段：

- `version`：运行时契约版本，本期固定为字符串 `"1"`。
- `operation`：本 Skill 注册表中的操作名。
- `inputs`：命名输入源。
- `parameters`：传给操作适配器的参数。
- `output`：产物目录、基础名称、格式和覆盖策略。
- `environment`：依赖环境策略。

未知顶层字段、未知操作和操作 Schema 不接受的参数默认拒绝。只有模型和 Selector 构造参数等明确声明为扩展参数对象的位置允许附加键。

## 8. 输入源

### 8.1 文件输入

数据文件支持 CSV、XLSX 和 Parquet。XLSX 可指定 Sheet；CSV 可指定编码和分隔符；Parquet 由基础依赖 `pyarrow` 为 pandas 提供读写引擎。

模型、分箱器、Selector 和其他 Python artifact 支持 hscredit 自身序列化格式。Pickle 或 joblib 仅在输入同时声明 `trusted: true` 时加载；错误响应必须说明反序列化不可信文件可能执行代码。

所有路径先解析为规范绝对路径。输出只能落入请求明确指定的目录，且默认不得覆盖已有文件。

### 8.2 对象引用

Python、Notebook 或常驻 Agent 可以向运行时传入对象注册表，并通过以下形式引用对象：

```json
{
  "kind": "object_ref",
  "ref": "model:baseline"
}
```

CLI 独立进程不接受没有调用方注册表的 `object_ref`。对象引用不通过 JSON 序列化，也不跨隔离解释器自动迁移。

如果 `object_ref` 操作缺少依赖，运行时不得把实时对象复制到新环境。它返回明确提示，要求用户选择 `environment.mode="current"`，或者先把对象保存为受支持 artifact。

## 9. 响应与产物

成功响应结构：

```json
{
  "status": "success",
  "operation": "feature_bin_stats",
  "summary": {
    "rows": 12,
    "columns": ["分箱", "样本数", "坏样本率", "IV"]
  },
  "preview": [],
  "artifacts": [
    {
      "type": "excel",
      "path": "outputs/变量分箱统计.xlsx"
    }
  ],
  "warnings": [],
  "environment": {
    "mode": "isolated",
    "python": "3.12",
    "hscredit": "0.1.2",
    "extras": []
  }
}
```

规则：

- DataFrame 结果返回行数、列名和受限预览；完整内容保存为请求指定的表格格式。
- Excel 报告产物为 `.xlsx`。
- 图像默认保存 PNG，也可显式选择 SVG。
- 模型、分箱器和 Selector 保存为可供后续 Skill 引用的 artifact。
- 产物写入目标目录内的临时区域，全部成功后再移动到最终名称。
- 失败时不得留下被声明为成功的半成品产物。

## 10. 依赖与隔离环境

### 10.1 默认隔离模式

`environment.mode` 默认是 `isolated`。每个 Skill 的标准库引导脚本执行以下流程：

1. 根据操作注册表确定需要的 hscredit extras。
2. 计算 Python 主次版本、hscredit 来源和 extras 的环境键。
3. 在用户缓存目录创建或复用专用 `venv`。
4. 系统存在 `uv` 时可用于加速；否则使用 `python -m venv` 和目标解释器的 `python -m pip`。
5. 只安装 `runtime.json` 和操作注册表允许的 hscredit 来源及 extras。
6. 使用隔离解释器重新执行请求。

在完整本地仓库中运行时，隔离环境可 editable 安装当前仓库。通过 GitHub 独立安装 Skill 时，`runtime.json` 提供受控 hscredit GitHub 来源或发行版本。发布流程把来源固定到对应的发布 tag 或 commit；从 `main` 安装的 Skill 使用同一仓库的 `main` 来源。

### 10.2 当前环境模式

只有请求明确包含以下内容时，运行时才允许在当前解释器安装缺失依赖：

```json
{
  "environment": {
    "mode": "current",
    "install_missing": true
  }
}
```

当前环境模式必须在响应中标记环境已被修改及实际安装的 extras。不得依据缺失依赖提示、对象引用或重试逻辑自行切换到当前环境模式。

### 10.3 安装限制

- 请求不能携带任意包名、任意索引 URL 或任意 Git URL。
- XGBoost、LightGBM、CatBoost 和 NGBoost 等模型映射到受控 `boost` extra。
- AutoTuner 和 ModelTuner 映射到受控 `tune` extra。
- 其他 extras 由操作注册表显式声明。
- 安装失败返回 `DEPENDENCY_INSTALL_FAILED`，包含精简命令上下文、退出码和脱敏日志。
- 安装完成后只执行一次尚未开始的业务操作，不重放已经开始的训练、调参、规则或报告任务。

## 11. 安全与授权边界

- 只允许注册表中的 Skill、操作、类名、Loss 名、Selector 名和绘图函数。
- Rule 表达式只交给 hscredit `Rule` 解析，不使用 Python `eval`。
- 不执行请求提供的 Python 文件、模块路径、shell 文本或安装命令。
- 不自动安装请求给出的第三方包。
- 不自动上传产物、创建远端 Skill、提交 Git、推送或发布。
- 响应不包含原始明细数据；调试模式只返回脱敏 traceback。
- 绝对路径和相对路径均需规范化；覆盖和临时目录清理只针对已验证的精确目标。
- 不预运行、抽样运行或重试用户函数和损失函数。

## 12. 错误模型

运行时定义 `SkillExecutionError`，包含稳定错误码、中文消息、字段路径、原始异常类型和可选脱敏调试信息。

首期错误码：

- `SCHEMA_INVALID`
- `OPERATION_NOT_ALLOWED`
- `INPUT_NOT_FOUND`
- `INPUT_FORMAT_UNSUPPORTED`
- `COLUMN_MISSING`
- `OBJECT_REF_NOT_FOUND`
- `OBJECT_REF_REQUIRES_CURRENT_ENV`
- `ARTIFACT_UNTRUSTED`
- `ARTIFACT_EXISTS`
- `DEPENDENCY_MISSING`
- `DEPENDENCY_INSTALL_FAILED`
- `HSCREDIT_EXECUTION_FAILED`
- `ARTIFACT_WRITE_FAILED`

CLI 失败返回非零退出码，并在标准输出写 JSON 错误信封。Python 调用抛出保留原始异常链的 `SkillExecutionError`。默认错误不得泄露数据内容、令牌、凭证或完整环境变量。

## 13. 在线安装与本地安装

### 13.1 Agent 从 GitHub 安装

支持 Agent 按 GitHub 仓库和目录安装。第一实现阶段的完整安装使用两个路径：

```text
skills/hsbin
skills/hsreport
```

每个目录自包含，因此只安装 `skills/hsbin` 或 `skills/hsreport` 也能运行。总体框架的其他 Skill 在实现后追加到安装清单。

### 13.2 仓库安装器

`python skills/install.py` 支持：

- `--suite hscredit`：安装当前版本已实现的整套 Skill；第一实现阶段为 `hsbin` 与 `hsreport`。
- `--skills hsbin hsreport`：安装指定 Skill。
- `--target-dir <path>`：安装到用户指定目录。
- 已存在目标目录时默认失败，不静默覆盖。

### 13.3 hscredit CLI

`python -m hscredit skills` 增加 `list`、`install`、`update` 和 `uninstall` 子命令。已安装 hscredit 包但没有源码仓库时，CLI 从与当前 hscredit 版本匹配的 GitHub tag 或发布包下载 Skill，不依赖 wheel 中复制一份根目录 Skill。

更新和卸载只能操作安装清单中记录的精确 Skill 目录。卸载不得递归删除用户自定义的未登记目录。

## 14. 测试策略

实现遵循测试先行：每项生产行为先写测试并确认因能力缺失而失败，再实现最小代码使测试通过。

### 14.1 结构和契约测试

- Skill Creator 结构校验覆盖当前已实现的两个 Skill。
- 校验所有 `SKILL.md` frontmatter、名称和描述。
- 校验所有 `agents/openai.yaml`。
- 校验所有请求 JSON Schema 和 `runtime.json`。
- 校验每个文档操作都存在注册表项，每个注册表项都存在操作说明。
- 校验每个 Skill 目录不包含指向兄弟目录的运行时依赖。

### 14.2 公共运行时测试

- CSV、XLSX、Parquet 加载。
- 对象注册、解析、缺失引用和进程边界。
- artifact 暂存、成功发布、失败清理和禁止覆盖。
- 白名单、未知参数、中文错误码和脱敏。
- 本地测试包驱动的真实 venv 创建与 pip 安装，不依赖公网。
- 当前环境安装的显式授权门禁；测试不得修改实际开发环境。

### 14.3 Skill 行为测试

- `hsbin`：所有登记统计、分箱、方法对比、二维分箱和分箱图运行真实 API。
- `hsreport`：真实生成特征、模型和策略 Excel；openpyxl 检查工作表、关键标题、表格和链接。
- 尚未实现的 `hscredit`、`hsmodel`、`hsrule`、`hsselect`、`hsviz` 和 `hsexcel` 不创建运行测试；结构测试确保其目录存在、只含 `.gitkeep`，且未被误标为可用。

核心计算测试使用真实 hscredit API。只有外部网络、进程边界或不可控宿主接口允许使用替身；替身不得替代分箱、建模、筛选、规则、绘图和报告结果断言。

### 14.4 验证命令

```powershell
pytest tests/test_skills -q
pytest tests/test_report tests/test_binning tests/test_visualization -q
python -m build
git diff --check
```

模型、Selector、规则和 Excel 相关的既有聚焦测试根据实际改动追加运行。完整测试与聚焦测试必须在交付说明中分别标注，不能把聚焦通过表述为全套通过。

## 15. 规划清单

`skills/ROADMAP.md` 收录但不在首期开放以下方向：

- EDA、客群画像、Vintage、Roll Rate 和目标分析专用 Skill。
- 模型监控、PSI/CSI、客群漂移和分数漂移专用 Skill。
- 编码器和特征工程专用 Skill。
- 拒绝推断和校准专用 Skill。
- 金融计算专用 Skill。
- 模型解释、SHAP 和上线导出专用 Skill。
- MCP Server、OpenAI Function Calling、Claude、Gemini 和其他宿主的原生 adapter。
- 远端 Skill 发布、版本回滚和签名校验工具。

规划项只能记录目标、依赖和优先级，不在 `CATALOG.md` 或 Skill 描述中宣称已经可用。

## 16. 首期验收标准

1. `hsbin` 与 `hsreport` 均能通过结构校验并被独立安装。
2. `$hsbin` 与 `$hsreport` 可直接调用；未实现 Skill 只有 `.gitkeep` 占位目录，不出现在已实现目录和安装清单中。
3. 文件输入和同进程对象输入均通过真实行为测试。
4. 缺失依赖默认在隔离环境安装；当前环境修改需要显式请求。
5. `object_ref` 在隔离环境不可迁移时返回明确错误和解决路径。
6. 所有首期操作都有白名单、Schema、操作说明和至少一个真实行为测试。
7. Excel、图片、表格和模型产物均有结构化 manifest，失败不留下成功状态的半成品。
8. 用户显式参数继续优先于 hscredit 自动值和默认值。
9. 现有工作区无关修改保持不变；实现过程不提交、不推送。

## 17. 第一阶段验证记录

验证日期：2026-08-20。

- `pytest tests/test_skills -q`：44 passed，1 warning。
- 文件型 launcher 当前环境端到端：`hsbin`、`hsreport` 2 passed。
- 文件型 launcher 默认隔离环境冷启动：2 passed，首次创建 Python 3.14 venv 并安装当前仓库耗时 370.75 秒。
- 相同隔离环境键复用：两个 launcher 共 13.10 秒。
- `pytest tests/test_report tests/test_binning tests/test_visualization -q`：1274 passed，13 warnings；这是相关套件，不表述为全仓库测试。
- 最终 `pytest tests/ -q`：2874 passed，1 skipped，494 warnings，耗时 455.09 秒；0 failed。
- Skill Creator `quick_validate.py`：`hsbin` 与 `hsreport` 均为 valid；Windows 下使用 `PYTHONUTF8=1` 读取中文文件。
- `python -m compileall`、`python -m build` 和 `git diff --check` 通过。
- 干净临时 build-base wheel：包含 14 个 `hscredit/skills_runtime` 条目，不包含顶层 `skills/`、`build/` 或 `scripts/`。
- 临时 venv 安装 wheel 后成功导入运行时；注册表包含 14 个 `hsbin` 操作和 3 个 `hsreport` 操作。
- 已实际查看 `bin_plot`、`bin_overdues_plot`、`bin_2d_plot` 和特征效率分析图片，未发现标题、图例、坐标轴或标注裁切。
