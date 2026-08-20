# Agent Skills

hscredit 提供可被 AI Agent 发现和调用的 Skills。Agent 负责理解分析目标、组装数据与参数，
Skill 负责校验请求、调用真实 hscredit API，并交付 Excel、图片、JSON 或可复用制品。

当前已实现两个 Skill：

| Skill | 适用任务 | 主要产物 |
|:---|:---|:---|
| `$hsbin` | 变量分箱、分箱统计、效率评估、分箱器拟合与分箱图 | Excel、PNG/SVG、规则 JSON、分箱器 artifact |
| `$hsreport` | 特征分析、模型评估和策略迭代完整报告 | 多 Sheet Excel 报告及报告图片 |

`hscredit`、`hsmodel`、`hsrule`、`hsselect`、`hsviz` 和 `hsexcel` 当前只是规划目录，
没有 `SKILL.md`，不能被 Agent 调用。

## 安装 Skills

### 从仓库安装

在 hscredit 仓库根目录安装当前已实现的整套 Skill：

```bash
python skills/install.py --suite hscredit
```

只安装一个 Skill：

```bash
python skills/install.py --skills hsbin
python skills/install.py --skills hsreport
```

指定 Agent 的 Skills 目录：

```bash
python skills/install.py --skills hsbin hsreport --target-dir /path/to/agent/skills
```

未指定 `--target-dir` 时，安装器优先使用 `$CODEX_HOME/skills`，否则使用
`~/.codex/skills`。安装器不会覆盖已经存在的 Skill 目录；更新前应先检查并保留自己的修改。

安装完成后，在 Agent 的下一轮对话中调用 Skill。

### 让 Agent 从 GitHub 安装

支持从 GitHub 路径安装 Skill 的 Agent，可以直接接收以下任务：

```text
请从 GitHub 仓库 hengshu-credit/hscredit 的 main 分支安装：
skills/hsbin
skills/hsreport
```

也可以只安装需要的目录：

```text
请安装 hengshu-credit/hscredit 中的 skills/hsbin。
```

公开仓库可直接下载；私有仓库需要 Agent 已具备相应 Git 凭证或访问令牌。Skill 目录是独立安装单位，
无需同时复制规划中的空目录。

### OpenAI 项目 Skills API

OpenAI Skills API 支持以文件目录集合或 ZIP 创建 Skill，并提供独立的 Skill ID 和版本管理。
发布时分别上传 `skills/hsbin` 与 `skills/hsreport`，不要把两个目录合并成一个不可区分的 Skill。

参考 [OpenAI Skills API](https://developers.openai.com/api/reference/python/resources/skills/methods/create)。

## 在 Agent 中调用

### 分箱分析

直接描述数据位置、字段角色、分析目标和输出要求：

```text
$hsbin 读取 data/credit.xlsx 的“建模样本”工作表，
对 score 和 age 按 target 做等频分箱，最多 5 箱，
生成分箱统计 Excel 和分箱图，保存到 outputs/。
```

```text
$hsbin 使用训练好的 outputs/binner.joblib，
把 data/oot.parquet 转换为 WOE，制品是可信的，
结果保存为 outputs/oot_woe.xlsx。
```

Agent 会根据任务选择 `feature_bin_stats`、`optimal_binning_fit`、`bin_plot` 等具体操作。
如果用户只需要单张统计表或图片，应使用 `$hsbin`，不必生成完整报告。

### 完整报告

```text
$hsreport 读取 data/credit.xlsx，使用 target 作为坏样本标签，
分析 score、age 和 debt_ratio，按 apply_date 查看月度趋势，
生成完整特征分析报告并保存为 outputs/特征分析报告.xlsx。
```

```text
$hsreport 使用可信模型 artifacts/model.joblib，
对 train.parquet 和 oot.parquet 生成模型报告，
目标字段是 target，不显示控制台报告，结果保存到 outputs/。
```

```text
$hsreport 分析规则“score < 560”和“age < 25”，
同时输出订单与 amount 金额口径、按 apply_date 的月度稳定性，
生成策略迭代报告。
```

`$hsreport` 只提供以下完整报告入口：

- `auto_feature_analysis`
- `auto_model_report`
- `swap_out_report`

## 数据输入

### 文件输入

文件模式适合普通 Chat、CLI 和独立 Agent 进程：

| 格式 | 配置 |
|:---|:---|
| CSV | 可指定 `encoding` 和 `separator` |
| XLSX | 可指定 `sheet_name` |
| Parquet | 自动启用受控 `parquet` 依赖组 |
| hscredit/joblib artifact | 必须显式声明 `trusted=true` |

Pickle 和 joblib 反序列化可能执行代码。只对自己生成或已经验证来源的制品使用可信模式。

### 同进程对象

Python、Notebook 或常驻 Agent 可以把 `DataFrame`、模型或分箱器放入对象注册表：

```python
from hscredit.skills_runtime import execute_skill

request = {
    "version": "1",
    "operation": "feature_bin_stats",
    "inputs": {"data": {"kind": "object_ref", "ref": "data:train"}},
    "parameters": {
        "feature": "score",
        "target": "target",
        "method": "quantile",
        "max_n_bins": 5,
        "n_jobs": 1,
    },
    "output": {
        "directory": "outputs",
        "name": "score_stats",
        "overwrite": False,
    },
    "environment": {"mode": "current", "install_missing": False},
}

result = execute_skill("hsbin", request, objects={"data:train": train_df})
```

`object_ref` 只存在于当前 Python 进程，不能自动传入新的隔离解释器。缺少依赖时，先把对象保存为可信
artifact，或者在用户明确同意后使用当前环境安装。

## 运行环境

默认 `environment.mode` 是 `isolated`：

1. Skill 在用户缓存目录创建专用 venv。
2. 根据操作和文件格式安装受控 hscredit extras。
3. 后续请求按 Python 版本、hscredit 来源和 extras 复用同一环境。

首次运行需要创建环境和安装依赖，耗时通常明显高于后续复用。Skill 不接受请求提供的任意包名、
包索引或 Git URL。

只有用户明确要求修改当前代码环境时，才使用：

```json
{
  "environment": {
    "mode": "current",
    "install_missing": true
  }
}
```

## 高级：手工 JSON 调用

普通 Agent 使用时不需要手写 JSON。调试、自动化或其他 Agent 适配器可以直接运行 Skill launcher。

`hsbin` 请求示例：

```json
{
  "version": "1",
  "operation": "feature_bin_stats",
  "inputs": {
    "data": {"kind": "file", "path": "data/credit.csv"}
  },
  "parameters": {
    "feature": ["score", "age"],
    "target": "target",
    "method": "quantile",
    "max_n_bins": 5,
    "n_jobs": 1
  },
  "output": {
    "directory": "outputs",
    "name": "变量分箱统计",
    "overwrite": false
  },
  "environment": {"mode": "isolated", "reuse": true}
}
```

```bash
python skills/hsbin/scripts/run.py request.json
```

launcher 输出一个 JSON 信封：

- `status`：执行状态。
- `summary`：行列信息、数据集名称或报告结束位置等紧凑摘要。
- `artifacts`：Excel、图片、JSON 或制品的绝对路径。
- `warnings`：非阻断警告。
- `environment`：实际 Python、hscredit 版本和 extras。

## 常见错误

| 错误码 | 含义 | 处理方式 |
|:---|:---|:---|
| `SCHEMA_INVALID` | 请求字段或参数不符合契约 | 检查对应 Skill 的 `schemas/request.schema.json` |
| `OPERATION_NOT_ALLOWED` | Skill 不支持该操作 | 改用已登记操作或正确的 Skill |
| `INPUT_NOT_FOUND` | 文件、工作表或对象引用不存在 | 检查路径、Sheet 名和对象注册表 |
| `COLUMN_MISSING` | 分析所需字段缺失 | 明确特征、目标、日期和金额字段 |
| `ARTIFACT_UNTRUSTED` | 未授权加载 pickle/joblib | 验证来源后显式设置 `trusted=true` |
| `ARTIFACT_EXISTS` | 输出文件已存在 | 更换名称，或明确设置 `overwrite=true` |
| `OBJECT_REF_REQUIRES_CURRENT_ENV` | 实时对象不能跨隔离解释器 | 使用当前环境或先保存 artifact |
| `DEPENDENCY_INSTALL_FAILED` | 隔离依赖安装失败 | 检查网络、Python 版本和精简后的 pip 日志 |

## 查看完整操作契约

- `skills/hsbin/references/operations.md`
- `skills/hsreport/references/operations.md`
- `skills/hsbin/schemas/request.schema.json`
- `skills/hsreport/schemas/request.schema.json`
