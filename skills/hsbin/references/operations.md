# hsbin 操作参考

## 公共信封

所有操作使用 `version="1"`。`inputs` 保存数据或 artifact 来源，`parameters` 只保存 hscredit 业务参数，`output` 控制产物目录、名称、格式和覆盖策略。

```json
{
  "version": "1",
  "operation": "feature_bin_stats",
  "inputs": {
    "data": {"kind": "file", "path": "data/sample.xlsx", "sheet_name": "建模样本"}
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

## 表格分析

| operation | 主要输入 | 产物 |
|---|---|---|
| `feature_bin_stats` | `data`; `feature`, `target` 或 `overdue+dpds` | 分箱统计 Excel |
| `benchmark_binning_methods` | `data`; `feature`, `overdue_col`, `dpds` | 方法对比 Excel |
| `feature_binning_summary` | `data`; `feature`, `methods`, 目标配置 | 摘要和各方法明细 Excel |
| `feature_group_binning_summary` | `data`; 上述参数，加 `date_col` 或 `group_col` | 分组摘要和明细 Excel |
| `feature_efficiency_analysis` | `data`; `feature`, 目标配置，可选日期/分组 | 手工/自动表、规则 JSON、比较图和趋势图 |

这些操作只接受对应 hscredit 函数显式声明的参数。底层函数的 `**kwargs` 不作为开放的任意参数入口。

## 分箱器生命周期

| operation | 必要参数 | 说明 |
|---|---|---|
| `optimal_binning_fit` | `inputs.data`; `features`, `target` | 拟合并输出 artifact、转换结果和分箱表 |
| `optimal_binning_fit_transform` | 同上 | 拟合并转换；与 fit 使用同一安全契约 |
| `optimal_binning_transform` | `inputs.data`, `inputs.binner`; 可选 `features`, `metric` | 使用已拟合一维分箱器 |
| `optimal_binning_2d_fit` | `inputs.data`; 两个 `features`, `target` | 拟合二维分箱器并输出 artifact |
| `optimal_binning_2d_transform` | `inputs.data`, `inputs.binner`; 可选 `metric` | 使用已拟合二维分箱器 |

`metric` 原样传给 hscredit。不要把默认 `indices` 解释为 WOE，也不要在没有明确请求时改为 `woe`。

可信 artifact 输入：

```json
{
  "kind": "file",
  "path": "outputs/binner.joblib",
  "trusted": true
}
```

## 绘图

| operation | 输入形式 | 产物 |
|---|---|---|
| `bin_plot` | 原始 DataFrame 或分箱统计表 | PNG/SVG |
| `bin_trend_plot` | DataFrame、特征、目标和日期/分组 | PNG/SVG |
| `bin_overdues_plot` | 原始逾期数据或 `inputs.bin_table` | PNG/SVG |
| `bin_2d_plot` | 原始数据或 `inputs.binner` | PNG/SVG |

图片格式由 `output.format` 选择，默认 `png`。显式 `anchor` 必须原样传递。输出路径由运行时注入，`parameters.save` 不允许使用。

## 输入边界

- Excel 使用 `sheet_name`；CSV 可使用 `encoding` 和 `separator`。
- Parquet 会触发受控 `parquet` extra。
- `object_ref` 只在同进程 Python 调用中可用，不能由独立 CLI JSON 自行解析。
- Pickle/joblib 可能执行代码，未设置 `trusted=true` 时必须拒绝。
