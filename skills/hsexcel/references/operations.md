# hsexcel 操作

| operation | 输入 | 产物 |
|---|---|---|
| `dataframe_to_excel` | `inputs.data` | 单 Sheet `.xlsx` |
| `workbook` | 多个 DataFrame named inputs | 多 Sheet `.xlsx` |

输出配置中的 `overwrite`、`name` 和 `directory` 由运行时管理；不要把绝对文件路径作为 `parameters.save` 传给适配器。
