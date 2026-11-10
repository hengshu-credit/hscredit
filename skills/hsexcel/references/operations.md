# hsexcel 操作

| operation | 输入 | 产物 |
|---|---|---|
| `dataframe_to_excel` | `inputs.data` | 单 Sheet `.xlsx` |
| `workbook` | 多个 DataFrame named inputs | 多 Sheet `.xlsx` |

输出配置中的 `overwrite`、`name` 和 `directory` 由运行时管理；不要把绝对文件路径作为 `parameters.save` 传给适配器。

## 条件格式

常规指标使用 `color_cols` 色阶，低值接近白色，高值接近指标色；`centered_bar_cols` 用于坏账改善等有符号指标，`condition_cols` 用于显式选择数据条。显式 `condition_color` 优先，Writer 配置放入 `writer_params`。

默认样本占比、金额占比、通过率为黄绿 `#BADC58`；规则命中率和坏样本率为珊瑚红 `#F76E6C`；好样本相关指标为浅主题蓝 `#A3ABF5`。保留指标原值与百分比格式，不把颜色深浅解释为业务好坏。

当前操作支持 DataFrame/多表导出以及 `dataframe2excel` 接受的图像与样式参数；没有创建原生透视表的 Skill operation。需要透视表时使用 Python/Notebook 中的 `ExcelWriter.insert_pivot_table2sheet`。
