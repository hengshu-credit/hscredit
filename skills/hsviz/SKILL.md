---
name: hsviz
description: Render allowlisted hscredit model, score, strategy, drift, tree, and binning visualizations to PNG or SVG. Use for standalone charts; use hsbin for a binning workflow that also computes tables.
---

# HSCredit 可视化

使用 `plot`，通过 `parameters.plot` 选择当前 hscredit 可视化白名单中的绘图入口。输入对象通过 `inputs` 按参数名映射，例如 `data`、`y_true`、`y_score`、`model`、`X`。绘图参数放在 `kwargs`；输出格式支持 PNG/SVG。

显式 `anchor`、`figsize`、`title`、`save` 语义以 hscredit 当前实现为准。图片保存为 artifact manifest，不把 Figure 对象放入 JSON。
