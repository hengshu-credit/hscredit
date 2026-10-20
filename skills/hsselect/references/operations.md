# hsselect 操作

`selector_fit` 的 `parameters` 形状：

```json
{
  "selector": "IVSelector",
  "selector_params": {"threshold": 0.02},
  "target": "target",
  "features": ["score", "age"]
}
```

当前支持 `hscredit.core.selectors.__all__` 中的公开筛选器，适配器会拒绝未登记类名。
