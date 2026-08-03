# dataframe2excel 保样式快速写入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `dataframe2excel` 增加 `fast=True` 保样式快速写入和与 ScoreCard 一致的 `decimal` 精度参数，并消除 DataFrame 自动列宽的逐单元格重复扫描。

**Architecture:** 默认路径保留现有行生成、样式选择和合并语义；快速路径复用这些规则，但通过整数坐标、缓存样式和一次值规范化直接写入 openpyxl 单元格。列宽在 DataFrame 写完后统一处理，快速路径边写边累计宽度，默认路径只执行一次批量调整。

**Tech Stack:** Python、pandas、openpyxl、pytest

## Global Constraints

- `hscredit.excel` 导出的 Excel 必须保留现有样式，不增加无样式或 write-only 模式。
- 公开参数使用 `fast: bool = False` 和 `decimal: Optional[int] = 4`。
- `decimal` 命名与 `ScoreCard` / `RoundScoreCard` 一致；`None` 表示不主动舍入。
- `fast=False` 的值、样式、合并、坐标和格式输出保持兼容。
- `fast=True` 必须保持 DataFrame 当前行列顺序和全部现有 Excel 功能。
- 不引入 openpyxl 之外的新 Excel 引擎依赖。

---

### Task 1: decimal 精度契约

**Files:**
- Modify: `hscredit/excel/writer.py:1399-1411,2710-3013`
- Test: `tests/test_report/test_excel_writer.py`

**Interfaces:**
- Consumes: `ExcelWriter.astype_insertvalue(value, decimal=4)`
- Produces: `dataframe2excel(..., decimal: Optional[int] = 4)`，以及对 `None`、非负整数和非法值的明确处理。

- [ ] **Step 1: 写入失败测试**

```python
def test_decimal_none_preserves_float_precision(self):
    value = 1.234567890123
    dataframe2excel(pd.DataFrame({'值': [value]}), self.test_file, decimal=None)
    ws = load_workbook(self.test_file, data_only=False).active
    assert ws['B3'].value == value

@pytest.mark.parametrize('decimal', [-1, True, 1.5, '2'])
def test_invalid_decimal_raises_chinese_value_error(self, decimal):
    with pytest.raises(ValueError, match='decimal 必须是大于等于 0 的整数或 None'):
        dataframe2excel(pd.DataFrame({'值': [1.2345]}), self.test_file, decimal=decimal)
```

- [ ] **Step 2: 运行测试并确认因参数缺失或未校验而失败**

Run: `pytest tests/test_report/test_excel_writer.py::TestDataframe2Excel::test_decimal_none_preserves_float_precision tests/test_report/test_excel_writer.py::TestDataframe2Excel::test_invalid_decimal_raises_chinese_value_error -v`

Expected: FAIL；当前 `dataframe2excel` 不接受公开 `decimal`，且底层固定使用四位小数。

- [ ] **Step 3: 实现最小精度参数**

```python
@staticmethod
def _validate_decimal(decimal: Optional[int]) -> None:
    if decimal is not None and (isinstance(decimal, (bool, np.bool_)) or not isinstance(decimal, (int, np.integer)) or int(decimal) < 0):
        raise ValueError("decimal 必须是大于等于 0 的整数或 None")

@staticmethod
def astype_insertvalue(value: Any, decimal: Optional[int] = 4) -> Any:
    if re.search('tuple|list|set|numpy.ndarray|Categorical|numpy.dtype|Interval', str(type(value))):
        return str(value)
    if re.search('float', str(type(value))):
        return float(value) if decimal is None else round(float(value), int(decimal))
    return value
```

将 `decimal` 从 `dataframe2excel` 传递至 `insert_df2sheet`、`insert_rows` 和 `insert_value2sheet`。

- [ ] **Step 4: 运行 Task 1 测试并确认通过**

Run: `pytest tests/test_report/test_excel_writer.py -k "decimal" -v`

Expected: PASS。

- [ ] **Step 5: 提交精度契约**

Run: `git add hscredit/excel/writer.py tests/test_report/test_excel_writer.py && git commit -m "feat: 统一 Excel 浮点精度参数"`

### Task 2: 保样式快速单元格写入

**Files:**
- Modify: `hscredit/excel/writer.py:900-1316`
- Test: `tests/test_report/test_excel_writer.py`

**Interfaces:**
- Consumes: `ExcelWriter.insert_df2sheet(..., fast=False, decimal=4)`
- Produces: `_insert_rows_fast` 和 `_insert_cell_fast` 私有写入器，使用整数坐标并复用现有样式名称、合并规则和值规范化。

- [ ] **Step 1: 写入快速模式内容、顺序和样式失败测试**

```python
def test_fast_mode_preserves_values_order_styles_and_coordinates(self):
    df = pd.DataFrame({'编号': ['001', '002'], '数值': [1.23456, 2.34567]}, index=['甲', '乙'])
    normal = os.path.join(self.temp_dir, 'normal.xlsx')
    fast = os.path.join(self.temp_dir, 'fast.xlsx')
    normal_end = dataframe2excel(df, normal, sheet_name='S', index=True, fill=True)
    fast_end = dataframe2excel(df, fast, sheet_name='S', index=True, fill=True, fast=True)
    normal_ws = load_workbook(normal).get_sheet_by_name('S')
    fast_ws = load_workbook(fast).get_sheet_by_name('S')
    assert fast_end == normal_end
    for row in range(2, normal_end[0]):
        for col in range(2, normal_end[1]):
            assert fast_ws.cell(row, col).value == normal_ws.cell(row, col).value
            assert fast_ws.cell(row, col).style_id == normal_ws.cell(row, col).style_id
```

另加 MultiIndex 表头、`fill=False`、`merge=True` 与前导零文本格式的独立行为测试。

- [ ] **Step 2: 运行快速模式测试并确认因 `fast` 未实现而失败**

Run: `pytest tests/test_report/test_excel_writer.py -k "fast_mode" -v`

Expected: FAIL，`fast` 当前会落入 `insert_df2sheet` 的未知参数。

- [ ] **Step 3: 实现直接坐标快速写入**

```python
def _insert_cell_fast(self, worksheet, row_index, col_index, value, style, decimal):
    cell = worksheet.cell(row=row_index, column=col_index)
    cell._style = copy.copy(self._style_cache[style])
    cell.value = self.astype_insertvalue(value, decimal=decimal)
    if self.is_numeric_like_string(value):
        cell.number_format = '@'
    return cell
```

在 `insert_df2sheet` 中继续使用现有 `_iter_rows` 和样式分支；`fast=True` 时调用整数坐标快速行写入，`fast=False` 时继续调用 `insert_rows`。多层表头仍沿用现有合并区间计算。

`ExcelWriter.__init__` 在命名样式注册完成后构建 `self._style_cache = {style.name: style.as_tuple() for style in self.name_styles}`，快速路径仅复用已注册样式。

- [ ] **Step 4: 运行快速模式测试并修正样式、合并或坐标差异**

Run: `pytest tests/test_report/test_excel_writer.py -k "fast_mode" -v`

Expected: PASS。

- [ ] **Step 5: 提交保样式快速写入**

Run: `git add hscredit/excel/writer.py tests/test_report/test_excel_writer.py && git commit -m "feat: 增加 Excel 保样式快速写入"`

### Task 3: 自动列宽、外层格式与性能验证

**Files:**
- Modify: `hscredit/excel/writer.py:297-369,900-1192,2850-3007`
- Modify: `docs/api/excel.rst`
- Test: `tests/test_report/test_excel_writer.py`

**Interfaces:**
- Consumes: Task 2 的快速单元格路径。
- Produces: DataFrame 写入期间不触发逐单元格自动列宽；快速模式累计列宽后一次设置；外层数字格式、条件格式、对齐、图片和筛选保持一致。

- [ ] **Step 1: 写入自动列宽单次处理和格式一致性失败测试**

```python
def test_dataframe_auto_width_does_not_adjust_each_cell(self, monkeypatch):
    calls = []
    original = ExcelWriter._get_column_cells_data
    def counted(writer, worksheet, col_letter):
        calls.append(col_letter)
        return original(writer, worksheet, col_letter)
    monkeypatch.setattr(ExcelWriter, '_get_column_cells_data', counted)
    writer = ExcelWriter()
    ws = writer.get_sheet_by_name('S')
    dataframe2excel(pd.DataFrame({'A': range(20), 'B': range(20)}), writer, sheet_name=ws, auto_width=True)
    assert len(calls) <= 2
```

增加 `percent_cols`、`custom_cols`、`condition_cols`、`color_cols`、`left_cols`、`right_cols`、图片占位和 `auto_filter` 的 normal/fast 工作簿比对测试。

- [ ] **Step 2: 运行测试并确认当前逐单元格列宽逻辑导致失败**

Run: `pytest tests/test_report/test_excel_writer.py -k "auto_width_does_not_adjust_each_cell or fast_mode_preserves_formats" -v`

Expected: FAIL，当前 `auto_width=True` 从每个单元格进入列快照和样式恢复。

- [ ] **Step 3: 将 DataFrame 自动列宽改为写完后一次处理**

```python
cell_auto_width = False
self.insert_rows(..., auto_width=cell_auto_width, decimal=decimal)
if auto_width:
    if fast:
        self._apply_accumulated_widths(worksheet, widths, start_col_idx)
    else:
        self.adjust_columns_width(worksheet, start_col=start_col_idx, end_col=end_col_idx - 1)
```

更新公开 docstring 和 `docs/api/excel.rst`，说明 `fast=True` 始终保留 hscredit 样式，`decimal=None` 仅关闭代码主动舍入。

- [ ] **Step 4: 运行 Excel Writer 回归测试**

Run: `pytest tests/test_report/test_excel_writer.py -v`

Expected: PASS。

- [ ] **Step 5: 运行相关报告测试**

Run: `pytest tests/test_report/test_model_report.py tests/test_report/test_rule_strategy.py tests/test_report/test_rule_analysis.py tests/test_report/test_population_drift.py -v`

Expected: PASS。

- [ ] **Step 6: 运行性能基准并记录结果**

Run: `python -m pytest tests/test_report/test_excel_writer.py -k "fast_mode" -v`，随后运行 1000×100 临时 DataFrame 的 normal/fast 写入脚本，分别记录写入、保存和总耗时。

Expected: `fast=True` 的写入耗时低于默认路径；输出文件可由 openpyxl 重新打开且逐格内容、样式和顺序一致。

- [ ] **Step 7: 最终静态与差异检查**

Run: `python -m compileall hscredit/excel tests/test_report/test_excel_writer.py`

Run: `git diff --check`

Expected: 两条命令均为 exit 0。

- [ ] **Step 8: 提交列宽、文档和验证改造**

Run: `git add hscredit/excel/writer.py tests/test_report/test_excel_writer.py docs/api/excel.rst && git commit -m "perf: 优化 Excel 大表写入"`
