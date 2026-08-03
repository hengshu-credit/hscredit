# dataframe2excel 自动速度选择实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用默认 `speed="auto"` 替换公开 `fast` 参数，按 DataFrame 大小和维数自动选择保样式兼容路径或快速路径，并允许显式覆盖。

**Architecture:** `ExcelWriter.insert_df2sheet` 统一校验和解析 `speed`，通过固定复合阈值返回实际的 `normal` / `fast` 路径；`dataframe2excel` 只负责公开参数转发。现有两条保样式写入实现不变，只替换路由入口和测试调用方式。

**Tech Stack:** Python、pandas、openpyxl、pytest

## Global Constraints

- 公开参数必须是 `speed: str = "auto"`，合法值为 `auto`、`normal`、`fast`。
- 删除公开 `fast` 参数，不保留两个相互冲突的速度开关。
- 自动快速阈值为：行数大于等于 500、有效列数大于等于 50、或有效单元格数大于等于 10,000，任一满足即使用快速路径。
- `index=True` 时有效列数包含索引层级；表头行不计入阈值。
- 显式 `normal` / `fast` 必须覆盖自动判断。
- 三种模式都必须保留 hscredit 完整样式、顺序和内容。
- `decimal` 语义保持不变。

---

### Task 1: speed 参数与自动路由

**Files:**
- Modify: `hscredit/excel/writer.py:941-955,2907-2936`
- Test: `tests/test_report/test_excel_writer.py:558-755`

**Interfaces:**
- Produces: `ExcelWriter._resolve_write_speed(data: pd.DataFrame, speed: str, index: bool = False) -> str`
- Produces: `ExcelWriter.insert_df2sheet(..., speed: str = "auto")`
- Produces: `dataframe2excel(..., speed: str = "auto")`

- [ ] **Step 1: 写入失败测试**

```python
def test_dataframe2excel_speed_defaults_to_auto_and_replaces_fast():
    params = inspect.signature(dataframe2excel).parameters
    assert params['speed'].default == 'auto'
    assert 'fast' not in params

@pytest.mark.parametrize(
    'rows,cols,index,expected',
    [
        (499, 1, False, 'normal'),
        (500, 1, False, 'fast'),
        (1, 49, False, 'normal'),
        (1, 50, False, 'fast'),
        (399, 25, False, 'normal'),
        (400, 25, False, 'fast'),
        (1, 49, True, 'fast'),
    ],
)
def test_auto_speed_uses_rows_effective_columns_and_cells(rows, cols, index, expected):
    data = pd.DataFrame(np.zeros((rows, cols)))
    assert ExcelWriter._resolve_write_speed(data, 'auto', index=index) == expected
```

增加显式 `normal` / `fast` 覆盖、`" FAST "` 归一化，以及 `None`、布尔值、数字和未知字符串抛出中文 `ValueError` 的独立测试。

- [ ] **Step 2: 运行测试并确认因 speed 不存在而失败**

Run: `python -m pytest tests/test_report/test_excel_writer.py -k "speed_defaults or auto_speed or explicit_speed or invalid_speed" -v`

Expected: FAIL，当前签名仍为 `fast=False` 且没有自动路由器。

- [ ] **Step 3: 实现最小路由器并替换公开参数**

```python
AUTO_FAST_MIN_ROWS = 500
AUTO_FAST_MIN_COLUMNS = 50
AUTO_FAST_MIN_CELLS = 10_000

@classmethod
def _resolve_write_speed(cls, data, speed, index=False):
    if not isinstance(speed, str):
        raise ValueError("speed 必须是 'auto'、'normal' 或 'fast'")
    normalized = speed.strip().lower()
    if normalized not in {'auto', 'normal', 'fast'}:
        raise ValueError("speed 必须是 'auto'、'normal' 或 'fast'")
    if normalized != 'auto':
        return normalized
    rows = len(data)
    columns = len(data.columns) + (data.index.nlevels if index else 0)
    if rows >= cls.AUTO_FAST_MIN_ROWS or columns >= cls.AUTO_FAST_MIN_COLUMNS or rows * columns >= cls.AUTO_FAST_MIN_CELLS:
        return 'fast'
    return 'normal'
```

`insert_df2sheet` 用 `resolved_speed == "fast"` 得到内部布尔分支；`dataframe2excel` 将 `speed` 原样传入。

- [ ] **Step 4: 运行 speed 路由测试并确认通过**

Run: `python -m pytest tests/test_report/test_excel_writer.py -k "speed_defaults or auto_speed or explicit_speed or invalid_speed" -v`

Expected: PASS。

### Task 2: 现有等价测试、文档与回归

**Files:**
- Modify: `tests/test_report/test_excel_writer.py:558-755`
- Modify: `docs/api/excel.rst:7-34`
- Modify: `hscredit/excel/writer.py:916-930,2960-2975`

**Interfaces:**
- Consumes: Task 1 的 `speed` API。
- Produces: 所有 normal/fast 样式等价测试改用显式 `speed`；新增默认 auto 的小表/大表实际路由与输出验证。

- [ ] **Step 1: 将现有 `fast=True` 测试改为 `speed="fast"` 并增加自动集成测试**

```python
def test_default_auto_speed_writes_small_and_large_tables(tmp_path):
    small = pd.DataFrame(np.zeros((10, 10)))
    large = pd.DataFrame(np.zeros((500, 2)))
    dataframe2excel(small, tmp_path / 'small.xlsx')
    dataframe2excel(large, tmp_path / 'large.xlsx')
    assert load_workbook(tmp_path / 'small.xlsx').active['B2'].value == 0
    assert load_workbook(tmp_path / 'large.xlsx').active['B2'].value == 0
```

保留逐格值、样式、数字格式、MultiIndex、合并、自动列宽、条件格式和筛选比对。

- [ ] **Step 2: 更新文档和 docstring**

文档示例改为默认 `speed="auto"`，列出显式 `normal` / `fast` 和三个自动阈值；删除所有公开 `fast=True` 示例。

- [ ] **Step 3: 运行 Excel Writer 与报告回归**

Run: `python -m pytest tests/test_report -q`

Expected: 166 项加新增用例全部 PASS。

- [ ] **Step 4: 运行编译、差异与签名检查**

Run: `python -m compileall -q hscredit/excel tests/test_report/test_excel_writer.py`

Run: `git diff --check`

Run: `python -c "import inspect; from hscredit.excel import dataframe2excel; p=inspect.signature(dataframe2excel).parameters; assert p['speed'].default == 'auto' and 'fast' not in p"`

Expected: 三条命令 exit 0。

- [ ] **Step 5: 提交实现**

Run: `git add hscredit/excel/writer.py tests/test_report/test_excel_writer.py docs/api/excel.rst`

Run: `git commit -m "feat: 自动选择 Excel 写入速度"`
