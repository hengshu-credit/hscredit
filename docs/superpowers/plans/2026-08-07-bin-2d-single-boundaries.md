# `bin_2d_plot` Single-Color Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-bin colored outlines in `bin_2d_plot` with one dark separator drawn exactly once between adjacent cells that belong to different final 2D bins.

**Architecture:** Keep the heatmap matrices, annotations, white pre-bin grid, and axes unchanged. Refactor `_draw_2d_bin_boundaries` to compare horizontally and vertically adjacent entries in the display-oriented `solution_`, collect only differing-bin interfaces, and render them in one `LineCollection`; the plot frame remains owned by the Matplotlib axes.

**Tech Stack:** Python 3, NumPy, Matplotlib `LineCollection`, pytest.

## Global Constraints

- Only adjacent cells with different final 2D bin IDs receive a dark separator.
- Each shared separator is drawn once with one color.
- The helper must not draw the heatmap's outer frame.
- Preserve every pre-bin cell's existing metric value and heatmap color.
- Preserve the existing thin white pre-bin grid.
- Add no bin IDs, labels, legends, markers, or merged-value annotations.
- Apply identical separator segments, color, and linewidth to all five metric heatmaps.
- Include missing-value rows and columns in adjacency checks through `solution_`.
- Do not modify `OptimalBinning2D`, KS plots, marginal bin plots, or public `bin_2d_plot` parameters.

---

### Task 1: Draw final-bin interfaces once in one color

**Files:**
- Modify: `tests/test_visualization/test_bin_plot_layout.py:107-218,362-370`
- Modify: `hscredit/core/viz/binning_plots.py:24-29,2496-2550,2910-2929`

**Interfaces:**
- Consumes: `solution: np.ndarray` whose rows are feature-1 pre-bins and columns are feature-2 pre-bins.
- Produces: `_draw_2d_bin_boundaries(ax, solution, *, expected_shape=None, color="#222222", linewidth=2.2) -> List[LineCollection]`, containing exactly one `LineCollection` with gid `bin-2d-boundaries`.

- [ ] **Step 1: Replace the helper regression test with literal internal-interface expectations**

```python
def test_draw_2d_bin_boundaries_draws_each_internal_separator_once():
    """异箱共享边只绘制一次，且不绘制热力图最外框。"""
    import matplotlib.pyplot as plt

    solution = np.array([[0, 0, 1], [0, 2, 1]])
    fig, ax = plt.subplots()
    try:
        artists = binning_plots._draw_2d_bin_boundaries(
            ax,
            solution,
            expected_shape=(2, 3),
        )

        assert len(artists) == 1
        boundary = artists[0]
        assert boundary.get_gid() == "bin-2d-boundaries"
        assert {
            _normalized_segment(segment)
            for segment in boundary.get_segments()
        } == {
            _normalized_segment(((0.5, -0.5), (0.5, 0.5))),
            _normalized_segment(((1.5, -0.5), (1.5, 0.5))),
            _normalized_segment(((1.5, 0.5), (1.5, 1.5))),
            _normalized_segment(((0.5, 0.5), (1.5, 0.5))),
        }
        assert tuple(boundary.get_colors()[0]) == pytest.approx(to_rgba("#222222"))
    finally:
        plt.close(fig)


def test_draw_2d_bin_boundaries_omits_frame_for_single_merged_bin():
    """单一最终箱没有内部边界，外框继续由坐标轴负责。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        artists = binning_plots._draw_2d_bin_boundaries(
            ax,
            np.zeros((2, 3), dtype=int),
            expected_shape=(2, 3),
        )
        assert len(artists) == 1
        assert artists[0].get_segments() == []
    finally:
        plt.close(fig)
```

Replace the five-axis integration assertion with:

```python
def test_bin_2d_plot_draws_same_single_color_boundaries_on_five_metric_axes(bin_2d_figure):
    """五个交叉指标子图必须复用同一组单色最终箱分界。"""
    axes = bin_2d_figure.axes[:9]
    metric_axis_indexes = (2, 3, 4, 6, 7)
    reference = _boundary_collections(axes[metric_axis_indexes[0]])

    assert len(reference) == 1
    reference_segments = [
        _normalized_segment(segment)
        for segment in reference[0].get_segments()
    ]
    reference_color = reference[0].get_colors()[0]
    for axis_index in metric_axis_indexes[1:]:
        actual = _boundary_collections(axes[axis_index])
        assert len(actual) == 1
        assert [
            _normalized_segment(segment)
            for segment in actual[0].get_segments()
        ] == reference_segments
        assert actual[0].get_colors()[0] == pytest.approx(reference_color)
    for axis_index in (0, 1, 5, 8):
        assert _boundary_collections(axes[axis_index]) == []
```

In the missing-bin test, replace the per-bin gid set with this exact assertion:

```python
for axis_index in (2, 3, 4, 6, 7):
    assert {
        artist.get_gid()
        for artist in _boundary_collections(axes[axis_index])
    } == {"bin-2d-boundaries"}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
pytest tests/test_visualization/test_bin_plot_layout.py -q
```

Expected: FAIL because the existing helper returns one colored collection per final bin, includes outer-perimeter segments, and uses per-bin gids/colors.

- [ ] **Step 3: Implement the minimal single-color adjacency algorithm**

```python
def _draw_2d_bin_boundaries(
    ax,
    solution: np.ndarray,
    *,
    expected_shape: Optional[tuple] = None,
    color: str = "#222222",
    linewidth: float = 2.2,
) -> List[LineCollection]:
    """只在不同最终二维箱之间绘制一次单色分界线。"""
    solution = np.asarray(solution)
    if solution.ndim != 2 or solution.size == 0:
        raise ValueError("二维分箱映射必须是非空二维矩阵")
    if expected_shape is not None and solution.shape != tuple(expected_shape):
        raise ValueError(
            f"二维分箱映射形状 {solution.shape} 与热力图形状 {tuple(expected_shape)} 不一致"
        )

    display_solution = np.flipud(solution)
    n_rows, n_cols = display_solution.shape
    segments = []
    for row in range(n_rows):
        for col in range(n_cols - 1):
            if display_solution[row, col] != display_solution[row, col + 1]:
                x = col + 0.5
                segments.append(((x, row - 0.5), (x, row + 0.5)))
    for row in range(n_rows - 1):
        for col in range(n_cols):
            if display_solution[row, col] != display_solution[row + 1, col]:
                y = row + 0.5
                segments.append(((col - 0.5, y), (col + 0.5, y)))

    artist = LineCollection(
        segments,
        colors=[color],
        linewidths=linewidth,
        capstyle="butt",
        joinstyle="miter",
        zorder=4,
    )
    artist.set_gid("bin-2d-boundaries")
    ax.add_collection(artist)
    return [artist]
```

Delete the per-bin colormap construction from `bin_2d_plot`, call the helper directly for all five heatmaps, and remove the now-unused `SEQUENTIAL_GRADIENT` import from this module.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
pytest tests/test_visualization/test_bin_plot_layout.py -q
```

Expected: all tests pass; only the pre-existing pandas dependency warnings may remain.

- [ ] **Step 5: Run broader 2D binning and visualization regression tests**

Run:

```powershell
pytest tests/test_binning/test_optimal_binning_2d.py tests/test_visualization -q
```

Expected: all collected tests pass with no new failures.

- [ ] **Step 6: Generate and inspect a real-data plot**

Using `examples/hscredit_yyp.xlsx`, fit `OptimalBinning2D` for the C3 score column and `CURRENT_DPD`, save the resulting `bin_2d_plot` to the task visualization directory, and inspect the PNG. Confirm the five heatmaps show dark internal final-bin separators, no colored outer-bin perimeter, unchanged cell values/colors, and intact blue axes frames.

- [ ] **Step 7: Commit the implementation**

```powershell
git add -- hscredit/core/viz/binning_plots.py tests/test_visualization/test_bin_plot_layout.py
git commit -m "fix: clarify 2d bin plot boundaries"
```
