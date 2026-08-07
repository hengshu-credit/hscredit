# `bin_2d_plot` Inset Colored Boundaries Implementation Plan

**Goal:** Draw a distinct, slightly inset colored outline for every final 2D bin and add its index at the bin's top-left, without changing heatmap values, fills, white pre-bin grids, or axes frames.

**Architecture:** Convert `solution_` to display orientation, extract each bin's exposed cell sides, shift every side inward by `0.04` cell units, and connect shifted endpoints at straight seams and concave corners. Allocate colors once from the project's categorical palette and reuse the mapping across all five metric heatmaps. Anchor one white-background, black-text index at the leftmost cell of each bin's highest displayed row.

**Tech Stack:** Python 3, NumPy, Matplotlib `LineCollection`, pytest.

## Constraints

- Shared boundaries show both neighboring bin colors on their respective sides.
- Every colored point is strictly inside the heatmap frame.
- Same-bin cell interfaces have no colored line.
- Existing cell metrics, heatmap colors, annotations, and white grids remain unchanged.
- Each final bin shows its actual ID once as a white-background, black-text index; no legends or point markers are added.
- Missing-value rows and columns participate through `solution_`.
- Public `bin_2d_plot` parameters and `OptimalBinning2D` remain unchanged.

## Task 1: Regression tests

**Files:**

- Modify: `tests/test_visualization/test_bin_plot_layout.py`

1. Replace the exact-edge colored-outline test with an L-shaped mapping assertion.
2. Verify the two sides of a shared boundary are at `0.46` and `0.54` for the default `0.04` inset.
3. Verify same-bin internal interfaces have no parallel colored sides.
4. Verify every outline coordinate is strictly inside the heatmap extent.
5. Verify a single final bin still receives one complete inset outline.
6. Verify all five metric axes reuse identical per-bin gids, segments, and distinct colors.
7. Verify missing-bin plots contain one outline collection for every ID in `solution_`.
8. Verify each final bin receives exactly one correctly positioned white-background, black-text index and that all five metric axes reuse the same labels.
9. Run:

   ```powershell
   pytest tests/test_visualization/test_bin_plot_layout.py -q
   ```

   Expected before implementation: failures caused by missing index labels.

## Task 2: Inset categorical outlines

**Files:**

- Modify: `hscredit/core/viz/binning_plots.py`

1. Restore per-bin `LineCollection` output with gid `bin-2d-boundary-<id>`.
2. Generate only the four cell sides exposed to another bin or the matrix exterior.
3. Shift each side and endpoint inward by `inset=0.04`.
4. At each original grid vertex, connect the shifted endpoint pair; when four endpoints represent diagonal components, pair endpoints by source cell.
5. Use `get_series_colors` to create a categorical `bin_id -> color` mapping once in `bin_2d_plot`.
6. Pass the same mapping to all five metric heatmaps.
7. Add one `bin-2d-index-<id>` text artist per final bin at the leftmost cell of its highest displayed row.
8. Style each index with an opaque white background and black text, offset inside the cell corner.
9. Run the focused test file and require all 20 tests to pass.

## Task 3: Regression and visual verification

1. Run:

   ```powershell
   pytest tests/test_binning/test_optimal_binning_2d.py tests/test_visualization -q
   ```

   Expected: all 100 tests pass.

2. Fit `OptimalBinning2D` against `examples/hscredit_yyp.xlsx` for the C3 score column and `CURRENT_DPD`.
3. Save and inspect the generated `bin_2d_plot`.
4. Confirm every metric axis has one collection and one distinct category color per final bin, one white-background black index per bin, all colored points are strictly inset, shared edges show both colors, and blue axes frames remain intact.
