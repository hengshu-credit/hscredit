"""验证 hscredit_hsk.xlsx 上全部分箱方法的类别变量支持。"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from hscredit.core.binning import OptimalBinning
from hscredit.core.binning.or_binning import ORTOOLS_AVAILABLE


METHODS = [
    "uniform",
    "quantile",
    "tree",
    "chi",
    "best_ks",
    "best_iv",
    "mdlp",
    "or_tools",
    "cp_sat",
    "cart",
    "kmeans",
    "monotonic",
    "genetic",
    "smooth",
    "kernel_density",
    "best_lift",
    "target_bad_rate",
]


def _method_kwargs(method: str) -> Dict:
    kwargs: Dict = {"lift_refine": False}
    if method == "genetic":
        kwargs.update(population_size=12, generations=4)
    elif method == "or_tools":
        kwargs.update(or_time_limit=2, n_prebins=10, max_candidates=20)
    elif method == "cp_sat":
        kwargs.update(cp_sat_time_limit=2, cp_sat_n_prebins=10, max_candidates=20)
    elif method == "kernel_density":
        kwargs.update(n_grid_points=128)
    elif method == "smooth":
        kwargs.update(n_prebins=20)
    elif method == "best_lift":
        kwargs.update(n_prebins=20, max_bin_size=None)
    return kwargs


def _make_binner(method: str, wage_column: str, wage_order: List) -> OptimalBinning:
    return OptimalBinning(
        method=method,
        max_n_bins=5,
        min_n_bins=1,
        min_bin_size=1,
        cat_cutoff=10,
        category_order={wage_column: wage_order},
        random_state=7,
        verbose=False,
        **_method_kwargs(method),
    )


def _expected_bad_rate_order(x: pd.Series, y: pd.Series) -> List:
    first_seen = x.dropna().drop_duplicates().tolist()
    return sorted(first_seen, key=lambda value: (float(y.loc[x == value].mean()), first_seen.index(value)))


def _assert_group_values_equal(actual: Iterable, expected: Iterable) -> None:
    actual = list(actual)
    expected = list(expected)
    assert len(actual) == len(expected)
    for actual_group, expected_group in zip(actual, expected):
        assert len(actual_group) == len(expected_group)
        for actual_value, expected_value in zip(actual_group, expected_group):
            if pd.isna(expected_value):
                assert pd.isna(actual_value)
            else:
                assert actual_value == expected_value


def validate_wage_rules(X: pd.DataFrame, y: pd.Series, wage_column: str, wage_order: List) -> None:
    wage = X[[wage_column]]
    automatic = OptimalBinning(
        method="uniform",
        min_n_bins=1,
        max_n_bins=5,
        min_bin_size=1,
    ).fit(wage, y)
    assert automatic._category_orders_[wage_column] == _expected_bad_rate_order(wage[wage_column], y)

    explicit = _make_binner("uniform", wage_column, wage_order).fit(wage, y)
    assert explicit._category_orders_[wage_column] == wage_order

    chunks = [wage_order[:4], wage_order[4:8], wage_order[8:]]
    missing_alone = [*chunks, [np.nan]]
    alone_binner = OptimalBinning(
        user_splits={wage_column: missing_alone},
        strict_user_splits=True,
        min_n_bins=1,
        max_n_bins=4,
        min_bin_size=1,
    ).fit(wage, y)
    _assert_group_values_equal(alone_binner.export_rules()[wage_column], missing_alone)

    missing_mixed = [chunks[0], chunks[1], [*chunks[2], np.nan]]
    mixed_binner = OptimalBinning(
        user_splits={wage_column: missing_mixed},
        strict_user_splits=True,
        missing_separate=False,
        min_n_bins=1,
        max_n_bins=3,
        min_bin_size=1,
    ).fit(wage, y)
    _assert_group_values_equal(mixed_binner.export_rules()[wage_column], missing_mixed)

    values = wage.iloc[[0]].copy()
    values.iloc[0, 0] = np.nan
    assert mixed_binner.transform(values, metric="indices").iloc[0, 0] == 2

    loaded = OptimalBinning(missing_separate=False).import_rules(mixed_binner.export_rules())
    assert loaded.transform(values, metric="woe").iloc[0, 0] == 0.0
    print("工资验证：默认坏样本率排序、显式排序、缺失单独/混合分组、规则往返均通过")


def validate_method(method: str, X: pd.DataFrame, y: pd.Series, wage_column: str, wage_order: List) -> Dict:
    started = perf_counter()
    binner = _make_binner(method, wage_column, wage_order).fit(X, y)
    indices = binner.transform(X, metric="indices")
    labels = binner.transform(X, metric="bins")
    woe = binner.transform(X, metric="woe")

    assert indices.shape == X.shape
    assert labels.shape == X.shape
    assert woe.shape == X.shape
    assert not woe.isna().any().any()
    assert int((indices == -3).sum().sum()) == 0
    assert binner._category_orders_[wage_column] == wage_order

    rules = binner.export_rules()
    categorical_count = 0
    for feature in X.columns:
        table = binner.get_bin_table(feature)
        assert int(table["样本总数"].sum()) == len(X), feature
        if binner.feature_types_[feature] == "categorical":
            categorical_count += 1
            if X[feature].notna().any():
                assert rules[feature], feature

    return {
        "method": method,
        "seconds": perf_counter() - started,
        "categorical": categorical_count,
        "numerical": len(X.columns) - categorical_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workbook",
        nargs="?",
        type=Path,
        default=Path("examples/hscredit_hsk.xlsx"),
        help="待验证的工作簿路径",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.workbook)
    if "target" not in df.columns or "工资" not in df.columns:
        raise ValueError("工作簿必须包含 target 和 工资 字段")
    if len(df) != 3381:
        raise ValueError(f"hscredit_hsk.xlsx 行数应为 3381，实际为 {len(df)}")

    X = df.drop(columns="target")
    y = df["target"]
    original_target_counts = y.value_counts(dropna=False).to_dict()
    wage_order = X["工资"].dropna().drop_duplicates().tolist()
    validate_wage_rules(X, y, "工资", wage_order)

    passed = 0
    skipped = 0
    for method in METHODS:
        if method in {"or_tools", "cp_sat"} and not ORTOOLS_AVAILABLE:
            skipped += 1
            print(f"{method:16s} 跳过：未安装 OR-Tools")
            continue
        result = validate_method(method, X, y, "工资", wage_order)
        passed += 1
        print(
            f"{method:16s} 通过 | 类别={result['categorical']:2d} "
            f"数值={result['numerical']:2d} 用时={result['seconds']:.2f}s"
        )

    assert len(df) == 3381
    assert y.value_counts(dropna=False).to_dict() == original_target_counts == {0: 2024, 1: 1357}
    print(f"验证完成：通过 {passed} 种，跳过 {skipped} 种，样本 {len(df)} 行，字段 {len(X.columns)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
