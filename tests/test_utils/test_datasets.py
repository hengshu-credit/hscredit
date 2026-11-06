"""内置现金贷数据集的原值保留、字段隔离和公开加载接口验证。"""

import os
from datetime import date
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

from hscredit import hscredit_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "hscredit/resources/datasets/hscredit.parquet"
ORDER_COLUMNS = ["放款编码", "放款银行", "流量渠道", "用户编码", "loan_amt", "periods", "授信时间", "放款时间", "etl_date"]
OVERDUE_COLUMNS = ["DPD", *[f"MOB{i}" for i in range(1, 13)], "CURRENT_DPD"]


def test_dataset_population_and_observation_window():
    with pq.ParquetFile(DATA_PATH) as parquet:
        assert (parquet.metadata.num_rows, parquet.metadata.num_columns) == (111929, 8516)
    order = hscredit_dataset(columns=ORDER_COLUMNS)
    assert order["放款编码"].is_unique and order["放款编码"].notna().all()
    assert order["用户编码"].nunique() == 110283
    assert order["放款银行"].nunique() == 6
    assert order["流量渠道"].nunique() == 8
    assert set(order["etl_date"]) == {date(2025, 11, 7)}
    assert order["授信时间"].min() == pd.Timestamp("2023-12-15 09:01:02")
    assert order["授信时间"].max() == pd.Timestamp("2024-06-17 21:34:05")
    assert order["放款时间"].min() == pd.Timestamp("2023-12-15 00:00:00")
    assert order["放款时间"].max() == pd.Timestamp("2024-06-17 23:57:25")


def test_missing_sentinel_and_feature_coverage():
    all_missing = []
    with pq.ParquetFile(DATA_PATH) as parquet:
        features = [name for name in parquet.schema_arrow.names if name not in ORDER_COLUMNS + OVERDUE_COLUMNS]
        for start in range(0, len(features), 64):
            block = parquet.read(columns=features[start : start + 64]).to_pandas()
            empty = block.isna().all()
            all_missing.extend(empty.index[empty])
    assert len(all_missing) == 74
    assert "DJK_X1701" in all_missing

    sample = hscredit_dataset(columns=["DJK_X1342", "DK_X311"])
    assert sample["DJK_X1342"].isna().sum() == 14207 + 68508
    assert not sample["DJK_X1342"].eq(-1).any()
    # DK_X311 同时含有其他负数，按规则保留原有的一笔 -1。
    assert sample["DK_X311"].eq(-1).sum() == 1
    assert sample["DK_X311"].lt(-1).any()


@pytest.fixture(scope="module")
def sample_package(tmp_path_factory):
    """取真实订单的全部字段，在受限内存中验证无参数加载和全列拆分。"""
    package = tmp_path_factory.mktemp("hscredit")
    dataset_dir = package / "resources" / "datasets"
    dataset_dir.mkdir(parents=True)
    with pq.ParquetFile(DATA_PATH) as parquet:
        names = parquet.schema_arrow.names
        arrays = []
        for start in range(0, len(names), 64):
            batch = next(parquet.iter_batches(batch_size=8, columns=names[start : start + 64]))
            arrays.extend(batch.columns)
        sample = pa.Table.from_arrays(arrays, schema=parquet.schema_arrow)
    pq.write_table(sample, dataset_dir / "hscredit.parquet")
    return package


def test_default_load_preserves_all_rows_columns_and_original_values(sample_package, monkeypatch):
    monkeypatch.setattr("hscredit.utils.datasets.files", lambda _: sample_package)
    data = hscredit_dataset()
    with pq.ParquetFile(DATA_PATH) as parquet:
        assert data.shape == (8, parquet.metadata.num_columns)
        assert data.columns.tolist() == parquet.schema_arrow.names
    columns = ORDER_COLUMNS + OVERDUE_COLUMNS + ["sex", "province", "age_year", "DJK_X986", "DK_BALANCE", "IS_PUBLIC_LIST", "LZ_SCORE"]
    expected = pd.read_parquet(DATA_PATH, columns=columns).iloc[:8]
    assert_frame_equal(data[columns], expected)
    assert data.index.equals(pd.RangeIndex(8))
    assert "target" not in data and "FPD" not in data


def test_full_split_is_exhaustive_and_excludes_order_and_outcome_from_features(sample_package, monkeypatch):
    monkeypatch.setattr("hscredit.utils.datasets.files", lambda _: sample_package)
    order, overdue, X = hscredit_dataset(split=True)
    assert order.columns.tolist() == ORDER_COLUMNS
    assert overdue.columns.tolist() == OVERDUE_COLUMNS
    with pq.ParquetFile(DATA_PATH) as parquet:
        assert X.columns.tolist() == [column for column in parquet.schema_arrow.names if column not in ORDER_COLUMNS + OVERDUE_COLUMNS]
    assert order.shape == (8, 9)
    assert overdue.shape == (8, 14)
    assert X.shape == (8, 8493)
    assert order.index.equals(overdue.index) and order.index.equals(X.index)
    # 授信前历史逾期仍属于 X，本笔借据贷后表现才归入 overdue。
    assert "MONTH6_OVER_NUM_COUNT" in X
    assert_frame_equal(order, pd.read_parquet(DATA_PATH, columns=ORDER_COLUMNS).iloc[:8])
    assert_frame_equal(overdue, pd.read_parquet(DATA_PATH, columns=OVERDUE_COLUMNS).iloc[:8])
    columns = ["age_year", "sex", "province", "MONTH6_OVER_NUM_COUNT", "DJK_X986"]
    assert_frame_equal(X[columns], pd.read_parquet(DATA_PATH, columns=columns).iloc[:8])


def test_projection_split_can_reconstruct_original_and_does_not_share_state():
    columns = ["MOB12", "sex", "用户编码", "DPD", "age_year", "授信时间", "DJK_X986", "CURRENT_DPD"]
    full = hscredit_dataset(columns=columns)
    parts = hscredit_dataset(split=True, columns=columns)
    assert full.columns.tolist() == columns
    assert_frame_equal(pd.concat(parts, axis=1)[columns], full)
    parts[2].iloc[0, 0] = "修改后的值"
    assert_frame_equal(hscredit_dataset(columns=columns), full)


@pytest.mark.parametrize("columns", [[], ["MOB1"], ["age_year"], ["用户编码"]])
def test_empty_groups_preserve_row_alignment(columns):
    full = hscredit_dataset(columns=columns)
    parts = hscredit_dataset(split=True, columns=columns)
    assert full.shape == (111929, len(columns))
    assert all(frame.index.equals(full.index) for frame in parts)
    assert_frame_equal(pd.concat(parts, axis=1)[columns], full)


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"split": "yes"}, TypeError, "布尔"),
        ({"columns": "age_year"}, TypeError, "序列"),
        ({"columns": [1]}, TypeError, "字符串"),
        ({"columns": ["MOB1", "MOB1"]}, ValueError, "重复"),
        ({"columns": ["不存在的字段"]}, ValueError, "不存在字段"),
    ],
)
def test_invalid_options_fail_explicitly(options, error, message):
    with pytest.raises(error, match=message):
        hscredit_dataset(**options)


@pytest.mark.parametrize("light_import", ["0", "1"])
def test_public_exports_and_loading_outside_repository(tmp_path, light_import):
    code = """
import hscredit
from hscredit import hscredit_dataset
from hscredit.utils import hscredit_dataset as utility_loader
from hscredit.utils.datasets import hscredit_dataset as dataset_loader
assert hscredit_dataset is utility_loader is dataset_loader
assert 'hscredit_dataset' in hscredit.__all__
assert hscredit_dataset.__name__ == 'hscredit_dataset'
assert hscredit_dataset(columns=['MOB1']).shape == (111929, 1)
"""
    env = dict(os.environ, HSCREDIT_LIGHT_IMPORT=light_import, PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr
