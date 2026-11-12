"""验证数据管理的跨页相关性、混合类型与版本隔离。"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy.stats.contingency import association

from hscredit.platform import create_app


def test_all_field_pairs_across_pages_and_versions(tmp_path):
    frame = pd.DataFrame({f"字段{i}": range(8) for i in range(40)})
    frame["金额"] = [10, 40, 90, 160, np.inf, 360, 490, np.nan]
    frame["反向金额"] = -frame["金额"]
    frame["渠道"] = ["甲", "甲", "乙", "乙", "丙", None, "乙", "甲"]
    frame["客群"] = ["一", "一", "一", "二", "二", "二", "二", None]
    frame["常量"] = 1
    frame["空列"] = np.nan
    frame["是否逾期"] = [True, False] * 4
    frame["授信日期"] = pd.date_range("2025-01-01", periods=8).strftime("%Y-%m-%d")
    frame.loc[7, "授信日期"] = None
    frame["编号"] = 2**60 + np.arange(8)
    with TestClient(create_app(tmp_path)) as client:
        workspace = client.app.state.workspace
        dataset = workspace.add_dataset(frame, "宽表样本")
        url = f"/api/datasets/{dataset['id']}/correlation"

        def block(row, column, version=1):
            response = client.get(url, params={"row_page": row, "column_page": column, "revision": version})
            assert response.status_code == 200, response.text
            matrix = response.json()
            assert matrix["revision"] == version
            assert matrix["total_fields"] == len(frame.columns)
            return pd.DataFrame(matrix["values"], index=matrix["rows"], columns=matrix["columns"], dtype=float)

        same = block(3, 3)
        assert same.loc["金额", "反向金额"] == pytest.approx(-1)
        categories = pd.crosstab(frame["渠道"], frame["客群"])
        assert same.loc["渠道", "客群"] == pytest.approx(association(categories.to_numpy(), method="cramer"))
        valid = frame[["金额", "渠道"]].replace(np.inf, np.nan).dropna()
        fitted = valid.groupby("渠道")["金额"].transform("mean")
        eta = np.sqrt(fitted.var() / valid["金额"].var())
        assert same.loc["金额", "渠道"] == pytest.approx(eta)
        assert same.loc["渠道", "渠道"] == pytest.approx(1)
        assert same[["常量", "空列"]].isna().all().all()
        assert same.loc["是否逾期", "字段39"] == pytest.approx(frame["是否逾期"].corr(frame["字段39"]))
        assert same.loc["授信日期", "字段39"] == pytest.approx(1)
        assert block(4, 1).loc["编号", "字段0"] == pytest.approx(1)
        pd.testing.assert_frame_equal(block(1, 3).T, block(3, 1))
        assert client.get(url, params={"column_page": 5}).status_code == 400
        workspace.replace_dataset(dataset["id"], frame.assign(反向金额=frame["金额"]), expected_revision=1, processing={"description": "调整字段"})
        assert block(3, 3, version=2).loc["金额", "反向金额"] == pytest.approx(1)
        assert block(3, 3).loc["金额", "反向金额"] == pytest.approx(-1)


def test_undefined_pairs_and_independent_categories(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        frame = pd.DataFrame({"渠道": ["甲", "甲", "乙", "乙"], "区域": ["南", "北"] * 2,
                              "全空": [None] * 4, "常量": ["值"] * 4, "仅一条": [4, np.nan, np.nan, np.nan]})
        dataset = client.app.state.workspace.add_dataset(frame, "类别样本")
        matrix = client.get(f"/api/datasets/{dataset['id']}/correlation").json()
        assert matrix["values"][0] == [1, 0, None, None, None]
        assert matrix["values"][4] == [None] * 5
