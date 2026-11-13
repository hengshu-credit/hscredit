"""文本表达式与配置树之间的语义往返及安全边界。"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hscredit.web import create_app
from hscredit.web.processing import parse_expression_text, process_frame


@pytest.mark.parametrize(
    "text,expected",
    [
        ('where(col("年 龄") >= 20, 收入 * 2, -1)', [-1, 4, 6]),
        ("1 + 收入 * (3 - 1)", [3, 5, 7]),
        ("where(1 < 收入 < 3, 1, 0)", [0, 1, 0]),
        ('where(not (收入 > 1), "低", "高")', ["低", "高", "高"]),
        ("-收入 + 10", [9, 8, 7]),
        ("np.sqrt(收入 * 收入)", [1, 2, 3]),
        ("where(收入 > 2, nan, 收入)", [1, 2, np.nan]),
    ],
)
def test_text_tree_roundtrip_keeps_results(text, expected):
    data = pd.DataFrame({"收入": [1, 2, 3], "年 龄": [18, 20, 30]})
    parsed = parse_expression_text(text, {"收入": "numeric", "年 龄": "numeric"})
    reparsed = parse_expression_text(parsed["text"], {"收入": "numeric", "年 龄": "numeric"})
    output, _, _ = process_frame(data, [{"mode": "add", "target": "结果", "expression": reparsed["expression"]}])
    pd.testing.assert_series_equal(output["结果"], pd.Series(expected, name="结果"), check_dtype=False)


@pytest.mark.parametrize("text", ['__import__("os").system("echo bad")', "收入.__class__", "[x for x in 收入]", "收入[0]", "lambda: 1", 'open("file")', "where(收入 > 1, 1)", "不存在 + 1", "收入 +"])
def test_invalid_text_is_rejected_without_evaluation(text):
    with pytest.raises(ValueError):
        parse_expression_text(text, {"收入": "numeric"})


def test_parse_api_uses_prior_fields_and_rejects_stale_revision(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        dataset = client.app.state.workspace.add_dataset(pd.DataFrame({"收入": [1, 2]}), "编辑表达式")
        endpoint = f"/api/datasets/{dataset['id']}/processing/parse"
        previous = {"mode": "add", "target": "新收入", "expression": {"kind": "FIELD", "name": "收入"}}
        response = client.post(endpoint, json={"text": "新收入 * 2", "expected_revision": 1, "prior_rules": [previous]})
        assert response.status_code == 200, response.text
        assert 'col("新收入")' in response.json()["text"]
        assert client.post(endpoint, json={"text": "收入 + 1", "expected_revision": 2}).status_code == 409
        assert client.post(endpoint, json={"text": "收入 + 1", "expected_revision": 1, "trusted_python": True}).status_code == 422
        assert client.app.state.workspace.read_dataset(dataset["id"]).columns.tolist() == ["收入"]
