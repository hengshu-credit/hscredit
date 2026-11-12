"""核对任务详情的工作簿内容、分页和历史任务兼容性。"""

from io import BytesIO
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image
from PIL import Image as PillowImage

from hscredit.platform.server import create_app
from hscredit.platform.storage import write_json


@pytest.fixture
def jobs(tmp_path):
    with TestClient(create_app(tmp_path)) as client:

        def record(artifacts, result=None, status="成功"):
            identifier = uuid4().hex
            directory = tmp_path / "jobs" / identifier
            directory.mkdir()
            write_json(directory / "metadata.json", {"id": identifier, "kind": "strategy", "created_at": "2026-01-01", "status": status, "config": {}, "artifacts": artifacts})
            if result is not None:
                write_json(directory / "result.json", result)
            return directory, f"/api/jobs/{identifier}"

        yield client, record


def test_native_report_matches_download_across_sheets_and_pages(jobs):
    client, record = jobs
    directory, url = record(["report.xlsx", "comparison.xlsx"])
    book = Workbook()
    book.active.title = "分箱"
    book.active.append(["分箱边界", "命中数"])
    book.active.append(["≤10", 12])
    book.active["C2"] = 0.125
    book.active["C2"].number_format = "0.00%"
    evaluation = book.create_sheet("评估")
    evaluation["AE51"] = 0.8125
    PillowImage.new("RGB", (4, 3), "red").save(directory / "picture.png")
    evaluation.add_image(Image(directory / "picture.png"), "C5")
    book.save(directory / "report.xlsx")
    book.remove(evaluation)
    book.save(directory / "comparison.xlsx")
    preview = url + "/workbooks/report.xlsx"
    assert client.get(preview).json()["rows"][1] == ["≤10", "12", "12.50%"]
    page = client.get(preview, params={"sheet": "评估", "page": 2, "column_page": 2}).json()
    assert (page["sheets"], page["first_row"], page["columns"], page["rows"]) == (["分箱", "评估"], 51, ["AE"], [["0.8125"]])
    assert page["image_count"] == 1
    assert page["pictures"] == []
    assert client.get(preview, params={"sheet": "评估"}).json()["pictures"] == [{"index": 0, "row": 5, "col": 3}]
    image = client.get(preview + "/images/0", params={"sheet": "评估"})
    assert image.headers["content-type"] == "image/png"
    assert image.content == (directory / "picture.png").read_bytes()
    assert client.get(url + "/workbooks/comparison.xlsx").json()["sheets"] == ["分箱"]
    downloaded = load_workbook(BytesIO(client.get(url + "/files/report.xlsx").content))
    assert downloaded["评估"]["AE51"].value == float(page["rows"][0][0])
    assert client.get(preview, params={"sheet": "未知"}).status_code == 400
    assert client.get(preview, params={"page": 3}).status_code == 400
    assert client.get(url + "/workbooks/other.xlsx").status_code == 404


def test_report_keeps_merged_labels_internal_navigation_and_number_formats(jobs):
    from openpyxl.styles import Alignment, Font

    client, record = jobs
    directory, url = record(["report.xlsx"])
    book = Workbook()
    ws = book.active
    ws.title = "目录"
    ws["B2"] = "模型评估报告"
    ws.merge_cells("B2:E2")
    ws["B2"].font = Font(bold=True)
    ws["B4"] = "查看评估指标"
    ws["B4"].hyperlink = "#'模型''评估'!$AE$51"
    ws["B5"] = "外部说明"
    ws["B5"].hyperlink = "https://example.org/"
    metrics = book.create_sheet("模型'评估")
    metrics["AD49"] = "测试集"
    metrics["AD49"].alignment = Alignment(horizontal="center")
    metrics.merge_cells("AD49:AF53")
    metrics["AG51"] = 1234.56789
    metrics["AG51"].alignment = Alignment(horizontal="right")
    metrics["AG51"].number_format = "#,##0.00"
    book.save(directory / "report.xlsx")
    route = url + "/workbooks/report.xlsx"
    directory_page = client.get(route).json()
    assert directory_page["outline"] == [{"text": "模型评估报告", "col": "B", "row": 2}]
    cells = directory_page["cells"]
    assert [(c["text"], c["colspan"], c["bold"]) for c in cells[:1]] == [("模型评估报告", 4, True)]
    assert cells[1]["link"] == {"sheet": metrics.title, "col": "AE", "row": 51}
    assert "link" not in cells[2]
    page = client.get(route, params={"sheet": metrics.title, "page": 2, "column_page": 2}).json()
    merged, number = page["cells"]
    assert (merged["text"], merged["row"], merged["col"], merged["rowspan"], merged["colspan"]) == ("测试集", 51, 31, 3, 2)
    assert number["text"] == "1,234.57" and number["numeric"]
    assert (merged["align"], number["align"]) == ("center", "right")
    assert page["rows"][0] == ["", "", "1,234.57"]
    window = client.get(route, params={"sheet": metrics.title, "page": 2, "first_col": 32, "column_count": 2, "include_catalog": "false"}).json()
    assert window["columns"] == ["AF", "AG"] and window["sheets"] == []
    assert window["cells"][0]["merge"] == [30, 49, 32, 53]
    assert window["cells"][0]["colspan"] == 1 and window["cells"][1]["text"] == "1,234.57"
    assert client.get(route, params={"column_count": 999}).status_code == 422


def test_historical_report_reuses_saved_result_without_dataset(jobs):
    client, record = jobs
    directory, url = record([], {"tables": {"收益 / 命中": [{"名称": "=保留原值", "样本数": 23}]}})
    (directory / "report.xlsx").write_bytes(b"unpublished")
    assert client.get(url).json()["artifacts"] == ["report.xlsx"]
    preview = client.get(url + "/workbooks/report.xlsx").json()
    assert preview["rows"] == [["名称", "样本数"], ["=保留原值", "23"]]
    book = load_workbook(BytesIO(client.get(url + "/files/report.xlsx").content))
    assert book.active["A2"].data_type == "s"
    assert book.active["B2"].value == 23
    assert book.active["B1"].alignment.horizontal == book.active["B2"].alignment.horizontal == "right"
    assert client.get(url + "/files/history-report.xlsx").status_code == 404


@pytest.mark.parametrize("status", ["排队中", "运行中", "停止中", "已停止", "失败"])
def test_partial_workbooks_stay_private(jobs, status):
    client, record = jobs
    directory, url = record(["report.xlsx"], status=status)
    (directory / "report.xlsx").touch()
    for route in ("/files/report.xlsx", "/workbooks/report.xlsx", "/workbooks/report.xlsx/images/0?sheet=表"):
        assert client.get(url + route).status_code == 404


def test_workbook_page_carries_only_headers_for_a_continuing_table(jobs):
    from openpyxl.styles import Font

    client, record = jobs
    directory, url = record(["report.xlsx"])
    book = Workbook()
    sheet = book.active
    for row in (["订单", None, None, "金额", None], ["分箱", "坏样本率", None, "分箱", "坏样本率"]):
        sheet.append(row)
    sheet.merge_cells("A1:B1")
    sheet.merge_cells("D1:E1")
    for row in sheet.iter_rows(max_row=2):
        for cell in row:
            if cell.value:
                cell.font = Font(bold=True)
    for r in range(3, 53):
        sheet.append([r, 0.1, None, r, 0.2])
    sheet["A101"] = "下一章"
    book.save(directory / "report.xlsx")
    route = url + "/workbooks/report.xlsx"
    continuation = client.get(route, params={"page": 2}).json()["header_context"]
    assert (continuation["first_row"], continuation["last_row"]) == (1, 2)
    assert [c["text"] for c in continuation["cells"]] == ["订单", "金额", "分箱", "坏样本率", "分箱", "坏样本率"]
    assert client.get(route, params={"page": 3}).json()["header_context"] is None
