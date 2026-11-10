"""验证报告统一的 save/sheet 类型、工作簿内容及调用方资源所有权。"""

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook, load_workbook
from sklearn.linear_model import LinearRegression, LogisticRegression

from hscredit.excel import ExcelWriter
from hscredit.report import auto_feature_analysis, auto_model_report, swap_out_report
from hscredit.report.mining import MultiwayTreeExtractor


@pytest.fixture
def sample():
    return pd.DataFrame({"评分": np.arange(24), "MOB1": [0, 2, 5, 9] * 6, "金额": np.arange(24) + 100})


def generate(kind, data, tmp_path, **output):
    labels = dict(overdue="MOB1", dpds=[3, 7], amount="金额")
    if kind in {"model", "regression"}:
        model = (LogisticRegression() if kind == "model" else LinearRegression()).fit(data[["评分"]], data.MOB1.gt(3) if kind == "model" else data["金额"])
        options = labels if kind == "model" else dict(target="金额")
        return auto_model_report(model, datasets={"验证集": data}, features=["评分"], long_format=True, with_plots=False, verbose=False, n_jobs=1, **options, **output)
    if kind == "feature":
        return auto_feature_analysis(data, features=["评分"], long_format=True, pictures=[], output_dir=str(tmp_path / "图表"), n_jobs=1, show_progress=False, **labels, **output)
    if kind == "swap":
        return swap_out_report(data, "评分 < 12", features=[], n_jobs=1, **labels, **output)
    tree = MultiwayTreeExtractor(target="MOB1", features=["评分"], n_jobs=1, max_depth=1).fit(data)
    result = tree.report(data, methods=[], **labels, **output)
    assert result.iloc[0]["样本总数"] == data["金额"].sum()
    return result


@pytest.mark.parametrize("kind", ["model", "regression", "feature", "swap", "tree"])
@pytest.mark.parametrize("sheet_object", [False, True])
def test_path_and_writer_export_same_contents(kind, sheet_object, sample, tmp_path):
    reference = tmp_path / "参考.xlsx"
    generate(kind, sample, tmp_path, save=reference, sheet="统一报告")
    writer = ExcelWriter()
    untouched = writer.get_sheet_by_name("既有数据")
    untouched["A1"] = "保留内容"
    untouched.column_dimensions["B"].width = 37
    sheet = writer.get_sheet_by_name("统一报告") if sheet_object else "统一报告"
    result = generate(kind, sample, tmp_path, save=writer, sheet=sheet)
    if sheet_object:
        assert writer.workbook["统一报告"] is sheet
    assert untouched["A1"].value == "保留内容"
    assert untouched.column_dimensions["B"].width == 37
    if kind == "model":
        table = result.get_bin_table("验证集", labels=["MOB1@3", "MOB1@7"], amount="金额")
        totals = table[table["分箱标签"] == "合计"].set_index("逾期标签")
        for dpd in (3, 7):
            assert totals.loc[f"MOB1@{dpd}", "样本总数"] == sample["金额"].sum()
            assert totals.loc[f"MOB1@{dpd}", "坏样本数"] == sample.loc[sample.MOB1.gt(dpd), "金额"].sum()
    output = tmp_path / "合并.xlsx"
    writer.save(output)
    expected, actual = load_workbook(reference), load_workbook(output)
    try:
        assert list(actual["统一报告"].values) == list(expected["统一报告"].values)
        if kind in {"model", "regression"}:
            assert any(cell.hyperlink and "统一报告" in (cell.hyperlink.target or cell.hyperlink.location or "") for row in actual["目录"] for cell in row)
    finally:
        expected.close()
        actual.close()


@pytest.mark.parametrize("kind", ["model", "regression", "feature", "swap", "tree"])
def test_raw_openpyxl_sheet_with_filename_preserves_identity(kind, sample, tmp_path):
    book = Workbook()
    sheet = book.active
    sheet.title = "自定义报告"
    other = book.create_sheet("原始数据")
    other["A1"] = 123
    path = tmp_path / "外部工作表.xlsx"
    generate(kind, sample, tmp_path, save=path, sheet=sheet)
    assert book["自定义报告"] is sheet
    assert sheet.max_row > 2 and other["A1"].value == 123
    # 仍可继续使用调用方的 Workbook，并保存报告图片。
    book.save(tmp_path / "再次保存.xlsx")
    book.close()
    assert path.is_file()


@pytest.mark.parametrize("kind", ["model", "regression", "feature", "swap", "tree"])
def test_foreign_sheet_is_rejected(kind, sample, tmp_path):
    writer = ExcelWriter()
    try:
        with pytest.raises(ValueError, match="属于.*writer"):
            generate(kind, sample, tmp_path, save=writer, sheet=Workbook().active)
    finally:
        writer.workbook.close()


@pytest.mark.parametrize("kind", ["model", "feature", "swap", "tree"])
@pytest.mark.parametrize("keyword", ["excel_path", "excel_writer", "sheet_name"])
def test_removed_output_names_raise(kind, keyword, sample, tmp_path):
    with pytest.raises(TypeError, match=keyword):
        generate(kind, sample, tmp_path, **{keyword: "旧参数.xlsx"})


def test_shared_feature_writer_supports_multiple_reports_and_images(sample, tmp_path):
    writer = ExcelWriter()
    options = dict(features=["评分"], overdue="MOB1", dpds=[3, 7], amount="金额", long_format=True, pictures=["bin"], output_dir=str(tmp_path / "图表"), n_jobs=1, show_progress=False)
    for name in ("第一份", "第二份"):
        auto_feature_analysis(sample, save=writer, sheet=name, **options)
    path = tmp_path / "合并.xlsx"
    writer.save(path)
    book = load_workbook(path)
    try:
        assert book.sheetnames == ["第一份", "第二份"]
        assert all(sheet._images for sheet in book)
    finally:
        book.close()


@pytest.mark.parametrize("value,exception", [(True, TypeError), (123, TypeError), ("", ValueError), ("  ", ValueError)])
def test_invalid_save_fails_before_model_prediction(value, exception):
    with pytest.raises(exception, match="save"):
        auto_model_report(object(), save=value)


@pytest.mark.parametrize("sheet", ["目录", "2-模型性能"])
def test_model_rejects_sheet_names_reserved_for_other_sections(sample, tmp_path, sheet):
    with pytest.raises(ValueError, match="其他章节重名"):
        generate("model", sample, tmp_path, save=tmp_path / "冲突.xlsx", sheet=sheet)
    assert not (tmp_path / "冲突.xlsx").exists()


@pytest.mark.parametrize("regression", [False, True])
def test_model_writer_keeps_images_after_temporary_assets_are_cleaned(sample, tmp_path, regression):
    writer = ExcelWriter()
    model = (LinearRegression() if regression else LogisticRegression()).fit(sample[["评分"]], sample["金额"] if regression else sample.MOB1.gt(3))
    options = dict(target="金额") if regression else dict(overdue="MOB1", dpds=[3, 7])
    report = auto_model_report(model, datasets={"验证集": sample}, features=["评分"], save=writer, sheet="基本信息", n_jobs=1, verbose=False, with_plots=True, **options)
    assert report.last_report_result_.artifacts == []
    assert report.last_report_result_.metadata["写入工作簿"]
    path = tmp_path / "带图模型.xlsx"
    writer.save(path)
    book = load_workbook(path)
    try:
        assert sum(len(sheet._images) for sheet in book) > 0
    finally:
        book.close()
