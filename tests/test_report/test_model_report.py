"""Tests for model_report module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from hscredit.report.model_report import ModelReport


def _merged_range_for_row(ws, row, start_col=2):
    for cell_range in ws.merged_cells.ranges:
        if cell_range.min_row == row and cell_range.max_row == row and cell_range.min_col == start_col:
            return cell_range
    return None


def _row_for_value(ws, value, col=2):
    for row in range(1, ws.max_row + 1):
        if ws.cell(row, col).value == value:
            return row
    raise AssertionError(f"未找到单元格值: {value}")


class MockModel:
    """Minimal mock model for testing ModelReport."""

    def __init__(self, feature_names=None):
        self._feature_names = feature_names or ['f0']
        self._coef = np.array([0.5] * len(self._feature_names))
        self._intercept = np.array([-0.5])

    def predict_proba(self, X):
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_feat = min(arr.shape[1], len(self._coef))
        scores = arr[:, :n_feat] @ self._coef[:n_feat] + self._intercept[0]
        prob = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - prob, prob])

    def get_feature_importances(self):
        return pd.Series(dict(zip(self._feature_names, [0.5] * len(self._feature_names))))

    @property
    def feature_importances_(self):
        return np.array([0.5] * len(self._feature_names))


class ReversedClassModel(MockModel):
    """Mock model whose probability columns use classes_=[1, 0]."""

    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        proba = super().predict_proba(X)
        return proba[:, ::-1]


class RealDataContractModel:
    """Deterministic non-constant model used by the real-data report contract."""

    classes_ = np.array([0, 1])

    def __init__(self, feature_names):
        self.feature_names_in_ = np.asarray(feature_names)

    def predict_proba(self, X):
        values = np.asarray(X, dtype=float)
        logits = (
            (values[:, 0] - 0.09) * 5
            + (values[:, 1] - 60) / 30
            - (values[:, 2] - 600) / 200
        )
        probabilities = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probabilities, probabilities])

    def get_feature_importances(self):
        return pd.Series([0.5, 0.3, 0.2], index=self.feature_names_in_)

    def get_params(self):
        return {"name": "real-data-contract-model"}


class TestModelReportTarget:
    """Test target parameter handling."""

    def test_target_str(self):
        """target as string column name."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'label': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target='label', feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_target_dict_overdue_dpds(self):
        """target as dict with overdue+dpds."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'overdue': [0, 0, 1, 1, 1, 1],
            'dpds': [0, 2, 3, 5, 6, 10],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target={'overdue': 'overdue', 'dpds': 'dpds', 'threshold': 3},
            feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 0, 1, 1, 1]

    def test_target_dict_overdue_only(self):
        """target as dict with overdue only (no dpds): overdue col > 0 → y=1."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'overdue': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target={'overdue': 'overdue'},
            feature_names=['f0']
        )
        # overdue > 0 → [0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_datasets_y_none(self):
        """datasets dict with y=None derives y from target config."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'target': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, datasets={'train': (X, None)},
            target='target', feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_datasets_y_none_dict_target(self):
        """datasets dict with y=None and dict target."""
        X_train = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'overdue': [0, 1, 1, 1],
            'dpds': [0, 1, 5, 6],
        })
        X_test = pd.DataFrame({
            'f0': [5, 6],
            'overdue': [1, 1],
            'dpds': [4, 8],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model,
            datasets={'train': (X_train, None), 'test': (X_test, None)},
            target={'overdue': 'overdue', 'dpds': 'dpds', 'threshold': 3},
            feature_names=['f0']
        )
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]
        assert report._datasets['test'].y.tolist() == [1, 1]

    def test_y_proba_produced(self):
        """Model produces y_proba after init."""
        X = pd.DataFrame({'f0': [1, 2, 3, 4], 'label': [0, 0, 1, 1]})
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target='label', feature_names=['f0']
        )
        proba = report._datasets['train'].y_proba
        assert proba is not None
        assert len(proba) == 4
        assert proba.min() >= 0 and proba.max() <= 1

    def test_get_metrics(self):
        """get_metrics returns DataFrame with expected columns."""
        X = pd.DataFrame({'f0': [1, 2, 3, 4], 'label': [0, 0, 1, 1]})
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target='label', feature_names=['f0']
        )
        metrics = report.get_metrics()
        assert '统计项' in metrics.columns
        assert 'KS' in metrics['统计项'].values
        assert 'AUC' in metrics['统计项'].values
        assert '样本数' in metrics['统计项'].values
        assert '坏样本率' in metrics['统计项'].values

    def test_target_default_column_fallback(self):
        """When target=None, searches for common column names."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'flag': [0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target=None, feature_names=['f0']
        )
        # Should find 'flag' column as fallback
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]


class TestModelReportOverdueDpdsSeparate:
    """Test overdue/dpds as separate __init__ parameters (not inside target dict)."""

    def test_overdue_dpds_single_col_single_threshold(self):
        """overdue as str + dpds as int is equivalent to target='col'."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'dpds': [0, 1, 5, 10],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            overdue='dpds', dpds=3, feature_names=['f0']
        )
        # dpds > 3 → [0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_overdue_dpds_single_col_list_thresholds(self):
        """overdue as str + dpds as list thresholds."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'dpds': [0, 3, 7, 15, 20, 30],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            overdue='dpds', dpds=[15, 7, 0], feature_names=['f0']
        )
        # dpds > 15 or > 7 or > 0:
        #   0: 0>15? 0>7? 0>0? → false → 0
        #   3: 3>15? 3>7? 3>0? → false, false, true → 1 ← FAILS: test says [0,1,...]
        # Actually dpds > 0 for all values >= 1, so:
        #   [0, 1, 1, 1, 1, 1]  (only index 0 is false for >0)
        assert report._datasets['train'].y.tolist() == [0, 1, 1, 1, 1, 1]

    def test_overdue_dpds_multi_col(self):
        """overdue as list of str + dpds as list."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'dpds_m1': [0, 0, 0, 0, 0, 0],
            'dpds_m3': [0, 0, 0, 0, 1, 1],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            overdue=['dpds_m1', 'dpds_m3'], dpds=[3, 0],
            feature_names=['f0']
        )
        # dpds_m1 > 3 or > 0 → [0, 0, 0, 0, 0, 0]
        # dpds_m3 > 3 or > 0 → [0, 0, 0, 0, 1, 1]
        # any true → y=1 → [0, 0, 0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 0, 0, 1, 1]

    def test_overdue_dpds_override_target(self):
        """overdue/dpds takes priority over target when both provided."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'label': [1, 1, 1, 1],  # would give all 1s
            'dpds': [0, 0, 5, 10],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model, X_train=X, y_train=None,
            target='label',  # ignored because overdue/dpds provided
            overdue='dpds', dpds=3, feature_names=['f0']
        )
        # dpds > 3 → [0, 0, 1, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]

    def test_overdue_dpds_with_datasets_dict(self):
        """overdue/dpds works with datasets dict."""
        X_train = pd.DataFrame({
            'f0': [1, 2, 3, 4],
            'dpds': [0, 1, 5, 10],
        })
        X_test = pd.DataFrame({
            'f0': [5, 6],
            'dpds': [3, 15],
        })
        model = MockModel(feature_names=['f0'])
        report = ModelReport(
            model=model,
            datasets={'train': (X_train, None), 'test': (X_test, None)},
            overdue='dpds', dpds=3, feature_names=['f0']
        )
        # train: dpds > 3 → [0, 0, 1, 1]
        # test:  dpds > 3 → [0, 1]
        assert report._datasets['train'].y.tolist() == [0, 0, 1, 1]
        assert report._datasets['test'].y.tolist() == [0, 1]

    def test_overdue_dpds_auto_model_report(self):
        """auto_model_report with overdue/dpds separate parameters."""
        from hscredit.report.model_report import auto_model_report
        X = pd.DataFrame({
            'f0': list(range(100)),
            'dpds': list(range(100)),
        })
        model = MockModel(feature_names=['f0'])
        report = auto_model_report(
            model=model, X_train=X,
            overdue='dpds', dpds=[30, 15, 7],
            feature_names=['f0'],
            verbose=False, with_plots=False,
        )
        # dpds > 30 or > 15 or > 7 → dpds > 7 → rows 8-99 → 92 out of 100
        assert report._datasets['train'].y.sum() == 92
        assert report._datasets['train'].y.mean() == 0.92

    def test_overdue_dpds_equivalent_to_dict_target(self):
        """overdue/dpds as separate params should produce same y as dict target."""
        X = pd.DataFrame({
            'f0': [1, 2, 3, 4, 5, 6],
            'dpds': [0, 2, 5, 8, 12, 20],
        })
        model = MockModel(feature_names=['f0'])

        # via dict target
        r1 = ModelReport(
            model=model, X_train=X, y_train=None,
            target={'overdue': 'dpds', 'dpds': [10, 5, 0]},
            feature_names=['f0']
        )

        # via separate params
        r2 = ModelReport(
            model=model, X_train=X, y_train=None,
            overdue='dpds', dpds=[10, 5, 0],
            feature_names=['f0']
        )

        assert r1._datasets['train'].y.tolist() == r2._datasets['train'].y.tolist()


class TestModelReportRegression:
    """Regression tests for report sections that previously exported blank."""

    @staticmethod
    def _multi_label_data():
        return pd.DataFrame({
            'f0': np.arange(20),
            'MOB1': [0, 1, 3, 7, 8] * 4,
            '放款金额': np.arange(100, 120),
            '放款时间': pd.date_range('2024-01-01', periods=20, freq='D'),
        })

    def test_positive_probability_respects_model_classes(self):
        X = pd.DataFrame({'f0': [1, 2, 3, 4]})
        y = pd.Series([0, 0, 1, 1])
        model = ReversedClassModel(feature_names=['f0'])

        report = ModelReport(model, X_train=X, y_train=y, feature_names=['f0'])

        expected = model.predict_proba(X)[:, 0]
        np.testing.assert_allclose(report._datasets['train'].y_proba, expected)

    def test_multi_label_lift_contains_values_and_amount_metrics(self):
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue=['MOB1'],
            dpds=[7, 3, 0],
            feature_names=['f0'],
        )

        order_table = report._get_top_n_lift_table(labels=report._label_names)
        amount_table = report._get_top_n_lift_table(
            labels=report._label_names,
            amount_col='放款金额',
        )

        assert report.feature_names == ['f0']
        assert not order_table.isna().any().any()
        assert not amount_table.isna().any().any()
        assert not order_table.equals(amount_table)

    def test_excel_contains_all_sections_and_multi_label_description(self, tmp_path):
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue=['MOB1'],
            dpds=[7, 3, 0],
            feature_names=['f0'],
        )
        output = tmp_path / 'model_report.xlsx'

        report.to_excel(
            str(output),
            with_plots=False,
            amount_col='放款金额',
            project_desc='测试项目描述',
            data_source='测试数据源',
        )

        workbook = load_workbook(output)
        assert workbook.sheetnames == [
            '目录', '1-基本信息', '2-模型性能', '3-入模变量分析',
            '4-稳定性分析', '5-模型参数', '6-模型部署需求',
        ]
        contents = [cell.value for row in workbook['目录'].iter_rows() for cell in row]
        basic_info = [cell.value for row in workbook['1-基本信息'].iter_rows() for cell in row]
        performance = [cell.value for row in workbook['2-模型性能'].iter_rows() for cell in row]
        feature_sheet = workbook['3-入模变量分析']
        summary_feature_cell = next(
            cell
            for row in feature_sheet.iter_rows()
            for cell in row
            if cell.value == 'f0' and cell.hyperlink is not None
        )
        feature_title_cell = next(
            cell
            for row in feature_sheet.iter_rows()
            for cell in row
            if cell.value == '3.1、f0 有效性分析'
        )

        assert '5-模型参数' in contents
        assert '6-模型部署需求' in contents
        assert '测试项目描述' in basic_info
        assert '测试数据源' in basic_info
        assert any(isinstance(value, str) and 'MOB1@7:' in value for value in basic_info)
        assert '各数据集标签坏样本率' not in basic_info
        assert any(isinstance(value, float) and not np.isnan(value) for value in performance)

        basic_info_sheet = workbook['1-基本信息']
        sample_total_header = next(
            cell
            for row in basic_info_sheet.iter_rows()
            for cell in row
            if cell.value == '样本总数'
        )
        assert basic_info_sheet.cell(sample_total_header.row - 1, sample_total_header.column).value == '统计详情'
        assert basic_info_sheet.cell(sample_total_header.row - 1, sample_total_header.column - 1).value is None

        feature_values = [cell.value for row in feature_sheet.iter_rows() for cell in row]
        assert feature_values.count('训练集 订单口径') == 1
        assert feature_values.count('测试集 订单口径') == 1
        assert summary_feature_cell.hyperlink.location == f"#'3-入模变量分析'!{feature_title_cell.coordinate}"
        assert feature_title_cell.hyperlink.location == f"#'3-入模变量分析'!{summary_feature_cell.coordinate}"

    def test_excel_contents_sheet_adjusts_column_width(self, tmp_path):
        X = pd.DataFrame({
            'f0': np.arange(20),
            'target': [0, 1] * 10,
        })
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X},
            target='target',
            feature_names=['f0'],
        )
        output = tmp_path / 'model_report_contents_width.xlsx'

        report.to_excel(str(output), with_plots=False, model_name='VeryLongModelNameForColumnWidthCheck')

        contents_sheet = load_workbook(output)['目录']
        assert contents_sheet.column_dimensions['B'].width > 8
        assert contents_sheet.column_dimensions['C'].width > 20
        assert contents_sheet.column_dimensions['D'].width > 30

    def test_excel_title_merges_follow_actual_content_width(self, tmp_path):
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue=['MOB1'],
            dpds=[7, 3, 0],
            feature_names=['f0'],
        )
        output = tmp_path / 'model_report_dynamic_title_merges.xlsx'

        report.to_excel(
            str(output),
            with_plots=False,
            amount_col='放款金额',
            date_col='放款时间',
        )

        workbook = load_workbook(output)
        contents = workbook['目录']
        basic = workbook['1-基本信息']
        performance = workbook['2-模型性能']
        feature_sheet = workbook['3-入模变量分析']

        assert _merged_range_for_row(contents, 2).max_col == 4
        assert _merged_range_for_row(basic, 2).max_col == basic.max_column
        assert _merged_range_for_row(performance, 2).max_col == performance.max_column
        assert _merged_range_for_row(feature_sheet, 2).max_col == feature_sheet.max_column
        assert _merged_range_for_row(basic, 2).max_col != 35

        desc_row = _row_for_value(basic, '2、数据样本描述')
        assert _merged_range_for_row(basic, desc_row).max_col == 3

        parent_row = _row_for_value(feature_sheet, '3、入模变量有效性分析')
        child_row = _row_for_value(feature_sheet, '3.1、f0 有效性分析')
        assert _merged_range_for_row(feature_sheet, parent_row).max_col == feature_sheet.max_column
        assert _merged_range_for_row(feature_sheet, child_row).max_col == feature_sheet.max_column

    def test_excel_skips_hyperlink_when_feature_missing_from_summary(self, tmp_path, monkeypatch):
        """特征不在重要性汇总表中时（summary_row为None），应跳过超链接而不是抛异常."""
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue=['MOB1'],
            dpds=[7, 3, 0],
            feature_names=['f0'],
        )

        original_summary = report._get_features_summary
        monkeypatch.setattr(
            report, '_get_features_summary',
            lambda: original_summary().iloc[0:0],
        )

        output = tmp_path / 'model_report_missing_feature.xlsx'
        report.to_excel(str(output), with_plots=False, amount_col='放款金额')

        feature_sheet = load_workbook(output)['3-入模变量分析']
        feature_title_cell = next(
            cell
            for row in feature_sheet.iter_rows()
            for cell in row
            if cell.value == '3.1、f0 有效性分析'
        )
        assert feature_title_cell.hyperlink is None

    def test_time_distribution_header_groups_detail_columns(self, tmp_path):
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue=['MOB1'],
            dpds=[7, 3, 0],
            feature_names=['f0'],
        )

        output = tmp_path / 'model_report_time_distribution.xlsx'
        report.to_excel(str(output), with_plots=False, date_col='放款时间')

        basic = load_workbook(output)['1-基本信息']
        data_group_header = next(cell for row in basic.iter_rows() for cell in row if cell.value == '数据分组')
        header_row = data_group_header.row
        data_set_header = next(
            basic.cell(header_row, col)
            for col in range(1, basic.max_column + 1)
            if basic.cell(header_row, col).value == '数据集'
        )
        sample_total_header = next(
            basic.cell(header_row, col)
            for col in range(1, basic.max_column + 1)
            if basic.cell(header_row, col).value == '样本总数'
        )

        assert basic.cell(header_row - 1, data_set_header.column).value == '统计详情'
        assert any(
            cell_range.min_row == header_row - 1
            and cell_range.min_col == data_set_header.column
            and cell_range.max_row == header_row - 1
            and cell_range.max_col == sample_total_header.column
            for cell_range in basic.merged_cells.ranges
        )

    def test_single_label_performance_metric_formats(self, tmp_path):
        X = pd.DataFrame({'f0': [1, 2, 3, 4, 5, 6], 'target': [0, 0, 0, 1, 1, 1]})
        report = ModelReport(
            MockModel(['f0']),
            X_train=X,
            y_train=None,
            target='target',
            feature_names=['f0'],
        )

        output = tmp_path / 'single_label_report.xlsx'
        report.to_excel(str(output), with_plots=False)

        performance = load_workbook(output)['2-模型性能']
        ks_cell = next(cell for row in performance.iter_rows() for cell in row if cell.value == 'KS')
        auc_cell = next(cell for row in performance.iter_rows() for cell in row if cell.value == 'AUC')
        sample_cell = next(cell for row in performance.iter_rows() for cell in row if cell.value == '样本总数')
        bad_rate_cell = next(cell for row in performance.iter_rows() for cell in row if cell.value == '坏样本率')

        assert performance.cell(ks_cell.row, ks_cell.column + 1).number_format == '0.00%'
        assert performance.cell(auc_cell.row, auc_cell.column + 1).number_format == '0.00%'
        assert performance.cell(bad_rate_cell.row, bad_rate_cell.column + 1).number_format == '0.00%'
        assert performance.cell(sample_cell.row, sample_cell.column + 1).number_format == '#,##0'
        assert isinstance(performance.cell(sample_cell.row, sample_cell.column + 1).value, int)

    def test_compare_models_summary_excel_keeps_ratio_values(self, tmp_path):
        from hscredit.report.model_report import compare_models

        X = pd.DataFrame({'f0': [1, 2, 3, 4, 5, 6]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        output = tmp_path / 'compare_models.xlsx'

        result = compare_models(
            {'LR': MockModel(['f0'])},
            X,
            y,
            X_test=X,
            y_test=y,
            excel_path=str(output),
        )

        assert result.loc[('LR', 'target'), ('坏样本率', '训练集')] == 0.5

        sheet = load_workbook(output)['Sheet1']
        bad_rate_header = next(
            cell
            for row in sheet.iter_rows()
            for cell in row
            if cell.value == '坏样本率'
        )
        train_header = next(
            sheet.cell(bad_rate_header.row + 1, col)
            for col in range(bad_rate_header.column, sheet.max_column + 1)
            if sheet.cell(bad_rate_header.row + 1, col).value == '训练集'
        )
        data_cell = sheet.cell(train_header.row + 1, train_header.column)
        assert data_cell.value == 0.5
        assert data_cell.number_format == '0.00%'

    def test_export_plots_contains_feature_psi(self, tmp_path):
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue='MOB1',
            dpds=3,
            feature_names=['f0'],
        )

        paths, tables = report._export_plots(tmp_path)

        assert 'feat_psi_f0' in paths
        assert Path(paths['feat_psi_f0'][0]).exists()
        assert 'feat_psi_f0' in tables
        assert not tables['feat_psi_f0'].empty
        assert tables['feat_psi_f0']['预期坏样本率'].gt(0).any()
        assert tables['feat_psi_f0']['实际坏样本率'].gt(0).any()

    def test_multi_label_tables_use_expected_layout(self, tmp_path):
        X = self._multi_label_data()
        report = ModelReport(
            MockModel(['f0']),
            datasets={'train': X, 'test': X.copy()},
            overdue='MOB1',
            dpds=[7, 3, 0],
            feature_names=['f0'],
        )

        bin_table = report.get_bin_table(labels=report._label_names)
        feature_table = report.get_feature_bin_table('f0', labels=report._label_names)
        for table in (bin_table, feature_table):
            assert isinstance(table.columns, pd.MultiIndex)
            assert {'MOB1>7', 'MOB1>3', 'MOB1>0'} <= set(table.columns.get_level_values(0))
            assert '指标名称' not in table.columns.get_level_values(-1)
            assert '指标含义' not in table.columns.get_level_values(-1)

        output = tmp_path / 'multi_layout.xlsx'
        report.to_excel(str(output), with_plots=False, amount_col='放款金额')
        workbook = load_workbook(output)
        performance = workbook['2-模型性能']
        basic = workbook['1-基本信息']

        assert performance['B7'].value == '统计项'
        assert [performance.cell(7, col).value for col in (3, 5, 7)] == [
            'MOB1>7', 'MOB1>3', 'MOB1>0'
        ]
        assert [performance.cell(8, col).value for col in range(2, 8)] == [
            '统计指标', '训练集', '测试集', '训练集', '测试集', '训练集'
        ]
        assert performance['B11'].value == '样本总数'
        assert performance['C9'].number_format == '0.00%'
        assert performance['C10'].number_format == '0.00%'
        assert performance['C11'].number_format == '#,##0'
        assert isinstance(performance['C11'].value, int)
        assert performance['B19'].value == '统计指标'
        assert performance.auto_filter.ref == 'B20:AJ26'
        sample_total_header = next(cell for row in basic.iter_rows() for cell in row if cell.value == '样本总数')
        assert basic.cell(sample_total_header.row - 1, sample_total_header.column).value == '统计详情'
        assert basic.cell(sample_total_header.row - 1, sample_total_header.column - 1).value is None
        assert basic.cell(sample_total_header.row, sample_total_header.column - 1).value == '数据集'


class TestModelReportRealDataContract:
    """真实放款数据的完整 Excel 报告契约。"""

    FEATURE_NAMES = ['衡枢鉴真分老客版', '近六个月非银多头机构数', '青云24']
    EXPECTED_SHEETS = [
        '目录', '1-基本信息', '2-模型性能', '3-入模变量分析',
        '4-稳定性分析', '5-模型参数', '6-模型部署需求',
    ]

    @staticmethod
    def _sheet_values(ws):
        return [cell.value for row in ws.iter_rows() for cell in row]

    @staticmethod
    def _numeric_values_after(ws, label):
        cell = next(cell for row in ws.iter_rows() for cell in row if cell.value == label)
        return [
            ws.cell(cell.row, col).value
            for col in range(cell.column + 1, ws.max_column + 1)
            if isinstance(ws.cell(cell.row, col).value, (int, float))
        ]

    def test_auto_model_report_real_data_excel_contract(self, tmp_path):
        from hscredit.report.model_report import auto_model_report

        source = pd.read_excel(Path(__file__).parents[2] / 'examples' / 'hscredit_yyp.xlsx')
        frames = {
            'train': source.iloc[:500].copy(),
            'test': source.iloc[500:750].copy(),
            'oot': source.iloc[750:].copy(),
        }
        expected_labels = ['MOB1>7', 'MOB1>3', 'MOB1>0']
        expected_display_labels = ['MOB1@7', 'MOB1@3', 'MOB1@0']
        expected_counts = [len(frame) for frame in frames.values()]
        expected_bad_rates = {
            label: [float((frame['MOB1'] > threshold).mean()) for frame in frames.values()]
            for label, threshold in zip(expected_labels, [7, 3, 0])
        }
        output = tmp_path / '真实数据模型评估报告.xlsx'

        report = auto_model_report(
            model=RealDataContractModel(self.FEATURE_NAMES),
            datasets=frames,
            feature_names=self.FEATURE_NAMES,
            overdue=['MOB1'],
            dpds=[7, 3, 0],
            excel_path=str(output),
            amount_col='放款金额',
            date_col='放款时间',
            group_col='商品类别',
            with_plots=True,
            verbose=False,
            model_name='真实数据契约模型',
            project_desc='真实放款数据多标签模型验证',
            data_source='examples/hscredit_yyp.xlsx',
            loc_cols='客户编号',
        )

        assert output.exists() and output.stat().st_size > 0
        assert report._label_names == expected_labels
        for dataset_key, frame in frames.items():
            dataset = report._datasets[dataset_key]
            assert len(dataset.y) == len(frame)
            assert np.ptp(dataset.y_proba) > 0
            for label, threshold in zip(expected_labels, [7, 3, 0]):
                expected_y = (frame['MOB1'] > threshold).astype(int).to_numpy()
                np.testing.assert_array_equal(dataset.y_dict[label], expected_y)

        summary = report.summary()
        assert summary.index.tolist() == expected_display_labels
        for label, display_label in zip(expected_labels, expected_display_labels):
            for dataset_label, expected_rate in zip(['训练集', '测试集', 'OOT集'], expected_bad_rates[label]):
                assert summary.loc[display_label, ('样本数', dataset_label)] == len(
                    frames[{'训练集': 'train', '测试集': 'test', 'OOT集': 'oot'}[dataset_label]]
                )
                assert summary.loc[display_label, ('坏样本率', dataset_label)] == expected_rate

        workbook = load_workbook(output, data_only=False)
        assert workbook.sheetnames == self.EXPECTED_SHEETS

        critical_headings = {
            '目录': ['模型评估报告'],
            '1-基本信息': ['一、基本信息', '1、项目目标', '2、数据样本描述', '3、数据样本统计', '4、样本分布情况'],
            '2-模型性能': ['二、模型性能评估', '1、模型性能验证指标'],
            '3-入模变量分析': ['三、入模变量分析', '1、入模变量重要性及分布情况', '2、入模变量相关性', '3、入模变量有效性分析'],
            '4-稳定性分析': ['四、模型稳定性分析', '1、评分分布统计'],
            '5-模型参数': ['五、模型选型及参数', '1、模型选型', '2、模型参数', '3、入模特征列表'],
            '6-模型部署需求': ['六、模型部署需求', '1、入模变量信息', '2、生产订单测试用例'],
        }
        for sheet_name, headings in critical_headings.items():
            values = self._sheet_values(workbook[sheet_name])
            for heading in headings:
                assert heading in values, f'{sheet_name} 缺少章节 {heading}'

        basic = workbook['1-基本信息']
        basic_values = self._sheet_values(basic)
        assert '真实放款数据多标签模型验证' in basic_values
        assert 'examples/hscredit_yyp.xlsx' in basic_values
        assert '月度分布' in basic_values
        assert '商品类别分布' in basic_values
        assert set(source['商品类别'].unique()) <= set(basic_values)
        assert all(any(isinstance(value, str) and f'{label}:' in value for value in basic_values) for label in expected_display_labels)

        performance = workbook['2-模型性能']
        assert self._numeric_values_after(performance, '样本总数')[:9] == expected_counts * 3
        workbook_bad_rates = self._numeric_values_after(performance, '坏样本率')[:9]
        np.testing.assert_allclose(
            workbook_bad_rates,
            [rate for label in expected_labels for rate in expected_bad_rates[label]],
            atol=5e-5,
        )
        bad_rate_cell = next(cell for row in performance.iter_rows() for cell in row if cell.value == '坏样本率')
        assert all(
            performance.cell(bad_rate_cell.row, col).number_format == '0.00%'
            for col in range(bad_rate_cell.column + 1, bad_rate_cell.column + 10)
        )

        for sheet_name in self.EXPECTED_SHEETS:
            ws = workbook[sheet_name]
            assert _merged_range_for_row(ws, 2) is not None
        for sheet_name in self.EXPECTED_SHEETS[1:]:
            assert workbook[sheet_name]['B2'].hyperlink.location == "#'目录'!B2"

        contents = workbook['目录']
        for sheet_name in self.EXPECTED_SHEETS[1:]:
            content_cell = next(cell for row in contents.iter_rows() for cell in row if cell.value == sheet_name)
            assert content_cell.hyperlink.location == f"#'{sheet_name}'!B2"

        feature_sheet = workbook['3-入模变量分析']
        for index, feature in enumerate(self.FEATURE_NAMES, start=1):
            summary_cell = next(
                cell for row in feature_sheet.iter_rows()
                for cell in row if cell.value == feature and cell.hyperlink is not None
            )
            title_cell = next(
                cell for row in feature_sheet.iter_rows()
                for cell in row if cell.value == f'3.{index}、{feature} 有效性分析'
            )
            assert summary_cell.hyperlink.location == f"#'3-入模变量分析'!{title_cell.coordinate}"
            assert title_cell.hyperlink.location == f"#'3-入模变量分析'!{summary_cell.coordinate}"

        for sheet_name in ['1-基本信息', '2-模型性能', '3-入模变量分析', '4-稳定性分析', '5-模型参数', '6-模型部署需求']:
            assert workbook[sheet_name].freeze_panes is not None
        for sheet_name in ['2-模型性能', '6-模型部署需求']:
            ref = workbook[sheet_name].auto_filter.ref
            assert ref and ':' in ref
        assert workbook['2-模型性能']._images
        assert feature_sheet._images

        deployment = workbook['6-模型部署需求']
        deployment_values = self._sheet_values(deployment)
        assert '客户编号' in deployment_values
        assert all(feature in deployment_values for feature in self.FEATURE_NAMES)
        assert '模型分数' in deployment_values
        model_score = next(cell for row in deployment.iter_rows() for cell in row if cell.value == '模型分数')
        scores = [
            deployment.cell(row, model_score.column).value
            for row in range(model_score.row + 1, model_score.row + 6)
        ]
        assert all(isinstance(value, (int, float)) for value in scores)
        assert len(set(scores)) > 1

        formula_errors = {'#NULL!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#NUM!', '#N/A'}
        assert not [
            (ws.title, cell.coordinate, cell.value)
            for ws in workbook.worksheets
            for row in ws.iter_rows()
            for cell in row
            if isinstance(cell.value, str) and cell.value in formula_errors
        ]

    def test_optional_plot_insert_failure_is_logged(self, tmp_path, monkeypatch, caplog):
        X = pd.DataFrame({'f0': range(20), 'target': [0, 1] * 10})
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])
        missing_plot = tmp_path / 'missing-model-plot.png'
        monkeypatch.setattr(report, '_export_plots', lambda *args, **kwargs: ({'model_train': [str(missing_plot)]}, {}))

        report.to_excel(str(tmp_path / 'plot-warning.xlsx'), with_plots=True)

        assert str(missing_plot) in caplog.text
        assert '2-模型性能' in caplog.text

    def test_optional_plot_generation_failure_is_logged_with_context(self, tmp_path, monkeypatch, caplog):
        X = pd.DataFrame({'f0': range(20), 'target': [0, 1] * 10})
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])

        def fail_bin_plot(*args, **kwargs):
            raise RuntimeError('injected model bin plot failure')

        monkeypatch.setattr('hscredit.core.viz.bin_plot', fail_bin_plot)
        report._export_plots(tmp_path)

        assert '模型评分分箱图' in caplog.text
        assert '训练集' in caplog.text
        assert str(tmp_path / 'bin_train.png') in caplog.text
        assert 'injected model bin plot failure' in caplog.text

    def test_required_feature_effectiveness_table_failure_surfaces(self, tmp_path, monkeypatch):
        X = pd.DataFrame({'f0': range(20), 'target': [0, 1] * 10})
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])

        def fail_feature_table(*args, **kwargs):
            raise ValueError('injected feature table failure')

        monkeypatch.setattr(report, 'get_feature_bin_table', fail_feature_table)

        with pytest.raises(RuntimeError, match=r'特征=f0.*数据集=训练集') as exc_info:
            report.to_excel(str(tmp_path / 'required-section.xlsx'), with_plots=False)

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert 'injected feature table failure' in str(exc_info.value.__cause__)

    def test_visibility_flags_hide_lift_and_importance_sections(self, tmp_path):
        X = pd.DataFrame({'f0': range(20), 'target': [0, 1] * 10})
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])
        output = tmp_path / 'visibility-flags.xlsx'

        report.to_excel(
            str(output),
            with_plots=False,
            show_lift=False,
            show_importance=False,
        )

        workbook = load_workbook(output)
        feature_values = self._sheet_values(workbook['3-入模变量分析'])
        assert not any(isinstance(value, str) and '入模变量重要性及分布情况' in value for value in feature_values)
        assert any(isinstance(value, str) and '入模变量相关性' in value for value in feature_values)
        assert any(isinstance(value, str) and '入模变量有效性分析' in value for value in feature_values)

    def test_export_plots_respects_show_lift(self, tmp_path, monkeypatch):
        X = pd.DataFrame({'f0': range(20), 'target': [0, 1] * 10})
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])

        def fail_if_called(*args, **kwargs):
            raise AssertionError('show_lift=False 时不应生成 LIFT 曲线')

        monkeypatch.setattr('hscredit.core.viz.lift_plot', fail_if_called)
        paths, _ = report._export_plots(tmp_path, show_lift=False)

        assert all('lift_' not in path for figures in paths.values() for path in figures)

    def test_auto_model_report_forwards_visibility_flags(self, tmp_path, monkeypatch):
        from hscredit.report.model_report import auto_model_report

        X = pd.DataFrame({'f0': range(20)})
        y = pd.Series([0, 1] * 10)
        captured = {}

        def capture_to_excel(self, filepath, **kwargs):
            captured.update(kwargs)
            return filepath

        monkeypatch.setattr(ModelReport, 'to_excel', capture_to_excel)
        auto_model_report(
            MockModel(['f0']),
            X_train=X,
            y_train=y,
            feature_names=['f0'],
            excel_path=str(tmp_path / 'forwarding.xlsx'),
            show_lift=False,
            show_importance=False,
            verbose=False,
        )

        assert captured['show_lift'] is False
        assert captured['show_importance'] is False

    def test_required_directory_hyperlink_failure_surfaces(self, tmp_path, monkeypatch):
        from hscredit.excel import ExcelWriter

        X = pd.DataFrame({'f0': range(20), 'target': [0, 1] * 10})
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])

        def fail_hyperlink(*args, **kwargs):
            raise ValueError('injected hyperlink failure')

        monkeypatch.setattr(ExcelWriter, 'insert_hyperlink2sheet', fail_hyperlink)
        with pytest.raises(RuntimeError, match='目录链接') as exc_info:
            report.to_excel(str(tmp_path / 'required-link.xlsx'), with_plots=False)

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_required_amount_score_table_failure_surfaces(self, tmp_path, monkeypatch):
        X = pd.DataFrame({
            'f0': range(20),
            'target': [0, 1] * 10,
            '放款金额': np.linspace(1000, 2000, 20),
        })
        report = ModelReport(MockModel(['f0']), datasets={'train': X}, target='target', feature_names=['f0'])
        original_get_bin_table = report.get_bin_table

        def fail_amount_table(*args, **kwargs):
            if kwargs.get('amount_col') is not None:
                raise ValueError('injected amount table failure')
            return original_get_bin_table(*args, **kwargs)

        monkeypatch.setattr(report, 'get_bin_table', fail_amount_table)
        with pytest.raises(RuntimeError, match=r'金额口径评分分箱.*数据集=训练集') as exc_info:
            report.to_excel(
                str(tmp_path / 'required-amount-table.xlsx'),
                with_plots=False,
                amount_col='放款金额',
            )

        assert isinstance(exc_info.value.__cause__, ValueError)
