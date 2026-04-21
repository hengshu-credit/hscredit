"""Tests for NumExprDerive feature engineering."""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.feature_engineering import NumExprDerive


class TestNumExprDeriveNumeric:
    """纯数值型 DataFrame 测试。"""

    def test_basic_arithmetic(self):
        X = pd.DataFrame({'f0': [2, 1.0, 3], 'f1': [4, 2, 3]})
        fd = NumExprDerive([('f2', 'f0+f1')])
        r = fd.fit_transform(X)
        assert list(r.columns) == ['f0', 'f1', 'f2']
        assert r['f2'].tolist() == [6, 3, 6]

    def test_where_numeric(self):
        X = pd.DataFrame({'f0': [2, 1.0, 3], 'f1': [4, 2, 3]})
        fd = NumExprDerive([('f2', 'where(f0>1, f0, f1)')])
        r = fd.fit_transform(X)
        assert r['f2'].tolist() == [2, 1.0, 3]

    def test_sin_abs(self):
        X = pd.DataFrame({'f0': [2.0, 0.0, 0.5]})
        fd = NumExprDerive([('s', 'sin(f0)'), ('a', 'abs(-f0)')])
        r = fd.fit_transform(X)
        assert list(r.columns) == ['f0', 's', 'a']
        assert r['a'].tolist() == [2.0, 0.0, 0.5]

    def test_inf(self):
        X = pd.DataFrame({'f0': [2.0, np.inf, 3]})
        fd = NumExprDerive([('f1', 'sin(f0)')])
        r = fd.fit_transform(X)
        assert pd.isna(r['f1'].iloc[1])  # sin(inf) -> nan


class TestNumExprDeriveMixed:
    """混合类型 DataFrame 测试。"""

    def test_string_where(self):
        X = pd.DataFrame({'score': [650, 580], 'status': ['正常', '逾期']})
        fd = NumExprDerive([('band', "where(score >= 600, '高', '低')")])
        r = fd.fit_transform(X)
        assert list(r.columns) == ['score', 'status', 'band']
        assert r['band'].tolist() == ['高', '低']

    def test_string_comparison(self):
        X = pd.DataFrame({'status': ['正常', '逾期', '关注']})
        fd = NumExprDerive([('is_overdue', "where(status == '逾期', 1, 0)")])
        r = fd.fit_transform(X)
        assert r['is_overdue'].tolist() == [0, 1, 0]

    def test_bool_numeric_where(self):
        X = pd.DataFrame({'is_vip': [True, False, True], 'score': [100, 200, 300]})
        fd = NumExprDerive([('adjusted', 'where(is_vip, score * 1.1, score)')])
        r = fd.fit_transform(X)
        assert r['adjusted'].tolist() == [110.0, 200.0, 330.0]

    def test_mixed_logic_or(self):
        X = pd.DataFrame({
            'status': ['正常', '逾期', '正常', '关注'],
            'is_vip': [True, False, True, False],
        })
        fd = NumExprDerive([('flag', "where((status == '逾期') | is_vip, 1, 0)")])
        r = fd.fit_transform(X)
        assert r['flag'].tolist() == [1, 1, 1, 0]

    def test_string_ne(self):
        X = pd.DataFrame({'name': ['Alice', 'Bob', 'Charlie']})
        fd = NumExprDerive([('not_bob', "where(name != 'Bob', 1, 0)")])
        r = fd.fit_transform(X)
        assert r['not_bob'].tolist() == [1, 0, 1]


class TestNumExprDeriveNdarray:
    """ndarray 输入测试。"""

    def test_ndarray_basic(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        fd = NumExprDerive([('f2', 'f0+f1'), ('f3', 'where(f0>2, f0, f1)')])
        r = fd.fit_transform(X)
        assert r.shape == (3, 4)
        assert r[:, 2].tolist() == [3, 7, 11]  # f0+f1
        assert r[:, 3].tolist() == [3, 4, 5]    # where(f0>2, f0, f1)


class TestNumExprDeriveEdge:
    """边界情况测试。"""

    def test_empty_derivings(self):
        with pytest.raises(ValueError):
            NumExprDerive([])

    def test_invalid_deriving_type(self):
        with pytest.raises(ValueError):
            NumExprDerive([("f", 123)])

    def test_fit_returns_self(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        fd = NumExprDerive([('c', 'a+b')])
        result = fd.fit(X)
        assert result is fd
