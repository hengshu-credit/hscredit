"""衍生表达式的中文引用和信任边界。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.feature_engineering import NumExprDerive


def test_chinese_and_non_identifier_columns():
    X = pd.DataFrame({"收入": [1, 2], "年 龄": [3, 4]})
    out = NumExprDerive([("合计", "收入 + col('年 龄')"), ("加倍", "合计 * 2")]).fit_transform(X)
    assert out["合计"].tolist() == [4, 6]
    assert out["加倍"].tolist() == [8, 12]


@pytest.mark.parametrize("expression", [
    "__import__('math').sqrt(4)", "np.__dict__", "收入.__class__", "np.load('file')",
    "(lambda: 1)()", "[x for x in 收入]", "getattr(np, 'load')('x')", "收入[0]",
    "np.maximum(收入, 0, 收入)", "np.clip(收入, 0, 1, out=收入)",
])
def test_untrusted_expression_cannot_execute_python(expression):
    with pytest.raises(ValueError):
        NumExprDerive([("z", expression)]).fit_transform(pd.DataFrame({"收入": [1, 2]}))


def test_trusted_python_is_explicit_opt_in():
    result = NumExprDerive([("z", "__import__('math').sqrt(4)")], trusted_python=True).fit_transform(pd.DataFrame({"x": [1, 2]}))
    assert result.z.tolist() == [2, 2]


def test_truthy_string_cannot_enable_trusted_mode():
    with pytest.raises(ValueError, match="布尔"):
        NumExprDerive([("z", "1")], trusted_python="False")
    transformer = NumExprDerive([("z", "1")]).set_params(trusted_python="false")
    with pytest.raises(ValueError, match="布尔"):
        transformer.transform(pd.DataFrame({"x": [1]}))


def test_allowed_numpy_and_boolean_comparisons():
    X = pd.DataFrame({"x": [1, 4, 9]})
    result = NumExprDerive([("z", "np.sqrt(x)"), ("flag", "where(1 < x < 8, 1, 0)")]).fit_transform(X)
    np.testing.assert_allclose(result.z, [1, 2, 3])
    assert result.flag.tolist() == [0, 1, 0]
