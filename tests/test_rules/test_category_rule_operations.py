"""类别集合规则经逻辑组合后保持真实命中语义。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.rules import Rule


@pytest.mark.parametrize("expression", ["学位 in ['硕士', '博士']", "学位 in ('硕士', '博士')", "学位.isin(['硕士', '博士'])"])
def test_category_membership_survives_rule_operations(expression):
    data = pd.DataFrame({"学位": ["学士", "硕士", "博士", "--", None], "账户": [3, 12, 4, 15, 2]})
    degree = Rule(expression)
    accounts = Rule("账户 >= 10")
    expected = data["学位"].isin(["硕士", "博士"])
    np.testing.assert_array_equal((~degree).predict(data), ~expected)
    np.testing.assert_array_equal((degree & accounts).predict(data), expected & data["账户"].ge(10))
    np.testing.assert_array_equal((degree | accounts).predict(data), expected | data["账户"].ge(10))
    np.testing.assert_array_equal((degree ^ accounts).predict(data), expected ^ data["账户"].ge(10))
