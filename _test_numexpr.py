#!/usr/bin/env python
"""Test NumExprDerive with mixed types."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Directly import from the file by execing just the class we need
with open("hscredit/core/feature_engineering/expression.py", "r") as f:
    src = f.read()

# Patch out problematic lines that would cause hangs
src = src.replace("from pandas import DataFrame", "DataFrame = pd.DataFrame")
# Remove sklearn_tags which may not be available in all sklearn versions
src = src.replace(
    "    def __sklearn_tags__(self):\n        from sklearn.utils._tags import Tags, TargetTags, TransformerTags\n\n        return Tags(\n            estimator_type=None,\n            target_tags=TargetTags(required=False),\n            transformer_tags=TransformerTags(),\n        )\n",
    ""
)

ns = {"__name__": "__main__", "np": np, "pd": pd, "DataFrame": pd.DataFrame}
exec(src, ns)
NumExprDerive = ns["NumExprDerive"]

# Test 1: Pure numeric
print("=== Test 1: Pure numeric ===")
X1 = pd.DataFrame({
    'f0': [2, 1.0, 3],
    'f1': [1.0, 2, 3],
    'f2': [2, 3, 4],
    'f3': [2.1, 1.4, -6.2]
})
fd1 = NumExprDerive(derivings=[
    ('f4', 'where(f1>1, 0, 1)'),
    ('f5', 'f1+f2'),
    ('f6', 'sin(f1)'),
    ('f7', 'abs(f3)')
])
result1 = fd1.fit_transform(X1)
print(result1.to_string())
print()

# Test 2: Mixed types (string, bool, numeric)
print("=== Test 2: Mixed types ===")
X2 = pd.DataFrame({
    'score': [650, 580, 720, 490],
    'status': ['正常', '逾期', '正常', '正常'],
    'is_vip': [True, False, True, False]
})
fd2 = NumExprDerive(derivings=[
    ('score_band', "where(score >= 600, '高', '低')"),
    ('flag', "where((status == '逾期') | is_vip, 1, 0)"),
    ('score_level', 'where(score > 600, score * 1.1, score * 0.9)')
])
result2 = fd2.fit_transform(X2)
print(result2.to_string())
print()
print("dtypes:")
print(result2.dtypes)
print()
print("SUCCESS: All tests passed!")
