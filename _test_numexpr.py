#!/usr/bin/env python
"""Minimal test to find where it hangs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: imports", flush=True)
import numpy as np
print("np imported", flush=True)
import pandas as pd
print("pd imported", flush=True)

print("Step 2: DataFrame creation", flush=True)
df = pd.DataFrame({'a': [1,2,3]})
print("df created", flush=True)

print("Step 3: Read expression.py", flush=True)
with open("hscredit/core/feature_engineering/expression.py", "r") as f:
    src = f.read()
print(f"read {len(src)} bytes", flush=True)

print("Step 4: Patch and exec", flush=True)
src = src.replace("from pandas import DataFrame", "DataFrame = pd.DataFrame")
src = src.replace(
    "    def __sklearn_tags__(self):\n        from sklearn.utils._tags import Tags, TargetTags, TransformerTags\n\n        return Tags(\n            estimator_type=None,\n            target_tags=TargetTags(required=False),\n            transformer_tags=TransformerTags(),\n        )\n",
    ""
)
print("patched", flush=True)

ns = {"__name__": "__main__", "np": np, "pd": pd, "DataFrame": pd.DataFrame}
exec(src, ns)
print("exec done", flush=True)

NumExprDerive = ns["NumExprDerive"]
print("Got NumExprDerive class", flush=True)

# Simple test
X = pd.DataFrame({'score': [650, 580, 720, 490]})
fd = NumExprDerive(derivings=[('level', 'where(score >= 600, 1, 0)')])
fd.fit(X)
print("fit done", flush=True)
result = fd.transform(X)
print("transform done", flush=True)
print(result.to_string())
print("SUCCESS")