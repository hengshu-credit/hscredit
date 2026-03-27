"""Phase 1 smoke test - run as: python test_phase1.py"""
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple

# ---- minimal stubs so finance.py can be exec'd standalone ----
class _FakeBase:
    @staticmethod
    def _validate_same_length(a, b, names):
        if len(a) != len(b):
            raise ValueError(f"Length mismatch: {names}")
    @staticmethod
    def _validate_binary_target(y):
        pass

_validate_same_length = _FakeBase._validate_same_length
_validate_binary_target = _FakeBase._validate_binary_target

def compute_bin_stats(*a, **kw):
    return pd.DataFrame()

# ---- exec finance.py ----
with open('hscredit/core/metrics/finance.py') as f:
    src = f.read()
exec(compile(src, 'finance.py', 'exec'), globals())

# ---- tests ----
np.random.seed(42)
y_true = np.random.binomial(1, 0.1, 2000)
y_prob = np.clip(y_true * 0.6 + np.random.beta(1, 5, 2000), 0, 1)

print("=== lift_at scalar ===")
val = lift_at(y_true, y_prob, ratios=0.05)
print(f"LIFT@5% = {val}")
assert isinstance(val, float), f"Expected float, got {type(val)}"
assert val > 1.0, f"Expected LIFT > 1 for correlated scores, got {val}"

print("=== lift_at list ===")
df = lift_at(y_true, y_prob, ratios=[0.01, 0.03, 0.05, 0.10])
print(df.to_string(index=False))
assert len(df) == 4
assert 'LIFT值' in df.columns

print("=== lift_at tail (ascending=True) ===")
df_tail = lift_at(y_true, y_prob, ratios=[0.10, 0.20], ascending=True)
print(df_tail.to_string(index=False))

print("=== lift_curve (new defaults) ===")
df2 = lift_curve(y_true, y_prob)
print(df2.to_string(index=False))
assert '1%' in df2['覆盖率'].values or df2['覆盖率'].iloc[0] == '1%'

print("=== lift_curve tail ===")
df3 = lift_curve(y_true, y_prob, tail=True)
print(df3.head(3).to_string(index=False))

print("=== lift_monotonicity_check ===")
result = lift_monotonicity_check(y_true, y_prob, n_bins=10, direction='both')
print(f"head_monotonic      : {result['head_monotonic']}")
print(f"head_violation_ratio: {result['head_violation_ratio']}")
print(f"tail_monotonic      : {result['tail_monotonic']}")
print(f"tail_violation_ratio: {result['tail_violation_ratio']}")
print("head_bin_table:")
print(result['head_bin_table'].to_string(index=False))
assert 'head_monotonic' in result
assert 'tail_bin_table' in result
assert isinstance(result['head_bin_table'], pd.DataFrame)
assert len(result['head_bin_table']) == 10

print("=== lift_monotonicity_check direction='head' ===")
r2 = lift_monotonicity_check(y_true, y_prob, n_bins=10, direction='head')
assert r2['tail_monotonic'] is None
print(f"head_monotonic: {r2['head_monotonic']}, tail: {r2['tail_monotonic']}")

print("\n✅ All Phase 1 tests passed.")
