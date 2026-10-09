"""金融计算浮点广播与现金流符号反转回归。"""

import numpy as np
import pytest
from hscredit.core.financial import fv, pv, pmt, nper, ipmt, ppmt, rate, irr


def test_zero_integer_rate_keeps_fractional_payment():
    np.testing.assert_allclose(pmt(np.array([0, 0]), 3, 10), [-10 / 3, -10 / 3])
    np.testing.assert_allclose(nper(np.array([0, 0]), -3, 10), [10 / 3, 10 / 3])


@pytest.mark.parametrize("function,args", [
    (fv, ([0, .01], 12, -100, 1000)), (pv, ([0, .01], 12, -100)),
    (pmt, ([0, .01], 12, 1000)), (nper, ([0, .01], -100, 1000)),
    (ipmt, ([0, .01], 2, 12, 1000)), (ppmt, ([0, .01], 2, 12, 1000)),
])
def test_scalar_array_broadcast_matches_scalar_calls(function, args):
    result = function(*args)
    expected = [function(r, *args[1:]) for r in args[0]]
    np.testing.assert_allclose(result, expected)


def test_when_strings_array_and_two_dimensional_broadcast():
    result = pmt(np.array([[0], [.01]]), np.array([6, 12]), 1000, when=np.array(["begin", "end"]))
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result[0], [-1000 / 6, -1000 / 12])


def test_interest_and_principal_reconcile():
    np.testing.assert_allclose(ipmt([0, .01], [1, 2], 12, 1000) + ppmt([0, .01], [1, 2], 12, 1000), pmt([0, .01], 12, 1000))


def test_rate_broadcast_matches_scalar():
    result = rate(np.array([12, 24]), -100, 1000)
    np.testing.assert_allclose(result, [rate(12, -100, 1000), rate(24, -100, 1000)])


@pytest.mark.parametrize("values", [[-100, 110], [-1000, 300, 400, 400, 300], [-100, 50]])
def test_irr_invariant_under_cash_flow_sign_reversal(values):
    assert irr(values) == pytest.approx(irr(-np.asarray(values)), abs=1e-7)
    assert abs(np.sum(np.asarray(values) / (1 + irr(values)) ** np.arange(len(values)))) < 1e-4
