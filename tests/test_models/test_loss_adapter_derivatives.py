"""自定义损失适配器对 raw margin 的导数契约测试。"""

import numpy as np
import pytest

from hscredit.core.models.base import resolve_custom_objective
from hscredit.core.models.losses import WeightedBCELoss
from hscredit.core.models.losses.adapters import (
    CatBoostLossAdapter,
    LightGBMLossAdapter,
    TabNetLossAdapter,
    XGBoostLossAdapter,
    _tabnet_binary_loss_and_gradient,
)


class _DMatrixLike:
    def __init__(self, labels):
        self._labels = np.asarray(labels)

    def get_label(self):
        return self._labels


def _finite_differences(loss, y_true, margin, eps=1e-5):
    sigmoid = lambda value: 1.0 / (1.0 + np.exp(-value))
    center = loss(y_true, sigmoid(margin))
    plus = loss(y_true, sigmoid(margin + eps))
    minus = loss(y_true, sigmoid(margin - eps))
    grad = (plus - minus) / (2 * eps)
    hess = (plus - 2 * center + minus) / (eps**2)
    return grad, hess


@pytest.mark.parametrize("margin", [-1.2, 0.0, 0.8])
def test_xgboost_adapter_derivatives_match_margin_finite_difference(margin):
    loss = WeightedBCELoss(pos_weight=1.7, neg_weight=0.6)
    y_true = np.array([1.0])
    expected_grad, expected_hess = _finite_differences(loss, y_true, np.array([margin]))

    grad, hess = XGBoostLossAdapter(loss).objective()(np.array([margin]), _DMatrixLike(y_true))

    np.testing.assert_allclose(grad[0], expected_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(hess[0], expected_hess, rtol=2e-4, atol=2e-5)


def test_lightgbm_and_sklearn_objectives_share_margin_derivatives():
    loss = WeightedBCELoss()
    y_true = np.array([0.0, 1.0])
    margin = np.array([-0.4, 0.9])

    lgb_grad, lgb_hess = LightGBMLossAdapter(loss).objective()(y_true, margin)
    sk_grad, sk_hess = resolve_custom_objective(loss)(y_true, margin)

    np.testing.assert_allclose(lgb_grad, sk_grad)
    np.testing.assert_allclose(lgb_hess, sk_hess)


def test_catboost_adapter_returns_negative_margin_derivatives():
    loss = WeightedBCELoss()
    y_true = np.array([0.0, 1.0])
    margin = np.array([-0.4, 0.9])
    expected_grad, expected_hess = LightGBMLossAdapter(loss).objective()(y_true, margin)

    derivatives = CatBoostLossAdapter(loss).objective().calc_ders_range(margin, y_true, None)
    der1, der2 = np.asarray(derivatives).T

    np.testing.assert_allclose(der1, -expected_grad)
    np.testing.assert_allclose(der2, -expected_hess)


def test_tabnet_adapter_keeps_autograd_graph_when_torch_is_available():
    torch = pytest.importorskip("torch")
    prediction = torch.tensor([0.2, 0.8], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.0, 1.0], dtype=torch.float64)

    value = TabNetLossAdapter(WeightedBCELoss()).loss_fn()(prediction, target)
    value.backward()

    assert value.requires_grad
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_tabnet_two_logit_gradient_matches_finite_difference_for_non_two_sample_batch():
    loss = WeightedBCELoss(pos_weight=1.4, neg_weight=0.7)
    logits = np.array([[-0.4, 0.6], [0.8, -0.2], [-0.1, 0.1]])
    y_true = np.array([1.0, 0.0, 1.0])
    _, gradient = _tabnet_binary_loss_and_gradient(loss, logits, y_true)

    numerical = np.zeros_like(logits)
    eps = 1e-5
    for row in range(logits.shape[0]):
        for column in range(logits.shape[1]):
            plus = logits.copy()
            minus = logits.copy()
            plus[row, column] += eps
            minus[row, column] -= eps
            plus_loss = _tabnet_binary_loss_and_gradient(loss, plus, y_true)[0]
            minus_loss = _tabnet_binary_loss_and_gradient(loss, minus, y_true)[0]
            numerical[row, column] = (plus_loss - minus_loss) / (2 * eps)

    assert gradient.shape == logits.shape
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-6)
