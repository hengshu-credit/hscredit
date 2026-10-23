"""损失、金融指标统一出口与旧API兼容验证。"""

import numpy as np
import pytest
from hscredit import get_loss, get_metric, list_losses, list_metrics
from hscredit.core.models.losses import BaseMetric


@pytest.mark.parametrize("key", list_losses()["标识"])
def test_all_losses_expose_one_metric_method_with_independent_snapshots(key):
    loss = get_loss(key)
    proxy = loss.metric()
    business = loss.metric(kind="business", name="业务效果")
    assert proxy.direction == "minimize"
    assert isinstance(business, BaseMetric)
    assert business.name == "业务效果"
    assert get_loss(loss) is not loss
    assert get_metric(loss).loss is not loss
    assert get_metric(loss, kind="business").direction == business.direction
    with pytest.warns(DeprecationWarning, match="metric"):
        assert loss.business_metric().direction == business.direction


@pytest.mark.parametrize("key", list_metrics()["标识"])
def test_all_registry_metrics_work_offline_and_preserve_direction(key):
    kwargs = {"expected": np.array([0.1, 0.2, 0.7, 0.9])} if key == "psi" else {}
    metric = get_metric(key, **kwargs)
    assert np.isfinite(metric.evaluate([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9]))
    assert get_metric(metric) is not metric
    assert metric.direction == list_metrics().set_index("标识").loc[key, "优化方向"]


def test_unified_auc_preserves_probability_direction():
    assert get_metric("auc")([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0
    assert get_metric("gini")([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == -1


def test_metric_parameters_snapshot_and_sample_metadata():
    loss = get_loss("cost_sensitive", fn_cost=7, fp_cost=2)
    metric = get_metric(loss, kind="business")
    loss.fn_cost = 99
    assert metric.fn_cost == 7
    amount = get_loss("amount_weighted", amounts=[1, 2])
    validation = amount.metric(kind="business", amounts=[5, 6])
    np.testing.assert_array_equal(validation.loss.amounts_, [5, 6])
    np.testing.assert_array_equal(amount.amounts_, [1, 2])


def test_custom_function_and_invalid_registry_configuration():
    metric = get_metric(lambda y, p: np.mean((y - p) ** 2), name="误差", greater_is_better=False)
    assert metric([0, 1], [0.2, 0.8]) == pytest.approx(0.04)
    with pytest.raises(ValueError, match="未知损失"):
        get_loss("typo")
    with pytest.raises(ValueError, match="未知指标"):
        get_metric("typo")
    with pytest.raises(ValueError, match="kind"):
        get_loss("focal").metric(kind="typo")
    with pytest.raises(ValueError, match="覆盖"):
        get_metric(metric, name="覆盖")
