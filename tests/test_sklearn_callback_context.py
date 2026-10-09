"""事务拟合保留父级调用上下文，允许 sklearn Pipeline 正常完成清理。"""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hscredit.core.binning import OptimalBinning, QuantileBinning
from hscredit.core.encoders import GBMEncoder, TargetEncoder, WOEEncoder


class _ParentContext:
    def __deepcopy__(self, memo):
        raise AssertionError("父级运行上下文不能被当作拟合状态复制")


@pytest.fixture(params=["target", "target_oof", "woe", "gbm", "optimal", "quantile"])
def transformer(request):
    if request.param == "gbm":
        pytest.importorskip("xgboost")
        return GBMEncoder(target="FPD", model_type="xgboost", n_estimators=3, max_depth=2, n_jobs=1)
    if request.param in {"target", "target_oof"}:
        return TargetEncoder(
            cols=["特征"],
            target="FPD",
            training_mode="oof" if request.param == "target_oof" else "in_sample",
            cv=2,
            n_jobs=1,
        )
    if request.param == "woe":
        return WOEEncoder(cols=["特征"], target="FPD", n_jobs=1)
    cls = OptimalBinning if request.param == "optimal" else QuantileBinning
    kwargs = {"method": "quantile"} if request.param == "optimal" else {}
    return cls(target="FPD", max_n_bins=3, n_jobs=1, **kwargs)


@pytest.fixture
def frame():
    return pd.DataFrame({"特征": [0, 1, 2, 3] * 10, "FPD": [0, 1, 0, 1] * 10})


@pytest.mark.parametrize("method", ["fit", "fit_transform"])
@pytest.mark.parametrize("has_context", [True, False])
def test_fit_keeps_context_identity_until_owner_cleanup(transformer, frame, method, has_context):
    """成功提交不清除父上下文，包括显式 None；clone 不携带运行上下文。"""
    context = _ParentContext() if has_context else None
    transformer._parent_callback_ctx = context

    getattr(transformer, method)(frame, frame.FPD)

    assert "_parent_callback_ctx" in transformer.__dict__
    assert transformer._parent_callback_ctx is context
    assert not hasattr(clone(transformer), "_parent_callback_ctx")
    del transformer._parent_callback_ctx
    expected = transformer.transform(frame[["特征"]])
    restored = pickle.loads(pickle.dumps(transformer))
    assert not hasattr(restored, "_parent_callback_ctx")
    np.testing.assert_allclose(restored.transform(frame[["特征"]]), expected)


def test_failed_fit_keeps_parent_context_and_previous_transform(transformer, frame):
    """失败仍抛出原始输入错误，旧拟合状态和父上下文都保留。"""
    transformer.fit(frame, frame.FPD)
    expected = transformer.transform(frame[["特征"]])
    context = _ParentContext()
    transformer._parent_callback_ctx = context

    with pytest.raises(ValueError, match="长度|样本|数量"):
        transformer.fit(frame, frame.FPD.iloc[:-1])

    assert transformer._parent_callback_ctx is context
    np.testing.assert_allclose(transformer.transform(frame[["特征"]]), expected)


@pytest.mark.parametrize("cached", [False, True])
def test_native_pipeline_cleans_context_after_fit(transformer, frame, cached, tmp_path):
    """真实 Pipeline 的正常和缓存克隆路径都能预测，结束后没有残留上下文。"""
    pipeline = Pipeline(
        [("transformer", transformer), ("model", LogisticRegression())],
        memory=str(tmp_path / "cache") if cached else None,
    ).fit(frame, frame.FPD)

    assert pipeline.predict(frame[["特征"]]).shape == (len(frame),)
    assert not hasattr(pipeline["transformer"], "_parent_callback_ctx")
