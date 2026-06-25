"""统一制品序列化协议测试."""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import UniformBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.models import LogisticRegression, ScoreCard
from hscredit.exceptions import SerializationError


@pytest.fixture
def binary_data():
    X = pd.DataFrame(
        {
            "年龄": [20, 25, 30, 35, 40, 45, 50, 55],
            "城市": ["北京", "上海", "北京", "广州", "上海", "广州", "北京", "上海"],
        }
    )
    y = pd.Series([0, 0, 0, 1, 0, 1, 1, 1], name="目标")
    return X, y


def test_binner_artifact_roundtrip(tmp_path, binary_data):
    X, y = binary_data
    binner = UniformBinning(max_n_bins=3).fit(X[["年龄"]], y)
    expected = binner.transform(X[["年龄"]], metric="woe")

    path = binner.save_artifact(tmp_path / "binner.joblib")
    restored = UniformBinning.load_artifact(path)

    pd.testing.assert_frame_equal(restored.transform(X[["年龄"]], metric="woe"), expected)
    assert restored.get_artifact_metadata()["kind"] == "分箱器"


def test_encoder_artifact_roundtrip(tmp_path, binary_data):
    X, y = binary_data
    encoder = WOEEncoder(cols=["城市"]).fit(X, y)
    expected = encoder.transform(X)

    path = encoder.save_artifact(tmp_path / "encoder.joblib")
    restored = WOEEncoder.load_artifact(path)

    pd.testing.assert_frame_equal(restored.transform(X), expected)
    assert restored.get_artifact_metadata()["kind"] == "编码器"


def test_model_artifact_roundtrip(tmp_path, binary_data):
    X, y = binary_data
    model = LogisticRegression(max_iter=200).fit(X[["年龄"]], y)
    expected = model.predict_proba(X[["年龄"]])

    path = model.save_artifact(tmp_path / "model.joblib")
    restored = LogisticRegression.load_artifact(path)

    np.testing.assert_allclose(restored.predict_proba(X[["年龄"]]), expected)
    assert restored.get_artifact_metadata()["kind"] == "风险模型"


def test_scorecard_artifact_roundtrip(tmp_path, binary_data):
    X, y = binary_data
    X_woe = pd.DataFrame({"年龄": np.linspace(-1.0, 1.0, len(X))})
    scorecard = ScoreCard().fit(X_woe, y)
    expected = scorecard.predict(X_woe, input_type="woe")

    path = scorecard.save_artifact(tmp_path / "scorecard.joblib")
    restored = ScoreCard.load_artifact(path)

    np.testing.assert_allclose(restored.predict(X_woe, input_type="woe"), expected)
    assert restored.get_artifact_metadata()["kind"] == "评分卡"


def test_artifact_rejects_incompatible_target_type(tmp_path, binary_data):
    X, y = binary_data
    encoder = WOEEncoder(cols=["城市"]).fit(X, y)
    path = encoder.save_artifact(tmp_path / "encoder.joblib")

    with pytest.raises(SerializationError, match="不能作为 UniformBinning 加载"):
        UniformBinning.load_artifact(path)
