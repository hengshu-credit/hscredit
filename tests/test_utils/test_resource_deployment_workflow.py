"""查询流、报告释放和可迁移部署包的端到端语义验证。"""

import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hscredit import LogisticRegression, ModelReport, BinStatsAccumulator
from hscredit.database.stream import QueryStream
from hscredit.exceptions import StateError, SerializationError


class Resource:
    columns = ["bin", "target"]

    def __init__(self, interrupted=False):
        self.batches = iter([[(0, 0), (1, 1)], [(0, 1), (1, 0)], []])
        self.closed = 0
        self.calls = 0
        self.interrupted = interrupted

    def fetchmany(self, size):
        self.calls += 1
        if self.interrupted and self.calls == 2:
            raise KeyboardInterrupt
        return next(self.batches)

    def close(self):
        self.closed += 1


def test_stream_aggregation_is_complete_bounded_and_equal_to_batch():
    resource = Resource()
    stream = QueryStream(resource, chunksize=2, retain=False)
    result = stream.aggregate(BinStatsAccumulator(), lambda frame: (frame.bin, frame.target))
    expected = BinStatsAccumulator().update(np.array([0, 1, 0, 1], dtype=np.int64), [0, 1, 1, 0]).finalize()
    pd.testing.assert_frame_equal(result, expected)
    assert result.attrs["数据源"]["完整"]
    assert stream.resource_usage()["保留字节"] == 0
    assert resource.closed == 1


def test_interrupted_stream_cannot_publish_complete_aggregate():
    resource = Resource(interrupted=True)
    stream = QueryStream(resource, chunksize=2, retain=False)
    with pytest.raises(StateError, match="未完整"):
        stream.aggregate(BinStatsAccumulator(), lambda frame: (frame.bin, frame.target))
    assert resource.closed == 1


def test_partial_query_dataframe_cannot_silently_enter_report():
    from hscredit import feature_bin_stats

    stream = QueryStream(Resource(), chunksize=2)
    next(stream)
    stream.stop()
    partial = stream.to_dataframe()
    with pytest.raises(StateError, match="未完成"):
        feature_bin_stats(partial, "bin", target="target", rules=[0.5])


@pytest.mark.parametrize("kind", ["encoder", "binning"])
def test_query_completion_metadata_survives_woe_transform_without_tag_collision(kind):
    from hscredit import WOEEncoder, OptimalBinning
    from hscredit.utils.arguments import validate_source_completion

    full = pd.DataFrame({"bin": [0, 1, 0, 1]})
    transformer = (
        WOEEncoder(features=["bin"], n_jobs=1) if kind == "encoder" else OptimalBinning(method="quantile", n_jobs=1)
    )
    transformer.fit(full, [0, 1, 1, 0])
    stream = QueryStream(Resource(), chunksize=2)
    next(stream)
    stream.stop()
    partial = stream.to_dataframe()[["bin"]]
    transformed = transformer.transform(partial, metric="woe") if kind == "binning" else transformer.transform(partial)
    assert isinstance(transformed.attrs["hscredit_source"], str)
    assert transformed.attrs["hscredit_query_source"]["completed"] is False
    with pytest.raises(StateError, match="未完成"):
        validate_source_completion(transformed)


def sample_model():
    X = pd.DataFrame({"x": np.linspace(-3, 3, 40), "z": np.arange(40) % 3})
    y = np.tile([0, 1], 20)
    return LogisticRegression(calculate_stats=False).fit(X, y), X, y


def test_report_release_preserves_summary_and_rejects_missing_row_data():
    model, X, y = sample_model()
    report = ModelReport(model, X_train=X, y_train=y, n_jobs=1, cache_policy="none")
    expected = report.get_metrics()
    assert report._metrics_cache == {}
    report.release_data()
    pd.testing.assert_frame_equal(report.get_metrics(), expected)
    pd.testing.assert_frame_equal(report.get_metrics(), expected)
    assert report.resource_usage()["保留行数"] == 0
    with pytest.raises(StateError, match="已释放"):
        report.get_bin_table()


def test_report_release_drops_explicit_explanation_data_and_keeps_metrics():
    import gc
    import weakref

    model, X, y = sample_model()
    background = pd.concat([X] * 1000, ignore_index=True)
    reference = weakref.ref(background)
    report = ModelReport(
        model, X_train=X, y_train=y, n_jobs=1, explain_config={"enabled": False, "background_data": background}
    )
    before = report.resource_usage()["估算保留字节"]
    expected = report.get_metrics()
    del background
    report.release_data()
    gc.collect()
    assert reference() is None
    assert report.resource_usage()["估算保留字节"] < before / 10
    pd.testing.assert_frame_equal(report.get_metrics(), expected)
    report.release_cache()
    with pytest.raises(StateError, match="已释放"):
        report.get_metrics()


def test_deployment_roundtrip_relocation_and_digest(tmp_path):
    model, X, _ = sample_model()
    expected = model.predict_proba(X)
    manifest = model.save_deployment(tmp_path / "original", validation_data=X)
    restored = LogisticRegression.load_deployment(manifest, metadata_policy="strict")
    np.testing.assert_allclose(restored.predict_proba(X[X.columns[::-1]]), expected)
    shutil.copytree(tmp_path / "original", tmp_path / "relocated")
    relocated = tmp_path / "relocated" / "model.manifest.json"
    np.testing.assert_allclose(LogisticRegression.load_deployment(relocated).predict_proba(X), expected)
    entry = next(item for item in json.loads(relocated.read_text(encoding="utf-8"))["制品"] if item["type"] == "model")
    (relocated.parent / entry["relative_path"]).write_bytes(b"corrupt")
    with pytest.raises(SerializationError, match="摘要"):
        LogisticRegression.load_deployment(relocated)


def test_deployment_requires_fitted_model_and_tracks_used_backend(tmp_path):
    from sklearn.exceptions import NotFittedError
    from hscredit import QuantileBinning, TargetEncoder

    for model in (LogisticRegression(), QuantileBinning(), TargetEncoder()):
        with pytest.raises((NotFittedError, ValueError)):
            model.save_deployment(tmp_path / "unfitted")
    pytest.importorskip("lightgbm")
    from hscredit import LightGBM

    X = pd.DataFrame({"x": np.arange(60)})
    model = LightGBM(n_estimators=2, n_jobs=1).fit(X, np.arange(60) % 2)
    manifest = model.save_deployment(tmp_path / "fitted", validation_data=X)
    entry = next(
        item for item in json.loads(Path(manifest).read_text(encoding="utf-8"))["制品"] if item["type"] == "deployment"
    )
    metadata = json.loads((Path(manifest).parent / entry["relative_path"]).read_text(encoding="utf-8"))
    assert "lightgbm" in metadata["dependencies"]
    assert "catboost" not in metadata["dependencies"]
