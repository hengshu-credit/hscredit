"""Zstandard 标准库与历史文件的读写兼容性测试。"""

import builtins
import pickle
import sys
from io import BytesIO

import pytest
import numpy as np

from hscredit.utils.io import load_pickle, save_pickle


@pytest.fixture
def compression_backend(monkeypatch):
    """新版 Python 验证未安装第三方库的路径，旧版验证原有实现。"""
    if sys.version_info >= (3, 14):
        original = builtins.__import__

        def without_zstandard(name, *args, **kwargs):
            if name == "zstandard":
                raise ModuleNotFoundError("测试环境未安装 zstandard", name="zstandard")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_zstandard)


@pytest.mark.parametrize("engine", ["pickle", "joblib", "dill", "cloudpickle"])
def test_zstd_roundtrip_with_available_backend(tmp_path, compression_backend, engine):
    value = {"数值": [1.25, None, -3], "分箱": ["缺失值", "合计"]}
    path = tmp_path / "模型.pkl.zst"
    save_pickle(value, path, engine=engine)
    assert load_pickle(path, engine=engine) == value
    with BytesIO(path.read_bytes()) as stream:
        assert load_pickle(stream, engine=engine, source_name=path.name) == value


def test_zstd_reads_frame_written_by_third_party_library(tmp_path, compression_backend):
    # zstandard.ZstdCompressor 写入的 protocol=4 文件，保持历史格式可读。
    frame = bytes.fromhex("28b52ffd202211010080049517000000000000007d948c0676616c756573945d94284b014b024b0365732e")
    path = tmp_path / "historical.pkl.zst"
    path.write_bytes(frame)
    assert load_pickle(path, engine="pickle") == {"values": [1, 2, 3]}


def test_zstd_preserves_uncompressed_payload(tmp_path, compression_backend):
    value = {"key": "value"}
    path = tmp_path / "model.pkl.zst"
    save_pickle(value, path, engine="pickle", protocol=4)
    if sys.version_info >= (3, 14):
        from compression import zstd
    else:
        import zstandard as zstd
    with zstd.open(path, "rb") as stream:
        assert stream.read() == pickle.dumps(value, protocol=4)


def test_joblib_zstd_preserves_arrays_across_buffer_boundaries(tmp_path, compression_backend):
    """非回退读取的 zstd 流也应完整恢复超过缓冲区大小的 NumPy 数据。"""
    value = {"特征": np.arange(8192, dtype=np.float64).reshape(1024, 8), "标签": np.arange(1024) % 2}
    path = tmp_path / "arrays.joblib.zst"
    save_pickle(value, path, engine="joblib")
    for source in (path, BytesIO(path.read_bytes())):
        actual = load_pickle(source, engine="joblib", source_name=path.name)
        for key in value:
            np.testing.assert_array_equal(actual[key], value[key])
