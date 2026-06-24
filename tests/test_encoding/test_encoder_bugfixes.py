"""编码器模块缺陷修复回归测试.

锁定本轮排查修复的问题，防止回归：
- BUG-1 WOE export/load 数字串类别往返破坏
- BUG-2 GBMEncoder embedding 列名与 feature_names_ 不一致
- BUG-3 OneHot/Ordinal 混合类型 sorted 崩溃
- BUG-4 WOE load(update=True) 未 fit 实例 TypeError
- BUG-5 base import_mapping 把 dict 误转 Series
- BUG-6 TargetEncoder noise 每列重置随机数
- BUG-7 handle_missing='error' fit 期未校验
- API   get_mapping/inverse_transform 统一、import_mapping 状态还原
"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.encoders import (
    WOEEncoder,
    TargetEncoder,
    CountEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    QuantileEncoder,
    CatBoostEncoder,
    CardinalityEncoder,
)
from hscredit.exceptions import NotFittedError, FeatureNotFoundError


# --------------------------------------------------------------------------- #
# BUG-1 WOE export/load 往返
# --------------------------------------------------------------------------- #
def test_woe_export_load_roundtrip_numeric_string_categories():
    """数字型字符串类别（如城市码）经 export->load 往返后编码结果须一致。"""
    X = pd.DataFrame({"code": ["100", "200", "100", "300", "200", "300"]})
    y = pd.Series([0, 1, 0, 1, 1, 0])

    enc = WOEEncoder(cols=["code"]).fit(X, y)
    before = enc.transform(X.copy())["code"].round(8).tolist()

    enc2 = WOEEncoder(cols=["code"]).load(enc.export())
    after = enc2.transform(X.copy())["code"].round(8).tolist()

    assert before == after


def test_woe_export_load_roundtrip_numeric_categories():
    """真数值类别经 export(str键)->load->transform(数值输入) 须靠字符串回退命中。"""
    X = pd.DataFrame({"age_bin": [1, 2, 1, 3, 2, 3]})
    y = pd.Series([0, 1, 0, 1, 1, 0])

    enc = WOEEncoder(cols=["age_bin"]).fit(X, y)
    before = enc.transform(X.copy())["age_bin"].round(8).tolist()

    enc2 = WOEEncoder(cols=["age_bin"]).load(enc.export())
    after = enc2.transform(X.copy())["age_bin"].round(8).tolist()

    assert before == after


def test_woe_unknown_still_handled_after_load():
    """load 后未知类别仍按 handle_unknown='value' 编码为 0.0。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "C"]})
    y = pd.Series([0, 1, 0, 1])
    enc = WOEEncoder(cols=["city"]).fit(X, y)
    enc2 = WOEEncoder(cols=["city"]).load(enc.export())

    out = enc2.transform(pd.DataFrame({"city": ["__new__"]}))["city"].tolist()
    assert out == [0.0]


def test_encoder_sets_sklearn_input_feature_attributes():
    """编码器 fit 后应提供 sklearn 兼容的输入特征元数据。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "C"], "channel": ["app", "web", "app", "api"]})
    y = pd.Series([0, 1, 0, 1])
    enc = WOEEncoder(cols=["city"]).fit(X, y)

    assert enc.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(enc.feature_names_in_, X.columns.to_numpy(dtype=object))


# --------------------------------------------------------------------------- #
# BUG-3 混合类型排序不崩溃
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [OneHotEncoder, OrdinalEncoder])
def test_mixed_type_categories_no_crash(cls):
    """类别同时含 int 与 str 时不再因 sorted 抛 TypeError。"""
    X = pd.DataFrame({"mix": [1, "a", 2, "b", 1]})
    out = cls(cols=["mix"]).fit_transform(X)
    assert len(out) == len(X)


# --------------------------------------------------------------------------- #
# BUG-4 WOE load(update=True) 未 fit 实例
# --------------------------------------------------------------------------- #
def test_woe_load_update_on_fresh_encoder():
    """在未 fit 的新实例上 load(update=True) 不应崩溃。"""
    enc = WOEEncoder()
    enc.load({"f1": {"A": 0.5}}, update=True)
    assert "f1" in enc.mapping_
    assert enc.mapping_["f1"]["A"] == 0.5


# --------------------------------------------------------------------------- #
# BUG-5 / API import_mapping 保持 dict 契约
# --------------------------------------------------------------------------- #
def test_import_mapping_keeps_dict_contract():
    X = pd.DataFrame({"city": ["A", "B", "A", "C", "B"]})
    enc = OrdinalEncoder(cols=["city"]).fit(X)
    assert isinstance(enc.mapping_["city"], dict)

    enc2 = OrdinalEncoder()
    enc2.import_mapping(enc.export_mapping())
    assert isinstance(enc2.mapping_["city"], dict)
    # 往返后 transform 结果一致
    assert enc.transform(X.copy())["city"].tolist() == enc2.transform(X.copy())["city"].tolist()


# --------------------------------------------------------------------------- #
# BUG-6 TargetEncoder noise 各列独立
# --------------------------------------------------------------------------- #
def test_target_encoder_noise_independent_across_columns():
    """相同输入的两列，加噪后不应得到完全相同的序列（噪声应各列独立）。"""
    X = pd.DataFrame({"c1": ["A", "B", "A", "C", "B"], "c2": ["A", "B", "A", "C", "B"]})
    y = pd.Series([0, 1, 0, 1, 1])
    enc = TargetEncoder(cols=["c1", "c2"], noise=0.1, random_state=42)
    enc.fit(X, y)
    out = enc.transform(X.copy(), y)
    assert not np.allclose(out["c1"], out["c2"])


# --------------------------------------------------------------------------- #
# BUG-7 handle_missing='error' fit 期即报错
# --------------------------------------------------------------------------- #
def test_handle_missing_error_raises_on_fit():
    X = pd.DataFrame({"city": ["A", np.nan, "B", "A"]})
    y = pd.Series([0, 1, 0, 1])
    with pytest.raises(ValueError):
        WOEEncoder(cols=["city"], handle_missing="error").fit(X, y)


# --------------------------------------------------------------------------- #
# API 统一：get_mapping 异常类型
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [WOEEncoder, TargetEncoder, OrdinalEncoder, CardinalityEncoder, CountEncoder])
def test_get_mapping_unknown_col_raises_feature_not_found(cls):
    X = pd.DataFrame({"city": ["A", "B", "A", "C"]})
    y = pd.Series([0, 1, 0, 1])
    enc = cls(cols=["city"])
    enc.fit(X, y) if cls in (WOEEncoder, TargetEncoder) else enc.fit(X)
    with pytest.raises(FeatureNotFoundError):
        enc.get_mapping("not_a_col")


def test_get_mapping_before_fit_raises_not_fitted():
    enc = OrdinalEncoder(cols=["city"])
    with pytest.raises(NotFittedError):
        enc.get_mapping("city")


# --------------------------------------------------------------------------- #
# API 统一：inverse_transform 占位
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [WOEEncoder, TargetEncoder, CountEncoder, QuantileEncoder, CatBoostEncoder])
def test_inverse_transform_unsupported_raises_not_implemented(cls):
    X = pd.DataFrame({"city": ["A", "B", "A"]})
    y = pd.Series([0, 1, 0])
    enc = cls(cols=["city"]).fit(X, y)
    with pytest.raises(NotImplementedError):
        enc.inverse_transform(enc.transform(X.copy()))


# --------------------------------------------------------------------------- #
# API import_mapping 还原 transform 必需的额外状态
# --------------------------------------------------------------------------- #
def test_import_mapping_restores_global_mean_for_unknown_fill():
    """Target/CatBoost/Quantile：import 后未知类别须用真实全局均值填充，而非 0。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "C", "B"]})
    y = pd.Series([0, 1, 0, 1, 1])
    enc = TargetEncoder(cols=["city"]).fit(X, y)
    enc2 = TargetEncoder(cols=["city"])
    enc2.import_mapping(enc.export_mapping())

    unknown = pd.DataFrame({"city": ["__unseen__"]})
    expected = enc.transform(unknown.copy())["city"].iloc[0]
    got = enc2.transform(unknown.copy())["city"].iloc[0]
    assert np.isclose(got, expected)
    assert np.isclose(enc2.global_mean_, enc.global_mean_)


def test_import_mapping_restores_onehot_categories():
    """OneHot：import 后 transform 须仍生成完整独热列。"""
    X = pd.DataFrame({"city": ["A", "B", "A", "C", "B"]})
    enc = OneHotEncoder(cols=["city"]).fit(X)
    enc2 = OneHotEncoder()
    enc2.import_mapping(enc.export_mapping())

    cols_before = list(enc.transform(X.copy()).columns)
    cols_after = list(enc2.transform(X.copy()).columns)
    assert cols_before == cols_after
    assert any(c.startswith("city_") for c in cols_after)


# --------------------------------------------------------------------------- #
# BUG-2 GBMEncoder embedding 列名与 feature_names_ 一致
# --------------------------------------------------------------------------- #
def test_gbm_embedding_feature_names_match_output_columns():
    pytest.importorskip("xgboost")
    from hscredit.core.encoders import GBMEncoder

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((X["f0"] + rng.normal(scale=0.5, size=200) > 0).astype(int))

    enc = GBMEncoder(
        model_type="xgboost",
        n_estimators=5,
        max_depth=2,
        output_type="embedding",
        random_state=0,
        drop_origin=True,
    )
    out = enc.fit_transform(X, y)
    assert set(enc.feature_names_) == set(out.columns)
    assert all(c.startswith("gbm_emb_") for c in out.columns)
