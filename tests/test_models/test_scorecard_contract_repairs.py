"""ScoreCard 跨格式评分契约回归测试。"""

import ast
import copy
import shutil
import sqlite3
import subprocess

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import RoundScoreCard, ScoreCard
from hscredit.exceptions import ValidationError


def _fit_simple_scorecard(direction="descending", scorecard_cls=ScoreCard, **kwargs):
    X = pd.DataFrame({"x": np.repeat([0.0, 1.0, 2.0, 3.0], 30)})
    y = pd.Series(
        ([0] * 27 + [1] * 3)
        + ([0] * 21 + [1] * 9)
        + ([0] * 9 + [1] * 21)
        + ([0] * 3 + [1] * 27)
    )
    binner = OptimalBinning(method="target_bad_rate", max_n_bins=4, n_jobs=1).fit(X, y)
    card = scorecard_cls(direction=direction, binner=binner, **kwargs)
    card.fit(binner.transform(X, metric="woe"), y, input_type="woe")
    return card, binner, X, y


def _fit_comma_category_scorecard():
    X = pd.DataFrame({"cat": ["A,B"] * 60 + ["C"] * 60})
    y = pd.Series(([0] * 50 + [1] * 10) + ([0] * 10 + [1] * 50))
    binner = OptimalBinning(
        method="target_bad_rate",
        user_splits={"cat": [["A,B"], ["C"]]},
        user_splits_fixed=True,
        min_n_bins=2,
        max_n_bins=2,
        n_jobs=1,
    ).fit(X, y)
    card = ScoreCard(binner=binner).fit(binner.transform(X, metric="woe"), y, input_type="woe")
    return card, X


def _python_deployment_scores(card, X, decimal=12):
    namespace = {}
    exec(card.export_deployment_code(language="python", decimal=decimal), namespace)
    return np.asarray(
        [namespace["calculate_score"](row) for row in X.to_dict("records")],
        dtype=float,
    )


def _sqlite_deployment_scores(card, X, decimal=12):
    connection = sqlite3.connect(":memory:")
    try:
        X.to_sql("your_table", connection, index=False)
        sql = card.export_deployment_code(language="sql", decimal=decimal)
        return np.asarray([row[0] for row in connection.execute(sql)], dtype=float)
    finally:
        connection.close()


def test_ascending_predict_probability_paths_share_one_direction_contract():
    """防止 feature 评分翻转、概率评分不翻转，导致同模型输出相反方向。"""
    card, _, X, _ = _fit_simple_scorecard(direction="ascending")

    feature_scores = card.predict(X, input_type="raw")
    probability_scores = card.predict_score(X, input_type="raw")
    bad_probability = card.predict_proba(X, input_type="raw")[:, 1]

    np.testing.assert_allclose(feature_scores, probability_scores, atol=1e-10)
    np.testing.assert_allclose(card.inverse_transform(feature_scores), bad_probability, atol=1e-12)


def test_raw_prediction_without_transformer_fails_instead_of_scoring_raw_values_as_woe():
    """防止缺少 binner/encoder 时把原始数值静默当成 WOE，输出看似正常的错误分数。"""
    X_woe = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 40)})
    y = pd.Series([0] * 20 + [1] * 20)
    card = ScoreCard().fit(X_woe, y, input_type="woe")
    X_raw = pd.DataFrame({"x": np.linspace(100.0, 1000.0, 40)})

    with pytest.raises(ValidationError, match="缺少可用的分箱器或WOE编码器"):
        card.predict(X_raw, input_type="raw")


def test_default_rule_export_is_complete_and_self_roundtrippable_without_original_binner():
    """防止默认 export 丢失截距和方向，使本库 load 后分数整体漂移。"""
    card, _, X, _ = _fit_simple_scorecard(direction="ascending")
    expected = card.predict(X, input_type="raw")

    rules = card.export(decimal=12)

    assert rules["__meta__"]["format"] == "hscredit-scorecard-rules"
    restored = ScoreCard().load_rules(rules)
    np.testing.assert_allclose(restored.predict(X, input_type="raw"), expected, atol=1e-9)


def test_default_rule_roundtrip_preserves_category_values_containing_commas():
    """防止结构化类别元数据退化成逗号分隔字符串。"""
    card, X = _fit_comma_category_scorecard()
    rules = card.export(decimal=12)

    restored = ScoreCard().load_rules(rules)

    assert restored.rules_["cat"]["bins"] == [["A,B"], ["C"]]
    np.testing.assert_allclose(
        restored.predict(X, input_type="raw"),
        card.predict(X, input_type="raw"),
        atol=1e-9,
    )


def test_rule_dataframe_carries_metadata_for_complete_roundtrip():
    """防止 to_frame=True 丢掉 include_meta=True 请求的完整评分状态。"""
    card, _, X, _ = _fit_simple_scorecard(direction="ascending")
    frame = card.export(to_frame=True, decimal=12)

    assert frame.attrs["scorecard_meta"]["format"] == "hscredit-scorecard-rules"
    restored = ScoreCard().load_rules(frame)
    np.testing.assert_allclose(
        restored.predict(X, input_type="raw"),
        card.predict(X, input_type="raw"),
        atol=1e-9,
    )


def test_explicit_toad_compatibility_export_keeps_legacy_metadata_free_shape():
    """防止完整格式默认值破坏显式请求的外部兼容规则。"""
    card, _, _, _ = _fit_simple_scorecard()

    rules = card.export(compatibility="toad")

    assert "__meta__" not in rules
    assert set(rules) == set(card.feature_names_)


def test_python_deployment_uses_zero_contribution_for_unknown_category():
    """防止部署端把未知类别回退到第一箱，而训练分箱器使用 WOE=0。"""
    card, _ = _fit_comma_category_scorecard()
    unknown = pd.DataFrame({"cat": ["UNKNOWN"]})

    expected = card.predict(unknown, input_type="raw")
    deployed = _python_deployment_scores(card, unknown)

    np.testing.assert_allclose(deployed, expected, atol=1e-9)


def test_python_deployment_preserves_numeric_category_types():
    """防止把数值类别渲染为字符串字面量，导致除默认箱外全部失配。"""
    X = pd.DataFrame({"code": [1] * 60 + [2] * 60 + [3] * 60})
    y = pd.Series(
        ([0] * 50 + [1] * 10)
        + ([0] * 30 + [1] * 30)
        + ([0] * 10 + [1] * 50)
    )
    binner = OptimalBinning(
        method="target_bad_rate",
        cat_cutoff=10,
        user_splits={"code": [[1], [2], [3]]},
        user_splits_fixed=True,
        min_n_bins=3,
        max_n_bins=3,
        n_jobs=1,
    ).fit(X, y)
    card = ScoreCard(binner=binner).fit(binner.transform(X, metric="woe"), y, input_type="woe")
    sample = pd.DataFrame({"code": [1, 2, 3]})

    np.testing.assert_allclose(
        _python_deployment_scores(card, sample),
        card.predict(sample, input_type="raw"),
        atol=1e-9,
    )


def test_python_and_sql_deployment_apply_the_same_score_clipping_as_predict():
    """防止部署代码返回未裁剪总分，而模型已限制在 lower/upper。"""
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "lower": 0.0,
                "upper": 100.0,
                "feature_names": ["x"],
                "coef": [1.0],
            },
            "x": {"[-inf, 0)": 50.0, "[0, +inf)": 60.0},
        }
    )
    sample = pd.DataFrame({"x": [-1.0, 1.0]})
    expected = np.array([100.0, 100.0])

    np.testing.assert_allclose(card.predict(sample, input_type="raw"), expected)
    np.testing.assert_allclose(card.transform(np.array([0.5])), np.array([100.0]))
    np.testing.assert_allclose(_python_deployment_scores(card, sample), expected)
    np.testing.assert_allclose(_sqlite_deployment_scores(card, sample), expected)


def test_clip_false_is_respected_by_model_and_deployment_code():
    """防止设置 ``clip=False`` 后导出端仍无条件截断分数。"""
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "lower": 0.0,
                "upper": 100.0,
                "clip": False,
                "feature_names": ["x"],
                "coef": [1.0],
            },
            "x": {"[-inf, +inf)": 50.0},
        }
    )
    sample = pd.DataFrame({"x": [1.0]})
    expected = np.array([650.0])

    np.testing.assert_allclose(card.predict(sample, input_type="raw"), expected)
    np.testing.assert_allclose(_python_deployment_scores(card, sample), expected)
    np.testing.assert_allclose(_sqlite_deployment_scores(card, sample), expected)


def test_full_range_numeric_bin_generates_valid_code_and_scores_normal_values():
    """防止全范围箱被渲染成带行尾注释的非法 ``if False:``。"""
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "feature_names": ["x"],
                "coef": [1.0],
            },
            "x": {"[-inf, +inf)": 10.0, "missing": -5.0},
        }
    )
    sample = pd.DataFrame({"x": [1.0, np.nan]})
    code = card.export_deployment_code(language="python", decimal=12)

    ast.parse(code)
    np.testing.assert_allclose(_python_deployment_scores(card, sample), np.array([610.0, 595.0]))
    np.testing.assert_allclose(_sqlite_deployment_scores(card, sample), np.array([610.0, 595.0]))


def test_scientific_notation_interval_is_shared_by_rule_python_and_sql_scoring():
    """防止部署端旧正则不能识别科学计数法边界。"""
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "feature_names": ["x"],
                "coef": [1.0],
            },
            "x": {"[-inf, 1e-05)": 10.0, "[1e-05, +inf)": 20.0},
        }
    )
    sample = pd.DataFrame({"x": [0.0, 0.1]})
    expected = np.array([610.0, 620.0])

    np.testing.assert_allclose(card.predict(sample, input_type="raw"), expected)
    np.testing.assert_allclose(_python_deployment_scores(card, sample), expected)
    np.testing.assert_allclose(_sqlite_deployment_scores(card, sample), expected)


def test_structured_category_group_can_contain_missing_and_regular_values():
    """防止 ``[missing, 'A']`` 在离线/SQL/Python 三端被拆成不同语义。"""
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "feature_names": ["cat"],
                "feature_types": {"cat": "categorical"},
                "categorical_bins": {"cat": [[None, "A"], ["B"]]},
                "coef": [1.0],
            },
            "cat": {"None, A": 20.0, "B": 10.0},
        }
    )
    sample = pd.DataFrame({"cat": [None, "A", "B", "UNKNOWN"]})
    expected = np.array([620.0, 620.0, 610.0, 600.0])

    np.testing.assert_allclose(card.predict(sample, input_type="raw"), expected)
    np.testing.assert_allclose(_python_deployment_scores(card, sample), expected)
    np.testing.assert_allclose(_sqlite_deployment_scores(card, sample), expected)


def test_unsafe_feature_name_is_escaped_in_python_and_sql_deployment_code():
    """防止引号字段名破坏生成代码或被解释为 SQL 片段。"""
    feature = 'bad"name'
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "feature_names": [feature],
                "coef": [1.0],
            },
            feature: {"[-inf, 0)": 10.0, "[0, +inf)": 20.0},
        }
    )
    sample = pd.DataFrame({feature: [-1.0, 1.0]})
    expected = np.array([610.0, 620.0])

    ast.parse(card.export_deployment_code(language="python", decimal=12))
    np.testing.assert_allclose(_python_deployment_scores(card, sample), expected)
    np.testing.assert_allclose(_sqlite_deployment_scores(card, sample), expected)


def test_export_load_then_python_deployment_preserves_structured_categories():
    """防止规则重载后部署生成器再次退化为逗号拆分类别。"""
    card, X = _fit_comma_category_scorecard()
    restored = ScoreCard().load_rules(card.export(decimal=12))

    np.testing.assert_allclose(
        _python_deployment_scores(restored, X),
        card.predict(X, input_type="raw"),
        atol=1e-9,
    )


def test_rule_roundtrip_preserves_handle_unknown_raise_policy():
    """评分卡规则恢复后不能把分箱器的未知类别报错策略退化为默认回退分。"""
    X = pd.DataFrame({"cat": ["A"] * 60 + ["B"] * 60})
    y = pd.Series(([0] * 50 + [1] * 10) + ([0] * 10 + [1] * 50))
    binner = OptimalBinning(
        method="target_bad_rate",
        user_splits={"cat": [["A"], ["B"]]},
        user_splits_fixed=True,
        min_n_bins=2,
        max_n_bins=2,
        handle_unknown="raise",
        n_jobs=1,
    ).fit(X, y)
    card = ScoreCard(binner=binner).fit(binner.transform(X, metric="woe"), y, input_type="woe")
    restored = ScoreCard().load_rules(card.export(decimal=12))

    for candidate in (card, restored):
        with pytest.raises(ValueError, match="特征 'cat'.*未知类别.*'UNKNOWN'"):
            candidate.predict(pd.DataFrame({"cat": ["UNKNOWN"]}), input_type="raw")


def test_rule_loader_normalizes_legacy_handle_unknown_value_to_minus_three():
    """历史规则元数据中的 value 必须按当前统一契约恢复为未知箱 -3。"""
    card, X = _fit_comma_category_scorecard()
    rules = card.export(decimal=12)
    rules["__meta__"]["handle_unknown"] = "value"

    restored = ScoreCard().load_rules(rules)

    assert restored.binner.handle_unknown == -3
    np.testing.assert_allclose(
        restored.predict(X, input_type="raw"),
        card.predict(X, input_type="raw"),
        atol=1e-9,
    )


def test_pmml_decode_bug_does_not_accept_a_stale_existing_destination(tmp_path, monkeypatch):
    """防止上次留下的 PMML 被误判为本次导出成功。"""
    sklearn2pmml_module = pytest.importorskip("sklearn2pmml")
    pytest.importorskip("sklearn_pandas")
    card, _, _, _ = _fit_simple_scorecard()
    destination = tmp_path / "scorecard.pmml"
    destination.write_text("<OLD/>", encoding="utf-8")

    def fail_without_writing(*args, **kwargs):
        raise TypeError("object of type 'NoneType' has no len()")

    monkeypatch.setattr(sklearn2pmml_module, "sklearn2pmml", fail_without_writing)

    with pytest.raises(TypeError, match="NoneType"):
        card.export_pmml(str(destination))

    assert destination.read_text(encoding="utf-8") == "<OLD/>"


def test_pmml_is_generated_to_a_temporary_file_then_atomically_replaced(tmp_path, monkeypatch):
    """防止导出失败时覆盖一个原本有效的 PMML 文件。"""
    sklearn2pmml_module = pytest.importorskip("sklearn2pmml")
    pytest.importorskip("sklearn_pandas")
    card, _, _, _ = _fit_simple_scorecard()
    destination = tmp_path / "scorecard.pmml"
    destination.write_text("<OLD/>", encoding="utf-8")
    generated_paths = []

    def write_pmml(pipeline, pmml_file, with_repr=True, debug=False):
        generated_paths.append(pmml_file)
        with open(pmml_file, "w", encoding="utf-8") as handle:
            handle.write("<PMML/>")

    monkeypatch.setattr(sklearn2pmml_module, "sklearn2pmml", write_pmml)

    card.export_pmml(str(destination))

    assert len(generated_paths) == 1
    assert generated_paths[0] != str(destination)
    assert destination.read_text(encoding="utf-8") == "<PMML/>"


def test_pmml_prediction_applies_the_same_score_clipping_as_predict(tmp_path):
    """防止 PMML 只输出未裁剪的线性总分。"""
    pytest.importorskip("sklearn2pmml")
    pytest.importorskip("sklearn_pandas")
    pypmml = pytest.importorskip("pypmml")
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "base_score": 750.0,
                "direction": "descending",
                "pdo": 60.0,
                "rate": 2.0,
                "base_odds": 35.0,
                "lower": 0.0,
                "upper": 100.0,
                "feature_names": ["x"],
                "coef": [1.0],
            },
            "x": {"[-inf, 0)": 50.0, "[0, +inf)": 60.0},
        }
    )
    sample = pd.DataFrame({"x": [-1.0, 1.0]})
    destination = tmp_path / "clipped.pmml"

    card.export_pmml(str(destination))
    pmml_scores = pypmml.Model.load(str(destination)).predict(sample)["predicted_score"].to_numpy()

    np.testing.assert_allclose(pmml_scores, card.predict(sample, input_type="raw"))


def test_ascending_round_scorecard_export_load_does_not_flip_scores_twice():
    """防止 RoundScoreCard 导出已翻转分数，load 后又按 ascending 再翻转一次。"""
    card, _, X, _ = _fit_simple_scorecard(
        direction="ascending",
        scorecard_cls=RoundScoreCard,
        decimal=1,
    )
    sample = X.iloc[[0, 30, 60, 90]].copy()
    rules = card.export(decimal=12)

    restored = RoundScoreCard().load_rules(rules)

    assert restored.decimal == 1
    np.testing.assert_allclose(
        restored.predict(sample, input_type="raw"),
        card.predict(sample, input_type="raw"),
        atol=1e-9,
    )


def test_scorecard_save_and_class_load_remain_a_pickle_roundtrip(tmp_path):
    """防止实例规则 load 覆盖父类类方法，使 ``ScoreCard.load(path)`` 失效。"""
    card, _, X, _ = _fit_simple_scorecard()
    destination = tmp_path / "scorecard.joblib"
    card.save(str(destination))

    restored = ScoreCard.load(str(destination))

    assert isinstance(restored, ScoreCard)
    np.testing.assert_allclose(
        restored.predict(X, input_type="raw"),
        card.predict(X, input_type="raw"),
        atol=1e-9,
    )


def test_instance_load_still_loads_rules_for_backward_compatibility():
    """在新增明确的 load_rules 后保留历史 ``ScoreCard().load(rules)``。"""
    card, _, X, _ = _fit_simple_scorecard()

    restored = ScoreCard().load(card.export(decimal=12))

    np.testing.assert_allclose(
        restored.predict(X, input_type="raw"),
        card.predict(X, input_type="raw"),
        atol=1e-9,
    )


def test_replacing_rules_on_a_fitted_card_does_not_keep_stale_model_components():
    """防止 load_rules 后仍走旧 LR/binner，导致新规则只导出却不参与实际评分。"""
    card, _, _, _ = _fit_simple_scorecard()
    replacement = {
        "__meta__": {
            "intercept_score": 600.0,
            "base_score": 750.0,
            "direction": "descending",
            "pdo": 60.0,
            "rate": 2.0,
            "base_odds": 35.0,
            "feature_names": ["x"],
            "coef": [1.0],
        },
        "x": {"[-inf, 0)": 10.0, "[0, +inf)": 20.0},
    }

    card.load_rules(replacement)

    sample = pd.DataFrame({"x": [-1.0, 1.0]})
    np.testing.assert_allclose(card.predict(sample, input_type="raw"), np.array([610.0, 620.0]))


def test_load_rules_with_explicit_binner_scores_loaded_rules_not_linear_model_path():
    """防止显式 binner 让离线规则误走 WOE 系数路径。"""
    card, binner, X, _ = _fit_simple_scorecard(direction="ascending")
    rules = card.export(decimal=12)
    first_bin = next(iter(rules["x"]))
    rules["x"][first_bin] += 100.0
    expected = ScoreCard().load_rules(copy.deepcopy(rules)).predict(X, input_type="raw")

    restored = ScoreCard().load_rules(rules, binner=binner)

    np.testing.assert_allclose(restored.predict(X, input_type="raw"), expected, atol=1e-9)


def test_invalid_python_or_java_function_name_is_rejected():
    """防止调用方提供的函数名直接拼入代码造成语法错误或代码注入。"""
    card = ScoreCard().load_rules(
        {
            "__meta__": {"intercept_score": 600.0, "feature_names": ["x"], "coef": [1.0]},
            "x": {"[-inf, +inf)": 10.0},
        }
    )

    for language in ("python", "java"):
        with pytest.raises(ValueError, match="函数名"):
            card.export_deployment_code(language=language, function_name="bad-name();")


def test_generated_java_is_self_contained_and_handles_colliding_feature_names(tmp_path):
    """防止 Java 缺少 Map import，或清洗后的特征变量名发生冲突。"""
    javac = shutil.which("javac")
    if javac is None:
        pytest.skip("当前环境没有 javac")
    card = ScoreCard().load_rules(
        {
            "__meta__": {
                "intercept_score": 600.0,
                "feature_names": ["a-b", "a_b"],
                "coef": [1.0, 1.0],
            },
            "a-b": {"[-inf, +inf)": 10.0},
            "a_b": {"[-inf, +inf)": 20.0},
        }
    )
    java_file = tmp_path / "ScoreCard.java"
    java_file.write_text(card.export_deployment_code(language="java"), encoding="utf-8")

    result = subprocess.run(
        [javac, "-encoding", "UTF-8", str(java_file)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_pmml_preserves_numeric_category_types_and_unknown_default(tmp_path):
    """防止 PMML lookup 把数值类别键字符串化，并确认未知类别贡献为 0。"""
    pytest.importorskip("sklearn2pmml")
    pytest.importorskip("sklearn_pandas")
    pypmml = pytest.importorskip("pypmml")
    X = pd.DataFrame({"code": [1] * 60 + [2] * 60 + [3] * 60})
    y = pd.Series(([0] * 50 + [1] * 10) + ([0] * 30 + [1] * 30) + ([0] * 10 + [1] * 50))
    binner = OptimalBinning(
        method="target_bad_rate",
        cat_cutoff=10,
        user_splits={"code": [[1], [2], [3]]},
        user_splits_fixed=True,
        min_n_bins=3,
        max_n_bins=3,
        n_jobs=1,
    ).fit(X, y)
    card = ScoreCard(binner=binner).fit(binner.transform(X, metric="woe"), y, input_type="woe")
    sample = pd.DataFrame({"code": [1, 2, 3, 4]})
    destination = tmp_path / "numeric-categories.pmml"

    card.export_pmml(str(destination))
    pmml_scores = pypmml.Model.load(str(destination)).predict(sample)["predicted_score"].to_numpy()

    np.testing.assert_allclose(pmml_scores, card.predict(sample, input_type="raw"), atol=1e-9)


def test_ascending_pmml_uses_the_same_single_direction_transform(tmp_path):
    """防止 PMML 与 ascending 本地评分使用不同的翻转次数。"""
    pytest.importorskip("sklearn2pmml")
    pytest.importorskip("sklearn_pandas")
    pypmml = pytest.importorskip("pypmml")
    card, _, X, _ = _fit_simple_scorecard(direction="ascending")
    sample = X.iloc[[0, 30, 60, 90]].copy()
    destination = tmp_path / "ascending.pmml"

    card.export_pmml(str(destination))
    pmml_scores = pypmml.Model.load(str(destination)).predict(sample)["predicted_score"].to_numpy()

    np.testing.assert_allclose(pmml_scores, card.predict(sample, input_type="raw"), atol=1e-9)
