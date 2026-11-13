"""验证参数目录、页面任务绑定和底层估计器实际使用的配置。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning import OptimalBinning
from hscredit.web.analysis import model_fit_options, model_pipeline
from hscredit.web.models import model_catalog
from hscredit.web.parameters import binning_params, parameter_catalog, validate_library_params
from hscredit.web.server import JobRequest, validate_job


def test_method_specific_kwargs_and_2d_options():
    catalog = parameter_catalog()
    assert catalog["methods"] == OptimalBinning.VALID_METHODS
    help_text = catalog["groups"]["binning"]["params"]["min_bad_rate"]["description"]
    assert "不参与该约束" in help_text
    assert ":type" not in help_text
    cfg = JobRequest(kind="strategy", dataset_id="demo", workflow=True, strategy_mode="pair", min_n_bins=2, method="tree", library_params={"binning": {"max_depth": 1}, "pair_binning": {"max_n_bins_y": 3}}).dict()
    validate_library_params(cfg)
    binner = OptimalBinning(**binning_params(cfg), n_jobs=1)
    x = pd.DataFrame({"字段": np.arange(120)})
    binner.fit(x, (x["字段"] >= 60).astype(int))
    assert len(binner.splits_["字段"]) == 1
    cfg["method"] = "quantile"
    with pytest.raises(ValueError, match="max_depth"):
        validate_library_params(cfg)


def test_native_forest_settings_survive_clone_and_align_fit_weights():
    catalog = model_catalog()["random_forest"]
    assert {"max_leaf_nodes", "class_weight"} <= catalog["extra_params"].keys()
    assert "sample_weight" in catalog["fit_params"]
    data = pd.DataFrame({"x": np.arange(120), "标签": np.arange(120) % 2, "权重": np.linspace(1, 5, 120)})
    cfg = JobRequest(kind="model", dataset_id="demo", target="标签", features=["x"], models=["random_forest"], model_params={"random_forest": {"n_estimators": 3, "max_leaf_nodes": 2, "bootstrap": False, "class_weight": {"0": 1, "1": 2}}}, model_fit_params={"random_forest": {"sample_weight": "权重"}}).dict()
    validate_job(data, cfg)
    rows = data.index[3::3]
    pipe = clone(model_pipeline("random_forest", cfg["model_params"]["random_forest"], ["x"], [], 42))
    pipe.fit(data.loc[rows, ["x"]], data.loc[rows, "标签"], **model_fit_options(cfg["model_fit_params"]["random_forest"], data, rows))
    trees = pipe.named_steps["model"]._model.estimators_
    assert all(tree.tree_.n_leaves == 2 for tree in trees)
    effective_weights = data.loc[rows, "权重"] * (1 + data.loc[rows, "标签"])
    assert trees[0].tree_.weighted_n_node_samples[0] == pytest.approx(effective_weights.sum())
    cfg["model_fit_params"]["random_forest"] = {"misspelled": 0}
    with pytest.raises(ValueError, match="misspelled"):
        validate_job(data, cfg)


@pytest.mark.parametrize("scope,key", [("feature_report", "save"), ("excel_writer", "style_excel"), ("other", "top_n")])
def test_report_rejects_unbound_parameters(scope, key):
    with pytest.raises(ValueError, match=key if scope != "other" else scope):
        validate_library_params({"kind": "third_party", "library_params": {scope: {key: "x"}}})


def test_summary_parameters_change_cache_and_exported_percentiles(tmp_path):
    from hscredit.web.storage import Workspace
    from hscredit.web.exploration import explore_dataset

    workspace = Workspace(tmp_path)
    data = pd.DataFrame({"分数": np.arange(100)})
    source = workspace.add_dataset(data, "样本")["id"]
    first = explore_dataset(workspace, source, library_params={"summary": {"percentiles": [0.37]}})
    second = explore_dataset(workspace, source, library_params={"summary": {"percentiles": [0.61]}})
    assert first["summary"][0]["37%"] == pytest.approx(data["分数"].quantile(0.37))
    assert second["summary"][0]["61%"] == pytest.approx(data["分数"].quantile(0.61))
    assert "37%" not in second["summary"][0]


def test_sampler_custom_kwargs_use_selected_backend():
    cfg = {"kind": "model", "library_params": {"tuner": {"sampler": "tpe"}, "sampler": {"n_startup_trials": 5}}}
    validate_library_params(cfg)
    cfg["library_params"]["tuner"]["sampler"] = "random"
    with pytest.raises(ValueError, match="n_startup_trials"):
        validate_library_params(cfg)
