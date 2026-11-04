"""轻量入口、公共符号注册及依赖版本自检。"""

import json
import os
import subprocess
import sys


def test_light_import_preserves_metrics_and_pandas_extensions_without_plotting():
    source = """
import sys, json, pandas as pd
import hscredit as h
assert hasattr(pd.DataFrame, 'summary') and hasattr(pd.DataFrame, 'hscredit')
assert h.auc([0, 1], [.1, .9]) == 1
assert not any(name in sys.modules for name in ['matplotlib.pyplot', 'shap', 'ortools', 'torch'])
print(json.dumps(sorted(h.__all__)))
"""
    environment = dict(os.environ, HSCREDIT_LIGHT_IMPORT="1")
    output = subprocess.check_output(
        [sys.executable, "-X", "utf8", "-c", source], env=environment, text=True, encoding="utf-8"
    )
    import hscredit

    assert json.loads(output) == sorted(hscredit.__all__)


def test_environment_reports_loaded_version_mismatch(monkeypatch):
    import numpy
    from hscredit.utils.environment import check_environment, runtime_version

    monkeypatch.setattr(numpy, "__version__", "999.0")
    assert runtime_version("numpy") == "999.0"
    report = check_environment(["numpy>=1.23"])
    assert not report["正常"]
    assert report["依赖"][0]["状态"] == "导入版本与安装元数据不一致"


def test_light_models_and_quantiles_load_plotting_only_on_demand():
    code = """
import hscredit as h, pandas as pd, numpy as np, sys
X = pd.DataFrame({'x': np.arange(40)})
y = np.arange(40) % 2
model = h.LogisticRegression(calculate_stats=False).fit(X, y)
assert model.predict_proba(X).shape == (40, 2)
h.QuantileBinning(n_jobs=1).fit(X, y)
h.OptimalBinning(method='quantile', n_jobs=1).fit(X, y)
assert not any(name in sys.modules for name in ['matplotlib.pyplot', 'ortools', 'shap', 'torch'])
"""
    subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        check=True,
        env=dict(os.environ, HSCREDIT_LIGHT_IMPORT="1"),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_light_registry_resolves_the_same_public_objects():
    import importlib
    from hscredit._public_api import EXPORTS

    for package_name in EXPORTS:
        package = importlib.import_module(package_name)
        for name, module_name in EXPORTS[package_name].items():
            if name in package.__all__:
                assert getattr(importlib.import_module(module_name), name) is getattr(package, name), (
                    package_name,
                    name,
                )
