"""具体模型公共名称与硬删除契约测试。"""

import importlib

import hscredit
from hscredit.core import models
from hscredit.core.models.classical.sklearn_models import SklearnRiskModel


EXPECTED_MODELS = {
    "XGBoost": "hscredit.core.models.boosting.xgboost_model",
    "LightGBM": "hscredit.core.models.boosting.lightgbm_model",
    "CatBoost": "hscredit.core.models.boosting.catboost_model",
    "NGBoost": "hscredit.core.models.boosting.ngboost_model",
    "RandomForest": "hscredit.core.models.classical.sklearn_models",
    "ExtraTrees": "hscredit.core.models.classical.sklearn_models",
    "GradientBoosting": "hscredit.core.models.classical.sklearn_models",
    "SVM": "hscredit.core.models.classical.sklearn_models",
    "DecisionTreeClassifier": "hscredit.core.models.classical.sklearn_models",
}

REMOVED_MODELS = {
    "XGBoostRiskModel": "hscredit.core.models.boosting.xgboost_model",
    "LightGBMRiskModel": "hscredit.core.models.boosting.lightgbm_model",
    "CatBoostRiskModel": "hscredit.core.models.boosting.catboost_model",
    "NGBoostRiskModel": "hscredit.core.models.boosting.ngboost_model",
    "RandomForestRiskModel": "hscredit.core.models.classical.sklearn_models",
    "ExtraTreesRiskModel": "hscredit.core.models.classical.sklearn_models",
    "GradientBoostingRiskModel": "hscredit.core.models.classical.sklearn_models",
}


def test_new_model_names_are_real_public_classes():
    """新名称若只是别名，类身份和新制品序列化路径仍会泄漏旧名称。"""
    for name, module_name in EXPECTED_MODELS.items():
        model_class = getattr(models, name)

        assert model_class.__name__ == name
        assert model_class.__module__ == module_name
        assert getattr(hscredit, name) is model_class


def test_old_concrete_model_names_are_removed_from_every_definition_module():
    """旧定义若残留，旧 pickle 和导入仍会被意外兼容。"""
    for old_name, module_name in REMOVED_MODELS.items():
        definition_module = importlib.import_module(module_name)

        assert not hasattr(hscredit, old_name)
        assert not hasattr(models, old_name)
        assert not hasattr(definition_module, old_name)


def test_base_model_layer_names_remain_unchanged():
    """具体模型重命名不能波及统一抽象层和 sklearn 适配层。"""
    assert models.BaseRiskModel.__name__ == "BaseRiskModel"
    assert SklearnRiskModel.__name__ == "SklearnRiskModel"
