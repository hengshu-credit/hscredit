"""公开建模链路的参数必须可在 help/IDE 中找到，示例不能只放在外部文章里。"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "fit",
    "tune",
    "evaluate",
    "report",
    "model_report",
    "get_best_model",
    "get_eval_results",
    "get_training_history",
    "get_learning_curve",
    "get_training_process",
    "get_metrics",
    "to_excel",
    "auto_model_report",
    "get_validation_predictions",
    "get_trial_result",
    "get_fold_models",
    "evaluate_trials",
    "evaluate_study_trials",
}
MODELS = {
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "NGBoost",
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "SVM",
    "DecisionTreeClassifier",
    "ScoreCard",
    "RoundScoreCard",
    "ProbabilityScoreCard",
    "ProbabilityCalibrator",
    "CalibratedModel",
    "RulesClassifier",
    "ModelTuner",
    "ModelReport",
}


def _entries():
    paths = [
        p
        for p in (ROOT / "hscredit/core/models").rglob("*.py")
        if "losses" not in p.parts and "explainability" not in p.parts and p.name != "_lifecycle.py"
    ]
    paths += [ROOT / "hscredit/report/model_report.py", ROOT / "hscredit/report/_model_context.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for owner, scope in [("", tree), *((n.name, n) for n in tree.body if isinstance(n, ast.ClassDef))]:
            if owner.startswith("_") or owner == "CatBoostKSMetric":
                continue
            for node in scope.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name in METHODS:
                    yield f"{path.name}:{owner}.{node.name}", node, ast.get_docstring(node), True
                elif node.name == "__init__" and owner in MODELS:
                    yield f"{path.name}:{owner}", node, (ast.get_docstring(scope) or "") + "\n" + (
                        ast.get_docstring(node) or ""
                    ), False


@pytest.mark.parametrize("entry", list(_entries()), ids=lambda entry: entry[0])
def test_chain_entry_has_parameters_and_examples(entry):
    title, node, doc, example_required = entry
    assert doc, f"{title} 的文档不可见，docstring 必须是函数/类的第一条语句"
    names = [
        a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs if a.arg not in {"self", "cls"}
    ]
    names += [a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None]
    missing = [name for name in names if not re.search(r":param\s+(?:[^:\n]+\s+)?" + re.escape(name) + ":", doc)]
    assert not missing, f"{title} 缺少参数说明：{missing}"
    if example_required:
        assert ">>>" in doc, f"{title} 缺少方法内部调用示例"
