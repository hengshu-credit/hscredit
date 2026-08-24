"""模型可解释性、原因码与反事实分析子包。"""

from .counterfactual import CounterfactualExplainer
from .explainer import ModelExplainer
from .plots import (
    plot_correlation,
    plot_decision,
    plot_distribution,
    plot_explanation_overview,
    plot_feature_clustering,
    plot_heatmap,
    plot_importance_overview,
    plot_interaction_bubble,
    plot_interaction_heatmap,
    plot_feature_importance,
    plot_importance_comparison,
    plot_shap_importance,
)
from .reason_codes import build_reason_codes
from .reports import model_explain_report
from .result import ExplanationResult

__all__ = [
    "ModelExplainer",
    "ExplanationResult",
    "CounterfactualExplainer",
    "build_reason_codes",
    "model_explain_report",
    "plot_feature_importance",
    "plot_shap_importance",
    "plot_importance_comparison",
    "plot_decision",
    "plot_heatmap",
    "plot_distribution",
    "plot_correlation",
    "plot_feature_clustering",
    "plot_interaction_heatmap",
    "plot_interaction_bubble",
    "plot_importance_overview",
    "plot_explanation_overview",
]
