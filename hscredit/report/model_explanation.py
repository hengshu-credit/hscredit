"""模型报告中的结构化解释构建与序列化。"""

import warnings
from typing import Any, Dict

import pandas as pd

from ..core.models.evaluation import ModelExplainer

DEFAULT_EXPLAIN_CONFIG = {
    "enabled": False,
    "data": None,
    "background_data": None,
    "target_class": 1,
    "model_output": "probability",
    "algorithm": "auto",
    "max_samples": 500,
    "representative_count": 6,
    "stability_mode": "sample",
    "n_bootstrap": 100,
    "top_k": 10,
    "random_state": 42,
    "risk_direction": "higher_output_higher_risk",
    "on_explain_error": "raise",
    "X_train": None,
    "y_train": None,
    "X_validation": None,
}


def normalize_explain_config(config=None):
    """合并并校验显式解释配置。"""
    merged = dict(DEFAULT_EXPLAIN_CONFIG)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("explain_config 必须是字典或 None")
        unknown = set(config) - set(merged)
        if unknown:
            raise ValueError(f"explain_config 包含未知配置: {sorted(unknown)}")
        merged.update(config)
    if merged["on_explain_error"] not in {"raise", "warn"}:
        raise ValueError("on_explain_error 必须是 raise 或 warn")
    return merged


def build_model_explanation(model, config: Dict[str, Any]) -> Dict[str, Any]:
    """按已校验配置生成报告所需的全部解释表。"""
    try:
        explainer = ModelExplainer(
            model,
            background_data=config["background_data"],
            target_class=config["target_class"],
            model_output=config["model_output"],
            algorithm=config["algorithm"],
            random_state=config["random_state"],
        )
        result = explainer.explain(config["data"], max_samples=config["max_samples"])
        representatives = explainer.select_representative_samples(result).head(config["representative_count"])
        return {
            "元信息": {**dict(result.metadata), "背景数据": dict(result.background_summary), "数据指纹": result.dataset_fingerprint},
            "全局解释": explainer.get_global_report(result),
            "相关性": explainer.get_correlation_report(result, kind="shap_shap"),
            "特征聚类": explainer.get_feature_clusters(result),
            "交互": explainer.get_feature_interactions(result=result),
            "稳定性": explainer.get_stability_report(
                result,
                mode=config["stability_mode"],
                X_train=config["X_train"],
                y_train=config["y_train"],
                X_validation=config["X_validation"],
                n_bootstrap=config["n_bootstrap"],
                top_k=config["top_k"],
                random_state=config["random_state"],
            ),
            "代表样本": representatives,
            "样本解释": {sample_id: explainer.get_sample_report(result, sample_id=sample_id) for sample_id in representatives["样本索引"]},
            "原因码": explainer.get_reason_codes(result, risk_direction=config["risk_direction"]),
            "解释结果": result,
            "解释器": explainer,
        }
    except Exception as exc:
        if config["on_explain_error"] == "warn":
            warnings.warn(f"生成模型解释失败: {exc}", RuntimeWarning, stacklevel=2)
            return {"失败原因": str(exc)}
        raise RuntimeError(f"生成模型解释失败: {exc}") from exc


def explanation_to_dict(explanation: Dict[str, Any]) -> Dict[str, Any]:
    """把解释节点转换为可序列化字典。"""
    serialized = {}
    for key, value in explanation.items():
        if key in {"解释结果", "解释器"}:
            continue
        if isinstance(value, pd.DataFrame):
            serialized[key] = value.to_dict(orient="records")
        elif isinstance(value, dict) and key == "样本解释":
            serialized[key] = {str(sample): table.to_dict(orient="records") for sample, table in value.items()}
        else:
            serialized[key] = value
    return serialized
