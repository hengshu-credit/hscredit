"""不依赖 SHAP 计算的轻量模型重要性报告。"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from ..base import BaseRiskModel


def _as_1d_importance(values: Any) -> np.ndarray:
    """将不同模型暴露的特征重要性压平成一维数组."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr[0] if arr.shape[0] == 1 else np.mean(np.abs(arr), axis=0)
    return arr.ravel()


def _infer_feature_names(
    model: Any,
    X: Optional[Union[np.ndarray, pd.DataFrame]],
    n_features: int,
) -> List[str]:
    """从输入数据或模型属性推断特征名."""
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "_feature_names"):
        return list(model._feature_names)
    if hasattr(model, "feature_names"):
        names = getattr(model, "feature_names")
        if names is not None:
            return list(names)
    return [f"feature_{i}" for i in range(n_features)]


def model_explain_report(
    model: BaseRiskModel,
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    importance_type: str = "gain",
    top_n: Optional[int] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """生成模型特征解释报告.

    不依赖 SHAP，优先复用模型自身 ``get_feature_importances``，再依次回退到
    ``feature_importances_`` / ``coef_``，用于没有安装解释扩展包时的基础模型解释。

    :param model: 已训练模型
    :param X: 特征矩阵，可选，用于推断特征名
    :param importance_type: 模型重要性类型，默认 ``'gain'``
    :param top_n: 返回前 N 个特征，None 表示全部返回
    :param normalize: 是否增加归一化重要性列，默认 True
    :return: 模型解释报告 DataFrame，列名为中文

    **参考样例**

    >>> report = model_explain_report(model, X_test, importance_type='coef')
    >>> print(report[['特征名', '重要性', '排名']].head())
    """
    if top_n is not None and (
        not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0
    ):
        raise ValueError("top_n 必须是正整数或 None")
    source = "get_feature_importances"
    direction = None

    if hasattr(model, "get_feature_importances"):
        importances = model.get_feature_importances(importance_type)
        if isinstance(importances, pd.Series):
            feature_names = importances.index.astype(str).tolist()
            values = importances.to_numpy(dtype=float)
        else:
            values = _as_1d_importance(importances)
            feature_names = _infer_feature_names(model, X, len(values))
    elif hasattr(model, "feature_importances_"):
        source = "feature_importances_"
        values = _as_1d_importance(getattr(model, "feature_importances_"))
        feature_names = _infer_feature_names(model, X, len(values))
    elif hasattr(model, "coef_"):
        source = "coef_"
        raw_coef = np.asarray(getattr(model, "coef_"), dtype=float)
        coef = _as_1d_importance(raw_coef)
        if raw_coef.ndim == 1 or (raw_coef.ndim == 2 and raw_coef.shape[0] == 1):
            direction = np.sign(coef)
        values = np.abs(coef)
        feature_names = _infer_feature_names(model, X, len(values))
    else:
        raise ValueError("模型未提供可解释的特征重要性或系数")

    if len(feature_names) != len(values):
        feature_names = [f"feature_{i}" for i in range(len(values))]

    result = pd.DataFrame(
        {
            "特征名": feature_names,
            "重要性": values,
            "重要性类型": importance_type if source == "get_feature_importances" else source,
            "来源": source,
        }
    )
    if direction is not None and len(direction) == len(result):
        direction_map = {1.0: "正向", -1.0: "负向", 0.0: "无方向"}
        result["影响方向"] = [direction_map.get(float(v), "无方向") for v in direction]

    result["排名"] = result["重要性"].rank(method="first", ascending=False).astype("Int64")
    result = result.sort_values(["排名", "特征名"]).reset_index(drop=True)

    if normalize:
        total = float(np.nansum(np.abs(result["重要性"].to_numpy(dtype=float))))
        result["归一化重要性"] = result["重要性"] / total if total > 0 else np.zeros(len(result), dtype=float)

    if top_n is not None:
        result = result.head(top_n).reset_index(drop=True)

    return result
