"""模型局部贡献到业务原因码的规范化转换。"""

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from hscredit.exceptions import ValidationError


REASON_CODE_COLUMNS = [
    "样本索引", "原因排名", "特征", "业务特征名", "特征值", "SHAP值", "风险贡献",
    "原因码", "原因描述", "原因状态", "目标类别", "输出尺度", "风险方向",
]


def build_reason_codes(
    result,
    *,
    keep: int = 3,
    risk_direction: str = "higher_output_higher_risk",
    feature_map: Optional[Mapping[str, str]] = None,
    reason_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """仅将风险方向一致的不利 SHAP 贡献转换为原因码。"""
    if risk_direction not in {"higher_output_higher_risk", "higher_output_lower_risk"}:
        raise ValidationError("risk_direction 必须是 higher_output_higher_risk 或 higher_output_lower_risk")
    if not isinstance(keep, int) or keep <= 0:
        raise ValidationError("keep 必须是正整数")
    sign = 1.0 if risk_direction == "higher_output_higher_risk" else -1.0
    feature_map = dict(feature_map or {})
    reason_map = dict(reason_map or {})
    rows = []
    for position, sample_id in enumerate(result.sample_ids):
        adverse = sign * result.values[position]
        order = [index for index in np.argsort(adverse, kind="stable")[::-1] if adverse[index] > 0][:keep]
        if not order:
            rows.append(
                {"样本索引": sample_id, "原因状态": "无不利贡献", "目标类别": result.target_class,
                 "输出尺度": result.model_output, "风险方向": risk_direction}
            )
            continue
        for rank, index in enumerate(order, 1):
            feature = result.feature_names[index]
            mapping = reason_map.get(feature, {})
            rows.append(
                {
                    "样本索引": sample_id,
                    "原因排名": rank,
                    "特征": feature,
                    "业务特征名": feature_map.get(feature, feature),
                    "特征值": result.data.iloc[position, index],
                    "SHAP值": result.values[position, index],
                    "风险贡献": adverse[index],
                    "原因码": mapping.get("code", f"MODEL_{index + 1:03d}"),
                    "原因描述": mapping.get("description", f"{feature_map.get(feature, feature)}对风险输出产生不利影响"),
                    "原因状态": "存在不利贡献",
                    "目标类别": result.target_class,
                    "输出尺度": result.model_output,
                    "风险方向": risk_direction,
                }
            )
    return pd.DataFrame(rows).reindex(columns=REASON_CODE_COLUMNS)
