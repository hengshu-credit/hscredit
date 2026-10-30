"""模型局部贡献到业务原因码的规范化转换。"""

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from hscredit.exceptions import ValidationError

REASON_CODE_COLUMNS = [
    "样本索引",
    "原因排名",
    "特征",
    "业务特征名",
    "特征值",
    "SHAP值",
    "风险贡献",
    "原因码",
    "原因描述",
    "原因状态",
    "目标类别",
    "输出尺度",
    "风险方向",
]


def build_reason_codes(
    result,
    *,
    keep: int = 3,
    risk_direction: str = "higher_output_higher_risk",
    feature_map: Optional[Mapping[str, str]] = None,
    reason_map: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """仅将风险方向一致的不利 SHAP 贡献转换为原因码。

    :param result: 已计算的 ExplanationResult，包含样本、贡献值及输出尺度。
    :param risk_direction: ``'higher_output_higher_risk'`` 表示输出越高风险越大；``'higher_output_lower_risk'`` 表示越低风险越大。 默认 ``'higher_output_higher_risk'``。
    :param keep: 每个样本最多保留的不利贡献原因数，必须为正整数。 默认 ``3``。
    :param feature_map: 原始字段名到业务名称的映射；None 保留原字段名。 默认 ``None``。
    :param reason_map: 字段名到原因配置的映射；每项支持 ``code`` 和 ``description``，None 使用默认原因码与描述。
    :return: 按样本及不利贡献排序的中文原因码表；无不利贡献的样本保留状态行。
    """
    if risk_direction not in {"higher_output_higher_risk", "higher_output_lower_risk"}:
        raise ValidationError("risk_direction 必须是 higher_output_higher_risk 或 higher_output_lower_risk")
    if not isinstance(keep, int) or keep <= 0:
        raise ValidationError("keep 必须是正整数")
    sign = 1.0 if risk_direction == "higher_output_higher_risk" else -1.0
    feature_map = dict(feature_map or {})
    reason_map = dict(reason_map or {})
    values = result.values
    data = result.data
    sample_ids = result.sample_ids
    rows = []
    for position, sample_id in enumerate(sample_ids):
        adverse = sign * values[position]
        order = [index for index in np.argsort(-adverse, kind="stable") if adverse[index] > 0][:keep]
        if not order:
            rows.append(
                {
                    "样本索引": sample_id,
                    "原因状态": "无不利贡献",
                    "目标类别": result.target_class,
                    "输出尺度": result.model_output,
                    "风险方向": risk_direction,
                }
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
                    "特征值": data.iloc[position, index],
                    "SHAP值": values[position, index],
                    "风险贡献": adverse[index],
                    "原因码": mapping.get("code", f"MODEL_{index + 1:03d}"),
                    "原因描述": mapping.get(
                        "description", f"{feature_map.get(feature, feature)}对风险输出产生不利影响"
                    ),
                    "原因状态": "存在不利贡献",
                    "目标类别": result.target_class,
                    "输出尺度": result.model_output,
                    "风险方向": risk_direction,
                }
            )
    return pd.DataFrame(rows).reindex(columns=REASON_CODE_COLUMNS)
