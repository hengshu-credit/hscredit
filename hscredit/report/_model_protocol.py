"""模型报告的标签名称、标签口径及推理入口。"""

import re

import numpy as np
import pandas as pd

from ..utils.input_utils import normalize_dpd_values


def display_label(name):
    """阈值式标签统一为字段@阈值；只更改名称，不从名称推断标签计算规则。"""
    match = re.fullmatch(
        r"(.+?)\s*(?:大于等于|小于等于|大于|小于|>=|<=|>|<)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        str(name),
    )
    if match:
        field, threshold = match.groups()
        return f"{field.strip()}@{normalize_dpd_values([float(threshold)])[0]}"
    return str(name)


def final_estimator(model):
    while getattr(model, "steps", None):
        model = model.steps[-1][1]
    return model


def is_score_transformer(model):
    from ..core.models.scorecard.score_transformer import BaseScoreTransformer

    model = final_estimator(model)
    # ScoreCard 继承标准转换器，但仍是可输出概率的分类模型。
    return isinstance(model, BaseScoreTransformer) and not callable(getattr(model, "predict_proba", None))


def default_prediction_method(model, task_type):
    if is_score_transformer(model):
        return "transform"
    return "predict" if task_type == "regression" else "predict_proba"


def label_definition_table(report):
    """逐数据集说明实际标签来源；显式 y 不冒充按逾期配置生成的标签。"""
    columns, thresholds = report._overdue_dpds()
    pairs = {f"{column}@{threshold}": (column, threshold) for column in columns for threshold in thresholds}
    rows = []
    for key, dataset in report._datasets.items():
        source = getattr(report, "_dataset_target_sources", {}).get(key, "overdue" if pairs else "target")
        labels = report._label_names or [report._target_name]
        for label in labels:
            values = np.asarray(report._get_y(key, label if report._label_names else None), dtype=float)
            explicit = source == "explicit_y"
            operator = report.overdue_operator if source == "overdue" else "不适用（直接标签）"
            if source == "overdue":
                column, threshold = pairs[label]
                bad = f"{column} {operator} {threshold}"
                if operator == ">" and threshold > 0:
                    grey = f"0 < {column} <= {threshold}"
                elif operator == ">=" and threshold > 0:
                    grey = f"0 < {column} < {threshold}"
                else:
                    grey = "无灰样本区间"
                rule = f"按本标签独立剔除满足 {grey} 的样本" if report.del_grey and grey != "无灰样本区间" else "不剔除灰样本"
                missing = ""
            elif explicit:
                bad, grey = "外部 y=1", "标签不是0或1（不含缺失）"
                rule = "剔除灰样本" if report.del_grey else "灰样本按0处理"
                missing = ""
            else:
                bad, grey = "传入标签=1", "标签不是0或1（不含缺失）"
                rule = "按本标签独立剔除灰样本" if report.del_grey else "灰样本按0处理"
                missing = ""
            missing = {"drop": "缺失标签剔除", "positive": "缺失标签按1处理", "negative": "缺失标签按0处理", "raise": "缺失标签报错"}[
                report.handle_missing_target
            ]
            rows.append(
                {
                    "数据集": dataset.label,
                    "标签": display_label(label),
                    "标签来源": "显式y" if explicit else "overdue+dpds" if source == "overdue" else "target",
                    "操作符": operator,
                    "坏样本定义": bad,
                    "剔灰配置": "是" if report.del_grey else "否",
                    "灰样本定义": grey,
                    "实际剔除规则": rule,
                    "缺失值处理": missing,
                    "原始样本数": len(values),
                    "有效样本数": int(np.isfinite(values).sum()),
                    "剔除样本数": int((~np.isfinite(values)).sum()),
                }
            )
    return pd.DataFrame(rows)


def label_configuration_table(report):
    """将标签口径合并为数据样本描述中的四项配置，同一口径不重复列数据集。"""
    definitions = label_definition_table(report)
    rows = []
    for field in ("标签来源", "操作符", "灰样本定义", "剔灰配置", "缺失值处理"):
        groups = {}
        for dataset, block in definitions.groupby("数据集", sort=False):
            values = list(dict.fromkeys(block[field].astype(str)))
            if field == "灰样本定义" and len(values) > 1:
                pairs = block[["标签", field]].drop_duplicates()
                value = "；".join(f"{label}：{rule}" for label, rule in pairs.itertuples(index=False, name=None))
            else:
                value = "；".join(values)
            groups.setdefault(value, []).append(dataset)
        content = (
            next(iter(groups))
            if len(groups) == 1
            else "；".join(f"{'、'.join(datasets)}：{value}" for value, datasets in groups.items())
        )
        rows.append({"配置项": field, "配置值": content})
    method = report.prediction_method
    rows.extend(
        [
            {
                "配置项": "预测函数",
                "配置值": ("predict_proba" if method == "predict_prob" else method)
                if isinstance(method, str)
                else getattr(method, "__name__", "自定义函数"),
            },
            {"配置项": "分箱基准", "配置值": "训练样本" if report.binning_reference == "train" else "各数据集独立"},
            {"配置项": "统计口径", "配置值": "实际订单及实际金额"},
            {"配置项": "加权性能对照", "配置值": "是" if report.show_weighted_metrics else "否"},
        ]
    )
    if report.show_weighted_metrics:
        weights = report.sample_weight
        description = weights if isinstance(weights, str) else "按数据集提供的逐行权重" if isinstance(weights, dict) else "逐行权重数组"
        rows.append({"配置项": "加权性能权重", "配置值": description})
    return pd.DataFrame(rows)
