"""不依赖额外解释库的受约束反事实建议。"""

import numpy as np
import pandas as pd

from hscredit.exceptions import ValidationError

COUNTERFACTUAL_COLUMNS = [
    "样本索引",
    "方案编号",
    "是否达标",
    "变更特征数",
    "总成本",
    "预测前值",
    "预测后值",
    "目标值",
    "特征",
    "原值",
    "新值",
    "变化方向",
    "约束检查",
    "失败原因",
    "说明",
]


class CounterfactualExplainer:
    """在模型与显式约束下搜索最小特征变更方案。

    输出仅表示模型条件下的非因果建议，不代表真实因果效果或授信承诺。
    """

    def __init__(
        self,
        model,
        reference_data,
        constraints=None,
        output_type="auto",
        positive_class=1,
        max_candidates=100,
    ):
        """创建受约束反事实搜索器。

        :param model: 已拟合模型；概率模式要求 ``predict_proba``，评分模式要求 ``predict_score``。
        :param reference_data: 用于候选值和数值尺度的非空 DataFrame。
        :param constraints: 各特征的 mutable、weight、direction、min、max 和 allowed 约束。
        :param output_type: ``auto``、``probability`` 或 ``score``。auto 优先使用概率接口。
        :param positive_class: 概率模式下的风险正类标签。
        :param max_candidates: 每个类别特征最多保留的确定性候选数。
        """
        if not isinstance(reference_data, pd.DataFrame) or reference_data.empty:
            raise ValidationError("reference_data 必须是非空 DataFrame")
        if output_type not in {"auto", "probability", "score"}:
            raise ValidationError("output_type 必须是 auto、probability 或 score")
        if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates <= 0:
            raise ValidationError("max_candidates 必须是正整数")
        self.model = model
        self.reference_data = reference_data.copy()
        self.positive_class = positive_class
        self.output_type = self._resolve_output_type(output_type)
        self.max_candidates = max_candidates
        self.constraints = self._normalize_constraints(constraints or {})
        self._candidates = self._build_candidates()

    def _resolve_output_type(self, output_type):
        """按显式配置或公开预测接口确定目标输出。"""
        if output_type == "auto":
            output_type = "probability" if hasattr(self.model, "predict_proba") else "score"
        required = "predict_proba" if output_type == "probability" else "predict_score"
        if not hasattr(self.model, required):
            raise ValidationError(f"{output_type} 模式要求模型实现 {required}")
        if output_type == "probability":
            classes = list(getattr(self.model, "classes_", [0, 1]))
            if self.positive_class not in classes:
                raise ValidationError(f"positive_class={self.positive_class!r} 不在模型类别 {classes!r} 中")
        return output_type

    def _normalize_constraints(self, constraints):
        unknown = set(constraints) - set(self.reference_data.columns)
        if unknown:
            raise ValidationError(f"约束包含未知特征: {sorted(unknown)}")
        normalized = {}
        for feature in self.reference_data.columns:
            config = dict(constraints.get(feature, {}))
            mutable = config.get("mutable", True)
            if not isinstance(mutable, bool):
                raise ValidationError(f"{feature} 的 mutable 必须是布尔值")
            weight = config.get("weight", 1.0)
            if isinstance(weight, bool) or not np.isscalar(weight) or not np.isfinite(weight) or float(weight) < 0:
                raise ValidationError(f"{feature} 的 weight 必须是有限非负数")
            direction = config.get("direction", "both")
            if direction not in {"both", "increase_only", "decrease_only"}:
                raise ValidationError(f"{feature} 的 direction 约束无效")
            if direction != "both" and not pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                raise ValidationError(f"{feature} 是类别字段，direction 只能是 both")
            for bound in ("min", "max"):
                if bound in config and (
                    isinstance(config[bound], bool)
                    or not np.isscalar(config[bound])
                    or not np.isfinite(config[bound])
                ):
                    raise ValidationError(f"{feature} 的 {bound} 必须是有限数")
            if "min" in config and "max" in config and config["min"] > config["max"]:
                raise ValidationError(f"{feature} 的 min 不能大于 max")
            config["mutable"] = mutable
            config["weight"] = float(weight)
            config["direction"] = direction
            normalized[feature] = config
        return normalized

    def _build_candidates(self):
        candidates = {}
        for feature in self.reference_data.columns:
            series = self.reference_data[feature].dropna()
            config = self.constraints[feature]
            if not config["mutable"]:
                candidates[feature] = []
            elif pd.api.types.is_numeric_dtype(series):
                values = series.quantile([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]).unique()
                lower = config.get("min", -np.inf)
                upper = config.get("max", np.inf)
                candidates[feature] = sorted(float(value) for value in values if lower <= value <= upper)
            else:
                allowed = config.get("allowed")
                observed = list(dict.fromkeys(series.tolist()))
                candidates[feature] = [value for value in observed if allowed is None or value in allowed][
                    : self.max_candidates
                ]
        return candidates

    def _uses_score_output(self):
        return self.output_type == "score"

    def _predict_many(self, frame):
        """批量返回当前输出尺度的一维预测。"""
        if self._uses_score_output():
            predictions = np.asarray(self.model.predict_score(frame), dtype=float).reshape(-1)
            if len(predictions) != len(frame):
                raise ValidationError("predict_score 返回数量与输入样本数不一致")
            return predictions
        probabilities = np.asarray(self.model.predict_proba(frame))
        if probabilities.ndim != 2 or probabilities.shape[0] != len(frame):
            raise ValidationError("predict_proba 必须返回与输入等行的二维概率")
        classes = list(getattr(self.model, "classes_", [0, 1]))
        index = classes.index(self.positive_class)
        if probabilities.shape[1] != len(classes):
            raise ValidationError("predict_proba 概率列数与 classes_ 不一致")
        return np.asarray(probabilities[:, index], dtype=float)

    def _predict(self, frame):
        """返回单行输入的标量预测。"""
        return float(self._predict_many(frame)[0])

    def _valid_value(self, feature, old, new):
        config = self.constraints[feature]
        if new == old:
            return False
        if config["direction"] == "increase_only" and new < old:
            return False
        if config["direction"] == "decrease_only" and new > old:
            return False
        if "min" in config and new < config["min"]:
            return False
        if "max" in config and new > config["max"]:
            return False
        if "allowed" in config and new not in config["allowed"]:
            return False
        return True

    def _cost(self, original, changed):
        total = 0.0
        for feature, new in changed.items():
            old = original[feature]
            weight = float(self.constraints[feature]["weight"])
            series = self.reference_data[feature]
            if pd.api.types.is_numeric_dtype(series):
                span = float(series.max() - series.min())
                scale = span if np.isfinite(span) and span > 0 else 1.0
                total += weight * abs(float(new) - float(old)) / scale
            else:
                total += weight
        return total

    def _search(self, subject, target, max_changes, top_n, beam_width):
        original_frame = subject.to_frame().T[self.reference_data.columns].copy()
        numeric_columns = [
            column
            for column in original_frame.columns
            if pd.api.types.is_numeric_dtype(self.reference_data[column])
        ]
        if numeric_columns:
            original_frame[numeric_columns] = original_frame[numeric_columns].astype(float)
        before = self._predict(original_frame)
        reached = (lambda value: value >= target) if self._uses_score_output() else (lambda value: value <= target)
        if reached(before):
            return before, [({}, before, 0.0)]
        states = [({}, before, 0.0)]
        solutions = []
        for _depth in range(1, max_changes + 1):
            expanded = {}
            pending = {}
            for changes, _prediction, _old_cost in states:
                for feature in self.reference_data.columns:
                    if feature in changes or not self.constraints[feature]["mutable"]:
                        continue
                    old = subject[feature]
                    for value in self._candidates[feature]:
                        if not self._valid_value(feature, old, value):
                            continue
                        candidate_changes = dict(changes)
                        candidate_changes[feature] = value
                        key = tuple(sorted((name, repr(item)) for name, item in candidate_changes.items()))
                        if key in pending:
                            continue
                        candidate = original_frame.copy()
                        for name, item in candidate_changes.items():
                            candidate.iloc[0, candidate.columns.get_loc(name)] = item
                        pending[key] = (candidate_changes, candidate)
            if not pending:
                states = []
                break
            candidate_frame = pd.concat(
                [candidate for _changes, candidate in pending.values()],
                ignore_index=True,
            )
            predictions = self._predict_many(candidate_frame)
            for (key, (candidate_changes, _candidate)), prediction in zip(pending.items(), predictions):
                cost = self._cost(subject, candidate_changes)
                expanded[key] = (candidate_changes, float(prediction), cost)
                if reached(prediction):
                    solutions.append((candidate_changes, float(prediction), cost))
            if solutions:
                break

            def objective_key(state):
                return abs(state[1] - target), state[2], repr(state[0])

            states = sorted(expanded.values(), key=objective_key)[:beam_width]
        solutions.sort(key=lambda state: (len(state[0]), state[2], abs(state[1] - target), repr(state[0])))
        return before, solutions[:top_n]

    def generate(
        self,
        X,
        *,
        target_probability=None,
        target_score=None,
        max_changes=3,
        top_n=5,
        beam_width=50,
    ) -> pd.DataFrame:
        """生成满足风险概率上限或评分下限的确定性候选方案。"""
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValidationError("X 必须是非空 DataFrame")
        if list(X.columns) != list(self.reference_data.columns):
            raise ValidationError("X 的特征名称或顺序与 reference_data 不一致")
        uses_score = self._uses_score_output()
        if target_probability is not None and target_score is not None:
            raise ValidationError("target_probability 与 target_score 不能同时提供")
        target = target_score if uses_score else target_probability
        if target is None:
            required = "target_score" if uses_score else "target_probability"
            raise ValidationError(f"必须提供 {required}")
        if isinstance(target, bool) or not np.isscalar(target) or not np.isfinite(target):
            required = "target_score" if uses_score else "target_probability"
            raise ValidationError(f"{required} 必须是有限数")
        target = float(target)
        if not uses_score and not 0 <= target <= 1:
            raise ValidationError("target_probability 必须在[0, 1]范围内")
        for name, value in (("max_changes", max_changes), ("top_n", top_n), ("beam_width", beam_width)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValidationError(f"{name} 必须是正整数")
        rows = []
        for sample_id, subject in X.iterrows():
            before, solutions = self._search(subject, target, max_changes, top_n, beam_width)
            if not solutions:
                immutable = all(not config["mutable"] for config in self.constraints.values())
                rows.append({"样本索引": sample_id, "方案编号": 0, "是否达标": "否", "变更特征数": 0, "总成本": np.nan, "预测前值": before, "预测后值": before, "目标值": target, "约束检查": "通过", "失败原因": "所有特征均不可变" if immutable else "约束和搜索范围内未找到可行方案", "说明": "模型条件下的非因果建议，不代表真实因果效果或授信承诺"})
                continue
            for plan_number, (changes, after, cost) in enumerate(solutions, 1):
                items = changes.items() or [(None, None)]
                for feature, new in items:
                    old = subject[feature] if feature is not None else np.nan
                    if feature is None:
                        direction = "不变"
                    elif (
                        pd.api.types.is_numeric_dtype(self.reference_data[feature])
                        and pd.notna(old)
                        and pd.notna(new)
                    ):
                        direction = "增加" if new > old else "减少" if new < old else "替换"
                    else:
                        direction = "替换"
                    rows.append({"样本索引": sample_id, "方案编号": plan_number, "是否达标": "是", "变更特征数": len(changes), "总成本": cost, "预测前值": before, "预测后值": after, "目标值": target, "特征": feature, "原值": old, "新值": new, "变化方向": direction, "约束检查": "通过", "失败原因": None, "说明": "模型条件下的非因果建议，不代表真实因果效果或授信承诺"})
        return pd.DataFrame(rows).reindex(columns=COUNTERFACTUAL_COLUMNS)
