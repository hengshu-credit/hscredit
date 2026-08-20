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

    def __init__(self, model, reference_data, constraints=None, random_state=42):
        if not isinstance(reference_data, pd.DataFrame) or reference_data.empty:
            raise ValidationError("reference_data 必须是非空 DataFrame")
        self.model = model
        self.reference_data = reference_data.copy()
        self.constraints = self._normalize_constraints(constraints or {})
        self.random_state = random_state
        self._candidates = self._build_candidates()

    def _normalize_constraints(self, constraints):
        unknown = set(constraints) - set(self.reference_data.columns)
        if unknown:
            raise ValidationError(f"约束包含未知特征: {sorted(unknown)}")
        normalized = {}
        for feature in self.reference_data.columns:
            config = dict(constraints.get(feature, {}))
            direction = config.get("direction", "both")
            if direction not in {"both", "increase_only", "decrease_only"}:
                raise ValidationError(f"{feature} 的 direction 约束无效")
            config.setdefault("mutable", True)
            config.setdefault("weight", 1.0)
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
                candidates[feature] = [value for value in observed if allowed is None or value in allowed]
        return candidates

    def _is_scorecard(self):
        return "scorecard" in self.model.__class__.__name__.lower() and hasattr(self.model, "predict_score")

    def _predict(self, frame):
        if self._is_scorecard():
            return float(np.asarray(self.model.predict_score(frame)).reshape(-1)[0])
        probabilities = np.asarray(self.model.predict_proba(frame))
        classes = list(getattr(self.model, "classes_", [0, 1]))
        index = classes.index(1) if 1 in classes else len(classes) - 1
        return float(probabilities[0, index])

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
                scale = float(series.max() - series.min()) or 1.0
                total += weight * abs(float(new) - float(old)) / scale
            else:
                total += weight
        return total

    def _search(self, subject, target, max_changes, top_n, beam_width):
        original_frame = subject.to_frame().T[self.reference_data.columns]
        before = self._predict(original_frame)
        reached = (lambda value: value >= target) if self._is_scorecard() else (lambda value: value <= target)
        if reached(before):
            return before, [({}, before, 0.0)]
        states = [({}, before, 0.0)]
        solutions = []
        for _depth in range(1, max_changes + 1):
            expanded = {}
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
                        if key in expanded:
                            continue
                        candidate = original_frame.copy()
                        for name, item in candidate_changes.items():
                            candidate.iloc[0, candidate.columns.get_loc(name)] = item
                        prediction = self._predict(candidate)
                        cost = self._cost(subject, candidate_changes)
                        expanded[key] = (candidate_changes, prediction, cost)
                        if reached(prediction):
                            solutions.append((candidate_changes, prediction, cost))
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
        is_scorecard = self._is_scorecard()
        target = target_score if is_scorecard else target_probability
        if target is None:
            required = "target_score" if is_scorecard else "target_probability"
            raise ValidationError(f"必须提供 {required}")
        if max_changes <= 0 or top_n <= 0 or beam_width <= 0:
            raise ValidationError("max_changes、top_n 和 beam_width 必须是正整数")
        rows = []
        for sample_id, subject in X.iterrows():
            before, solutions = self._search(subject, float(target), max_changes, top_n, beam_width)
            if not solutions:
                immutable = all(not config["mutable"] for config in self.constraints.values())
                rows.append({"样本索引": sample_id, "方案编号": 0, "是否达标": "否", "变更特征数": 0, "总成本": np.nan, "预测前值": before, "预测后值": before, "目标值": target, "约束检查": "通过", "失败原因": "所有特征均不可变" if immutable else "约束和搜索范围内未找到可行方案", "说明": "模型条件下的非因果建议，不代表真实因果效果或授信承诺"})
                continue
            for plan_number, (changes, after, cost) in enumerate(solutions, 1):
                items = changes.items() or [(None, None)]
                for feature, new in items:
                    old = subject[feature] if feature is not None else np.nan
                    direction = "不变" if feature is None else "增加" if new > old else "减少" if new < old else "替换"
                    rows.append({"样本索引": sample_id, "方案编号": plan_number, "是否达标": "是", "变更特征数": len(changes), "总成本": cost, "预测前值": before, "预测后值": after, "目标值": target, "特征": feature, "原值": old, "新值": new, "变化方向": direction, "约束检查": "通过", "失败原因": None, "说明": "模型条件下的非因果建议，不代表真实因果效果或授信承诺"})
        return pd.DataFrame(rows).reindex(columns=COUNTERFACTUAL_COLUMNS)
