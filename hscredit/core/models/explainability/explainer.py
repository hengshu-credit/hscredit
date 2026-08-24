"""统一 SHAP 模型解释器与结构化分析。"""

from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone

from hscredit.exceptions import ValidationError

from .result import ExplanationResult, coerce_explanation_frame, fingerprint_frame


def _load_shap():
    try:
        import shap
    except Exception as exc:
        raise ImportError(f"SHAP基础依赖加载失败，请检查shap/numba依赖兼容性: {exc}") from exc
    return shap


class ModelExplainer:
    """面向信贷模型的结构化 SHAP 解释器。

    **参数**

    :param model: 已拟合且提供预测接口的模型。
    :param background_data: SHAP 背景数据；未提供时从解释数据确定性抽样。
    :param algorithm: ``auto``、``tree``、``linear``、``permutation`` 或 ``kernel``。

    **属性**

    :attr:`last_result_`: 最近一次 :class:`ExplanationResult`。

    **参考样例**

    >>> result = ModelExplainer(model, background_data=X_train).explain(X_test)
    >>> ModelExplainer(model).get_global_report(result)
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
        background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        algorithm: str = "auto",
        model_output: str = "probability",
        target_class: Any = 1,
        max_background: int = 200,
        random_state: int = 42,
        explainer_type: Optional[str] = None,
    ):
        if model_output not in {"probability", "raw", "score"}:
            raise ValidationError("model_output 必须是 probability、raw 或 score")
        if not isinstance(max_background, int) or max_background <= 0:
            raise ValidationError("max_background 必须是正整数")
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else self._model_feature_names()
        self.background_data = background_data
        self.algorithm = explainer_type or algorithm
        self.explainer_type = self.algorithm  # 旧属性兼容
        self.model_output = model_output
        self.target_class = target_class
        self.max_background = max_background
        self.random_state = random_state
        self.last_result_: Optional[ExplanationResult] = None
        self._explainer = None
        self._explainer_signature = None
        self._interaction_explainer = None
        self._shap_values = None
        self._expected_value = None

    def _native_model(self):
        return getattr(self.model, "_model", self.model)

    def _model_feature_names(self) -> Optional[List[str]]:
        for candidate in (self.model, getattr(self.model, "_model", None)):
            if candidate is not None and hasattr(candidate, "feature_names_in_"):
                return list(candidate.feature_names_in_)
        return None

    def _is_tree_model(self) -> bool:
        native = self._native_model()
        name = native.__class__.__name__.lower()
        return hasattr(native, "tree_") or hasattr(native, "estimators_") or any(token in name for token in ("forest", "tree", "boost", "xgb", "lgbm", "catboost"))

    def _is_linear_model(self) -> bool:
        native = self._native_model()
        return hasattr(native, "coef_") and not self._is_tree_model()

    def _resolve_target_class(self):
        classes = getattr(self.model, "classes_", getattr(self._native_model(), "classes_", None))
        if classes is None:
            if self.model_output == "probability":
                raise ValidationError("概率解释要求模型提供 classes_ 和 predict_proba")
            return None, self.target_class
        classes = list(classes)
        if len(classes) > 2 and self.target_class is None:
            raise ValidationError("多分类模型必须显式指定 target_class")
        target = self.target_class
        if target is None:
            target = 1 if 1 in classes else classes[-1]
        if target not in classes:
            raise ValidationError(f"target_class={target!r} 不在模型类别 {classes!r} 中")
        return classes.index(target), target

    def _sample(self, frame: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
        if limit is None or len(frame) <= limit:
            return frame
        if not isinstance(limit, int) or limit <= 0:
            raise ValidationError("max_samples 必须是正整数")
        return frame.sample(n=limit, random_state=self.random_state).sort_index()

    def _resolve_background(self, frame: pd.DataFrame) -> pd.DataFrame:
        source = frame if self.background_data is None else coerce_explanation_frame(self.background_data, feature_names=frame.columns)
        return self._sample(source, self.max_background)

    def _predict_selected(self, frame, class_index):
        if self.model_output == "score":
            predictor = getattr(self.model, "predict_score", getattr(self.model, "predict", None))
            return np.asarray(predictor(frame), dtype=float).reshape(-1)
        if self.model_output == "raw":
            if hasattr(self.model, "decision_function"):
                output = np.asarray(self.model.decision_function(frame))
                if output.ndim == 1:
                    return -output if class_index == 0 else output
                return output[:, class_index]
            return np.asarray(self.model.predict(frame), dtype=float).reshape(-1)
        probabilities = np.asarray(self.model.predict_proba(frame))
        return probabilities[:, class_index]

    def _choose_algorithm(self):
        if self.algorithm != "auto":
            return self.algorithm
        if self.model_output == "score":
            return "permutation"
        if self._is_tree_model():
            return "tree"
        if self._is_linear_model() and self.model_output == "raw":
            return "linear"
        return "permutation"

    def _build_explainer(self, background: pd.DataFrame, class_index: Optional[int]):
        shap = _load_shap()
        algorithm = self._choose_algorithm()
        signature = (algorithm, fingerprint_frame(background), self.model_output, class_index)
        if self._explainer is not None and self._explainer_signature == signature:
            return self._explainer, algorithm
        native = self._native_model()
        if algorithm == "tree":
            kwargs = {}
            if self.model_output == "probability":
                kwargs = {
                    "data": background,
                    "feature_perturbation": "interventional",
                    "model_output": "probability",
                }
            elif self.model_output != "raw":
                raise ValidationError("树解释器只支持 probability 或 raw 输出尺度")
            backend = shap.TreeExplainer(native, **kwargs)
        elif algorithm == "linear":
            if self.model_output != "raw":
                raise ValidationError("LinearExplainer 仅支持 raw 输出；概率尺度请使用 permutation")
            backend = shap.LinearExplainer(native, background)
        elif algorithm == "permutation":
            predictor = lambda values: self._predict_selected(pd.DataFrame(values, columns=background.columns), class_index)  # noqa: E731
            backend = shap.Explainer(predictor, background, algorithm="permutation")
        elif algorithm == "kernel":
            predictor = lambda values: self._predict_selected(pd.DataFrame(values, columns=background.columns), class_index)  # noqa: E731
            backend = shap.KernelExplainer(predictor, background)
        else:
            raise ValidationError(f"不支持的解释算法: {algorithm}")
        self._explainer = backend
        self._explainer_signature = signature
        self.explainer_type = algorithm
        return backend, algorithm

    def explain(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        *,
        max_samples: Optional[int] = None,
        max_evals: Optional[int] = None,
        check_additivity: bool = True,
    ) -> ExplanationResult:
        """计算选定类别的结构化 SHAP 解释。"""
        frame = self._sample(coerce_explanation_frame(X, self.feature_names), max_samples)
        if self.feature_names is None:
            self.feature_names = list(frame.columns)
        class_index, class_label = self._resolve_target_class()
        background = self._resolve_background(frame)
        backend, algorithm = self._build_explainer(background, class_index)
        shap = _load_shap()
        if algorithm == "kernel":
            raw = backend.shap_values(frame, silent=True)
            explanation = shap.Explanation(
                values=raw,
                base_values=backend.expected_value,
                data=frame.to_numpy(),
                feature_names=list(frame.columns),
            )
            output_index = None
        else:
            kwargs = {}
            if algorithm == "tree":
                kwargs["check_additivity"] = check_additivity
            if algorithm == "permutation":
                kwargs["max_evals"] = max_evals or max(2 * frame.shape[1] + 1, 50)
            explanation = backend(frame, **kwargs)
            output_index = class_index if np.asarray(explanation.values).ndim == 3 else None
        predictions = self._predict_selected(frame, class_index)
        metadata = {
            "模型类型": self.model.__class__.__name__,
            "SHAP版本": shap.__version__,
            "解释算法": algorithm,
            "计算时间": datetime.now(timezone.utc).isoformat(),
            "随机种子": self.random_state,
            "样本数": len(frame),
            "特征数": frame.shape[1],
            "特征顺序": tuple(frame.columns),
            "数据类型": tuple(map(str, frame.dtypes)),
            "目标类别": class_label,
            "请求输出尺度": self.model_output,
            "实际输出尺度": self.model_output,
            "风险方向": "higher_output_lower_risk" if self.model_output == "score" else "higher_output_higher_risk",
            "模型输出": tuple(map(float, predictions)),
        }
        result = ExplanationResult.from_explanation(
            explanation,
            data=frame,
            target_class=class_label,
            output_index=output_index,
            model_output=self.model_output,
            explainer_type=algorithm,
            background_summary={"样本数": len(background), "来源": "解释数据" if self.background_data is None else "显式背景数据"},
            metadata=metadata,
        )
        self.last_result_ = result
        self._shap_values = result.values
        self._expected_value = result.base_values
        return result

    def _require_result(self, result=None) -> ExplanationResult:
        resolved = self.last_result_ if result is None else result
        if not isinstance(resolved, ExplanationResult):
            raise ValidationError("请先调用 explain()，或传入 ExplanationResult")
        return resolved

    def compute_shap_values(self, X, check_additivity: bool = True) -> np.ndarray:
        """计算并返回选定类别的二维 SHAP 数组。"""
        return self.explain(X, check_additivity=check_additivity).values

    def get_shap_importance(self, X=None) -> pd.Series:
        """返回按平均绝对 SHAP 值稳定降序排列的重要性 Series。"""
        result = self.explain(X) if X is not None else self._require_result()
        values = np.abs(result.values).mean(axis=0)
        return pd.Series(values, index=result.feature_names, name="SHAP重要性").sort_values(ascending=False, kind="mergesort")

    def _native_importance(self, names):
        native = self._native_model()
        raw = getattr(native, "feature_importances_", None)
        if raw is None and hasattr(native, "coef_"):
            raw = np.abs(np.asarray(native.coef_)).reshape(-1)
        if raw is None or len(np.asarray(raw).reshape(-1)) != len(names):
            return pd.Series(np.nan, index=names)
        return pd.Series(np.asarray(raw).reshape(-1), index=names, dtype=float)

    def get_global_report(self, result=None) -> pd.DataFrame:
        """生成含重要性、方向、分位数、原生排名和相关性的中文全局报告。"""
        result = self._require_result(result)
        values = result.values
        data = result.data
        mean_abs = np.abs(values).mean(axis=0)
        total = mean_abs.sum()
        native = self._native_importance(result.feature_names)
        rows = []
        for i, name in enumerate(result.feature_names):
            x = pd.to_numeric(data.iloc[:, i], errors="coerce")
            s = pd.Series(values[:, i], index=data.index)
            rows.append(
                {
                    "特征": name,
                    "平均绝对SHAP值": mean_abs[i],
                    "SHAP重要性占比": mean_abs[i] / total if total else 0.0,
                    "平均SHAP值": values[:, i].mean(),
                    "正向影响占比": (values[:, i] > 0).mean(),
                    "负向影响占比": (values[:, i] < 0).mean(),
                    "影响标准差": values[:, i].std(),
                    "P25": np.quantile(values[:, i], 0.25),
                    "P50": np.quantile(values[:, i], 0.50),
                    "P75": np.quantile(values[:, i], 0.75),
                    "原生特征重要性": native[name],
                    "Pearson相关系数": x.corr(s, method="pearson"),
                    "Spearman相关系数": x.corr(s, method="spearman"),
                }
            )
        table = pd.DataFrame(rows).sort_values(["平均绝对SHAP值", "特征"], ascending=[False, True], kind="mergesort")
        table["SHAP排名"] = range(1, len(table) + 1)
        table["原生排名"] = table["原生特征重要性"].rank(method="min", ascending=False)
        table["排名差"] = table["原生排名"] - table["SHAP排名"]
        return table.reset_index(drop=True)

    def get_sample_report(self, result=None, *, sample_id=None, position=None, top_n=None) -> pd.DataFrame:
        """按样本索引或位置生成局部贡献长表。"""
        result = self._require_result(result)
        if sample_id is not None and position is not None:
            raise ValidationError("sample_id 与 position 不能同时指定")
        if sample_id is not None:
            position = result.position_for(sample_id)
        position = 0 if position is None else int(position)
        if position < 0 or position >= len(result.data):
            raise ValidationError("样本位置超出范围")
        values = result.values[position]
        order = np.argsort(-np.abs(values), kind="stable")
        if top_n is not None:
            if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
                raise ValidationError("top_n 必须是正整数或 None")
            order = order[:top_n]
        cumulative = 0.0
        rows = []
        output = float(result.metadata["模型输出"][position])
        for rank, index in enumerate(order, 1):
            cumulative += float(values[index])
            rows.append(
                {
                    "样本索引": result.sample_ids[position],
                    "目标类别": result.target_class,
                    "模型输出": output,
                    "基准值": result.base_values[position],
                    "特征": result.feature_names[index],
                    "特征值": result.data.iloc[position, index],
                    "SHAP值": values[index],
                    "绝对贡献": abs(values[index]),
                    "贡献方向": "提高输出" if values[index] > 0 else "降低输出" if values[index] < 0 else "无影响",
                    "累计贡献": cumulative,
                    "贡献排名": rank,
                }
            )
        return pd.DataFrame(rows)

    def select_representative_samples(self, result=None, threshold=0.5, risk_direction=None) -> pd.DataFrame:
        """选择最高/最低风险、阈值附近、中位输出和贡献最大的代表样本。

        :param result: 结构化解释结果；None 时使用最近一次结果。
        :param threshold: 当前输出尺度下的业务决策阈值。
        :param risk_direction: ``higher_output_higher_risk`` 或 ``higher_output_lower_risk``；
            None 时从解释元信息推导。
        :return: 包含样本索引、选择理由、模型输出、风险排名和阈值距离的中文表。
        """
        result = self._require_result(result)
        outputs = np.asarray(result.metadata["模型输出"], dtype=float)
        if not np.isfinite(outputs).all() or outputs.size == 0:
            raise ValidationError("代表样本要求模型输出为非空有限数组")
        if not np.isscalar(threshold) or not np.isfinite(threshold):
            raise ValidationError("threshold 必须是有限数")
        direction = risk_direction or result.metadata.get("风险方向")
        if direction is None:
            direction = "higher_output_lower_risk" if result.model_output == "score" else "higher_output_higher_risk"
        if direction not in {"higher_output_higher_risk", "higher_output_lower_risk"}:
            raise ValidationError("risk_direction 必须是 higher_output_higher_risk 或 higher_output_lower_risk")
        highest_risk = int(np.argmax(outputs)) if direction == "higher_output_higher_risk" else int(np.argmin(outputs))
        lowest_risk = int(np.argmin(outputs)) if direction == "higher_output_higher_risk" else int(np.argmax(outputs))
        total_abs = np.abs(result.values).sum(axis=1)
        candidates = [
            (highest_risk, "最高风险"),
            (lowest_risk, "最低风险"),
            (int(np.argmin(np.abs(outputs - threshold))), "最接近决策阈值"),
            (int(np.argmin(np.abs(outputs - np.median(outputs)))), "最接近总体中位输出"),
            (int(np.argmax(total_abs)), "总绝对贡献最大"),
        ]
        if result.model_output == "probability":
            candidates.append((int(np.argmin(np.abs(outputs - 0.5))), "最不确定样本"))
        elif result.model_output == "raw":
            candidates.append((int(np.argmin(np.abs(outputs))), "最不确定样本"))
        reasons = {}
        for position, reason in candidates:
            reasons.setdefault(position, []).append(reason)
        ranks = pd.Series(outputs).rank(
            method="min", ascending=direction == "higher_output_lower_risk"
        ).astype(int).to_numpy()
        return pd.DataFrame(
            [
                {
                    "样本索引": result.sample_ids[position],
                    "选择理由": "、".join(labels),
                    "模型输出": outputs[position],
                    "风险排名": ranks[position],
                    "阈值距离": abs(outputs[position] - threshold),
                }
                for position, labels in reasons.items()
            ]
        )

    def get_correlation_report(self, result=None, kind="feature_shap") -> pd.DataFrame:
        """返回特征-SHAP 或 SHAP-SHAP 的相关性报告。"""
        result = self._require_result(result)
        if kind == "shap_shap":
            return pd.DataFrame(result.values, columns=result.feature_names).corr(method="spearman")
        if kind != "feature_shap":
            raise ValidationError("kind 必须是 feature_shap 或 shap_shap")
        table = self.get_global_report(result)
        return table[["特征", "Pearson相关系数", "Spearman相关系数"]]

    def get_feature_clusters(self, result=None, max_clusters=None) -> pd.DataFrame:
        """按 SHAP 贡献相关性返回层次聚类叶序和聚类编号。"""
        result = self._require_result(result)
        n_features = len(result.feature_names)
        if max_clusters is not None and (
            not isinstance(max_clusters, int) or isinstance(max_clusters, bool) or max_clusters <= 0
        ):
            raise ValidationError("max_clusters 必须是正整数或 None")
        if n_features == 1:
            return pd.DataFrame({"特征": result.feature_names, "叶序": [1], "聚类编号": [1]})
        corr = pd.DataFrame(result.values).corr(method="spearman").fillna(0).to_numpy()
        np.fill_diagonal(corr, 1.0)
        distance = np.clip(1 - np.abs(corr), 0, 1)
        tree = linkage(squareform(distance, checks=False), method="average", optimal_ordering=True)
        leaves = leaves_list(tree)
        count = max_clusters or min(4, n_features)
        labels = fcluster(tree, t=count, criterion="maxclust")
        leaf_order = {int(feature): position + 1 for position, feature in enumerate(leaves)}
        return pd.DataFrame({"特征": result.feature_names, "叶序": [leaf_order[i] for i in range(n_features)], "聚类编号": labels}).sort_values("叶序").reset_index(drop=True)

    def get_feature_interactions(self, X=None, top_n=10, result=None) -> pd.DataFrame:
        """返回树模型精确交互或非树模型近似交互的前 N 个特征对。"""
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            raise ValidationError("top_n 必须是正整数")
        result = self._require_result(result) if result is not None else (self.explain(X) if X is not None else self._require_result())
        if not self._is_tree_model():
            return self.get_approximate_interactions(result, top_n=top_n)
        shap = _load_shap()
        injected = self._interaction_explainer
        if injected is not None and hasattr(injected, "shap_interaction_values"):
            backend = injected
        else:
            backend = shap.TreeExplainer(self._native_model())
        # 概率尺度不支持精确交互，使用同一树模型的 raw TreeExplainer。
        raw = backend.shap_interaction_values(result.data)
        if isinstance(raw, list):
            raw = raw[result.output_index or 0]
        array = np.asarray(raw)
        if array.ndim == 4:
            array = array[:, :, :, result.output_index or 0]
        strength = np.mean(np.abs(array), axis=0)
        rows = []
        for left in range(strength.shape[0]):
            for right in range(left + 1, strength.shape[1]):
                rows.append({"特征1": result.feature_names[left], "特征2": result.feature_names[right], "交互强度": strength[left, right]})
        columns = ["特征1", "特征2", "交互强度"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns).sort_values(["交互强度", "特征1", "特征2"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)

    def get_approximate_interactions(self, result=None, top_n=10) -> pd.DataFrame:
        """根据 SHAP 贡献 Spearman 相关性返回近似交互特征对。"""
        if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
            raise ValidationError("top_n 必须是正整数")
        result = self._require_result(result)
        corr = pd.DataFrame(result.values, columns=result.feature_names).corr(method="spearman").abs().fillna(0)
        rows = []
        for left in range(len(result.feature_names)):
            for right in range(left + 1, len(result.feature_names)):
                rows.append({"特征1": result.feature_names[left], "特征2": result.feature_names[right], "交互强度": corr.iloc[left, right], "交互类型": "近似"})
        columns = ["特征1", "特征2", "交互强度", "交互类型"]
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns).sort_values(["交互强度", "特征1", "特征2"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)

    def get_stability_report(
        self,
        result=None,
        *,
        mode="sample",
        X_train=None,
        y_train=None,
        X_validation=None,
        n_bootstrap=100,
        confidence_level=0.95,
        top_k=10,
        random_state=None,
    ) -> pd.DataFrame:
        """评估固定样本 Bootstrap 或模型重训后的解释稳定性。"""
        if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap < 2:
            raise ValidationError("n_bootstrap 必须是不小于 2 的整数")
        if not np.isscalar(confidence_level) or not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValidationError("confidence_level 必须是(0, 1)范围内的有限数")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValidationError("top_k 必须是正整数")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        if mode == "sample":
            resolved = self._require_result(result)
            names = resolved.feature_names
            runs = [np.abs(resolved.values[rng.integers(0, len(resolved.data), len(resolved.data))]).mean(axis=0) for _ in range(n_bootstrap)]
            label = "样本Bootstrap"
        elif mode == "refit":
            if X_train is None or y_train is None or X_validation is None:
                raise ValidationError("refit 模式必须提供训练数据、标签和固定验证数据")
            train = coerce_explanation_frame(X_train, self.feature_names)
            validation = coerce_explanation_frame(X_validation, train.columns)
            target = np.asarray(y_train)
            names = list(train.columns)
            runs = []
            for bootstrap_index in range(n_bootstrap):
                indices = rng.integers(0, len(train), len(train))
                try:
                    fitted = clone(self.model).fit(train.iloc[indices], target[indices])
                    child = ModelExplainer(
                        fitted,
                        background_data=train.iloc[indices[: self.max_background]],
                        algorithm=self.algorithm,
                        model_output=self.model_output,
                        target_class=self.target_class,
                        random_state=self.random_state,
                    )
                    runs.append(np.abs(child.explain(validation).values).mean(axis=0))
                except Exception as exc:
                    raise ValidationError(f"第 {bootstrap_index + 1} 次重训解释失败: {exc}") from exc
            label = "重训Bootstrap"
        else:
            raise ValidationError("mode 必须是 sample 或 refit")
        matrix = np.asarray(runs)
        ranks = np.argsort(np.argsort(-matrix, axis=1), axis=1) + 1
        alpha = (1 - confidence_level) / 2
        rows = []
        for index, name in enumerate(names):
            rows.append(
                {
                    "特征": name,
                    "稳定性模式": label,
                    "平均绝对SHAP值": matrix[:, index].mean(),
                    "置信区间下限": np.quantile(matrix[:, index], alpha),
                    "置信区间上限": np.quantile(matrix[:, index], 1 - alpha),
                    "排名均值": ranks[:, index].mean(),
                    "排名标准差": ranks[:, index].std(),
                    "Top-K入选率": (ranks[:, index] <= min(top_k, len(names))).mean(),
                }
            )
        return pd.DataFrame(rows).sort_values(["排名均值", "特征"]).reset_index(drop=True)

    def get_reason_codes(
        self,
        result=None,
        *,
        keep=3,
        risk_direction="higher_output_higher_risk",
        feature_map=None,
        reason_map=None,
    ) -> pd.DataFrame:
        """返回只包含不利局部贡献的中文业务原因码。"""
        from .reason_codes import build_reason_codes

        return build_reason_codes(
            self._require_result(result),
            keep=keep,
            risk_direction=risk_direction,
            feature_map=feature_map,
            reason_map=reason_map,
        )

    # 新图形入口延迟导入，避免核心计算依赖绘图实现。
    def _plot(self, name, result=None, **kwargs):
        from . import plots as explanation_plots

        return getattr(explanation_plots, name)(self._require_result(result), explainer=self, **kwargs)

    def plot_decision(self, result=None, **kwargs):
        """绘制单样本 SHAP 决策贡献条形图并返回 Figure。"""
        return self._plot("plot_decision", result, **kwargs)

    def plot_heatmap(self, result=None, **kwargs):
        """绘制多样本 SHAP 贡献热力图并返回 Figure。"""
        return self._plot("plot_heatmap", result, **kwargs)

    def plot_distribution(self, result=None, **kwargs):
        """绘制指定特征值与 SHAP 贡献分布并返回 Figure。"""
        return self._plot("plot_distribution", result, **kwargs)

    def plot_correlation(self, result=None, **kwargs):
        """绘制 SHAP 贡献相关性热力图并返回 Figure。"""
        return self._plot("plot_correlation", result, **kwargs)

    def plot_feature_clustering(self, result=None, **kwargs):
        """绘制基于 SHAP 贡献距离的特征层次聚类图。"""
        return self._plot("plot_feature_clustering", result, **kwargs)

    def plot_interaction_heatmap(self, result=None, **kwargs):
        """绘制树精确或近似 SHAP 交互强度热力图。"""
        return self._plot("plot_interaction_heatmap", result, **kwargs)

    def plot_interaction_bubble(self, result=None, **kwargs):
        """绘制主要 SHAP 交互特征对气泡图。"""
        return self._plot("plot_interaction_bubble", result, **kwargs)

    def plot_importance_overview(self, result=None, **kwargs):
        """绘制 SHAP 贡献分布与全局重要性组合图。"""
        return self._plot("plot_importance_overview", result, **kwargs)

    def plot_explanation_overview(self, result=None, **kwargs):
        """绘制重要性、方向、相关性与代表样本综合总览。"""
        return self._plot("plot_explanation_overview", result, **kwargs)

    def plot_shap_summary(self, X=None, plot_type="dot", max_display=20, show=True, **kwargs):
        """绘制 SHAP summary 图；支持 ``dot``、``violin`` 和 ``bar``。"""
        result = X if isinstance(X, ExplanationResult) else (self.explain(X) if X is not None else self._require_result())
        if plot_type == "bar":
            return self.plot_shap_bar(result, max_display=max_display, show=show, **kwargs)
        if plot_type not in {"dot", "violin"}:
            raise ValidationError("plot_type 必须是 dot、violin 或 bar")
        shap = _load_shap()
        import matplotlib.pyplot as plt

        shap.summary_plot(
            result.values,
            result.data,
            feature_names=result.feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )
        figure = plt.gcf()
        if "figsize" in kwargs:
            figure.set_size_inches(*kwargs["figsize"])
        if kwargs.get("title"):
            figure.axes[0].set_title(kwargs["title"])
        if show:
            plt.show()
        return figure

    def plot_shap_bar(self, X=None, max_display=20, show=True, **kwargs):
        """绘制单面板平均绝对 SHAP 重要性条形图。"""
        result = X if isinstance(X, ExplanationResult) else (self.explain(X) if X is not None else self._require_result())
        from .plots import plot_shap_result_importance

        return plot_shap_result_importance(result, max_display=max_display, show=show, **kwargs)

    def plot_shap_dependence(self, feature, X=None, show=True, **kwargs):
        """绘制指定特征值与其 SHAP 贡献的依赖散点图。"""
        result = self.explain(X) if X is not None else self._require_result()
        return self.plot_distribution(result, feature=feature, show=show, **kwargs)

    def plot_combined_importance(self, X=None, top_n=15, show=True, **kwargs):
        """并排绘制模型原生重要性与 SHAP 重要性。"""
        result = self.explain(X) if X is not None else self._require_result()
        from .plots import plot_importance_comparison

        return plot_importance_comparison(self.model, result.data, top_n=top_n, show=show, **kwargs)

    def plot_shap_waterfall(self, X, sample_idx=0, max_display=15, show=True, **kwargs):
        """绘制真实 SHAP waterfall 图并返回 Figure。"""
        result = self.explain(X) if not isinstance(X, ExplanationResult) else X
        if not isinstance(sample_idx, int) or isinstance(sample_idx, bool) or not 0 <= sample_idx < len(result.data):
            raise ValidationError("sample_idx 超出解释样本范围")
        shap = _load_shap()
        import matplotlib.pyplot as plt

        axis = shap.plots.waterfall(result.explanation[sample_idx], max_display=max_display, show=False)
        figure = axis.figure
        if "figsize" in kwargs:
            figure.set_size_inches(*kwargs["figsize"])
        axis.set_title(kwargs.get("title", "单样本SHAP瀑布图"))
        figure.tight_layout()
        if show:
            plt.show()
        return figure

    def plot_shap_force(self, X, sample_idx=0, show=True, **kwargs):
        """绘制 Matplotlib SHAP force 图并返回 Figure。"""
        result = self.explain(X) if not isinstance(X, ExplanationResult) else X
        if not isinstance(sample_idx, int) or isinstance(sample_idx, bool) or not 0 <= sample_idx < len(result.data):
            raise ValidationError("sample_idx 超出解释样本范围")
        shap = _load_shap()
        import matplotlib.pyplot as plt

        shap.force_plot(
            result.base_values[sample_idx],
            result.values[sample_idx],
            result.data.iloc[sample_idx],
            feature_names=result.feature_names,
            matplotlib=True,
            show=False,
        )
        figure = plt.gcf()
        if "figsize" in kwargs:
            figure.set_size_inches(*kwargs["figsize"])
        figure.axes[0].set_title(kwargs.get("title", "单样本SHAP力图"))
        if show:
            plt.show()
        return figure
