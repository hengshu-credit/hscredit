"""统一 SHAP 模型解释器与结构化分析。"""

from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone

from hscredit.exceptions import ValidationError

from .explanation import ExplanationResult, coerce_explanation_frame, fingerprint_frame


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

        # 保留历史上可替换 _explainer 的树交互测试/扩展点；正式 explain 会按尺度和背景重建。
        if self.algorithm in {"auto", "tree"} and self._is_tree_model():
            try:
                self._explainer = _load_shap().TreeExplainer(self._native_model())
                self._interaction_explainer = self._explainer
            except Exception:
                self._explainer = None

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
                return output if output.ndim == 1 else output[:, class_index]
            return np.asarray(self.model.predict(frame), dtype=float).reshape(-1)
        probabilities = np.asarray(self.model.predict_proba(frame))
        return probabilities[:, class_index]

    def _choose_algorithm(self):
        if self.algorithm != "auto":
            return self.algorithm
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
        return self.explain(X, check_additivity=check_additivity).values

    def get_shap_importance(self, X=None) -> pd.Series:
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
        result = self._require_result(result)
        values = result.values
        mean_abs = np.abs(values).mean(axis=0)
        total = mean_abs.sum()
        native = self._native_importance(result.feature_names)
        rows = []
        for i, name in enumerate(result.feature_names):
            x = pd.to_numeric(result.data.iloc[:, i], errors="coerce")
            s = pd.Series(values[:, i], index=result.data.index)
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
        result = self._require_result(result)
        if sample_id is not None and position is not None:
            raise ValidationError("sample_id 与 position 不能同时指定")
        if sample_id is not None:
            position = result.position_for(sample_id)
        position = 0 if position is None else int(position)
        if position < 0 or position >= len(result.data):
            raise ValidationError("样本位置超出范围")
        values = result.values[position]
        order = np.argsort(np.abs(values), kind="stable")[::-1]
        if top_n is not None:
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

    def select_representative_samples(self, result=None, threshold=0.5) -> pd.DataFrame:
        result = self._require_result(result)
        outputs = np.asarray(result.metadata["模型输出"], dtype=float)
        total_abs = np.abs(result.values).sum(axis=1)
        candidates = [
            (int(np.argmax(outputs)), "最高风险"),
            (int(np.argmin(outputs)), "最低风险"),
            (int(np.argmin(np.abs(outputs - threshold))), "最接近决策阈值"),
            (int(np.argmin(np.abs(outputs - np.median(outputs)))), "最接近总体中位输出"),
            (int(np.argmin(np.abs(outputs - 0.5))), "最不确定样本"),
            (int(np.argmax(total_abs)), "总绝对贡献最大"),
        ]
        reasons = {}
        for position, reason in candidates:
            reasons.setdefault(position, []).append(reason)
        ranks = pd.Series(outputs).rank(method="min", ascending=False).astype(int).to_numpy()
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
        result = self._require_result(result)
        if kind == "shap_shap":
            return pd.DataFrame(result.values, columns=result.feature_names).corr(method="spearman")
        if kind != "feature_shap":
            raise ValidationError("kind 必须是 feature_shap 或 shap_shap")
        table = self.get_global_report(result)
        return table[["特征", "Pearson相关系数", "Spearman相关系数"]]

    def get_feature_clusters(self, result=None, max_clusters=None) -> pd.DataFrame:
        result = self._require_result(result)
        n_features = len(result.feature_names)
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
        result = self._require_result(result) if result is not None else (self.explain(X) if X is not None else self._require_result())
        if not self._is_tree_model():
            return self.get_approximate_interactions(result, top_n=top_n)
        shap = _load_shap()
        injected = self._interaction_explainer
        if injected is not None and "shap_interaction_values" in getattr(injected, "__dict__", {}):
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
        return pd.DataFrame(rows).sort_values(["交互强度", "特征1", "特征2"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)

    def get_approximate_interactions(self, result=None, top_n=10) -> pd.DataFrame:
        result = self._require_result(result)
        corr = pd.DataFrame(result.values, columns=result.feature_names).corr(method="spearman").abs().fillna(0)
        rows = []
        for left in range(len(result.feature_names)):
            for right in range(left + 1, len(result.feature_names)):
                rows.append({"特征1": result.feature_names[left], "特征2": result.feature_names[right], "交互强度": corr.iloc[left, right], "交互类型": "近似"})
        return pd.DataFrame(rows).sort_values(["交互强度", "特征1", "特征2"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)

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
        if not isinstance(n_bootstrap, int) or n_bootstrap < 2:
            raise ValidationError("n_bootstrap 必须是不小于 2 的整数")
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
        from . import explanation_plots

        return getattr(explanation_plots, name)(self._require_result(result), explainer=self, **kwargs)

    def plot_decision(self, result=None, **kwargs):
        return self._plot("plot_decision", result, **kwargs)

    def plot_heatmap(self, result=None, **kwargs):
        return self._plot("plot_heatmap", result, **kwargs)

    def plot_distribution(self, result=None, **kwargs):
        return self._plot("plot_distribution", result, **kwargs)

    def plot_correlation(self, result=None, **kwargs):
        return self._plot("plot_correlation", result, **kwargs)

    def plot_feature_clustering(self, result=None, **kwargs):
        return self._plot("plot_feature_clustering", result, **kwargs)

    def plot_interaction_heatmap(self, result=None, **kwargs):
        return self._plot("plot_interaction_heatmap", result, **kwargs)

    def plot_interaction_bubble(self, result=None, **kwargs):
        return self._plot("plot_interaction_bubble", result, **kwargs)

    def plot_importance_overview(self, result=None, **kwargs):
        return self._plot("plot_importance_overview", result, **kwargs)

    def plot_explanation_overview(self, result=None, **kwargs):
        return self._plot("plot_explanation_overview", result, **kwargs)

    # 旧绘图方法保留原名，转到统一结果。
    def plot_shap_summary(self, X=None, plot_type="dot", max_display=20, show=True, **kwargs):
        result = self.explain(X) if X is not None else self._require_result()
        return self.plot_importance_overview(result, max_display=max_display, show=show)

    def plot_shap_bar(self, X=None, max_display=20, show=True, **kwargs):
        result = self.explain(X) if X is not None else self._require_result()
        return self.plot_importance_overview(result, max_display=max_display, show=show)

    def plot_shap_dependence(self, feature, X=None, show=True, **kwargs):
        result = self.explain(X) if X is not None else self._require_result()
        return self.plot_distribution(result, feature=feature, show=show)

    def plot_combined_importance(self, X=None, top_n=15, show=True, **kwargs):
        result = self.explain(X) if X is not None else self._require_result()
        return self.plot_importance_overview(result, max_display=top_n, show=show)

    def plot_shap_waterfall(self, X, sample_idx=0, max_display=15, show=True, **kwargs):
        result = self.explain(X) if not isinstance(X, ExplanationResult) else X
        return self.plot_decision(result, position=sample_idx, max_display=max_display, show=show)

    def plot_shap_force(self, X, sample_idx=0, show=True, **kwargs):
        return self.plot_shap_waterfall(X, sample_idx=sample_idx, show=show, **kwargs)
