"""分类模型、评分卡和校准包装器共用的验证、过程查询与报告入口。"""

import copy

import numpy as np
import pandas as pd


def prepare_eval_sets(model, eval_set, *, target=None):
    """解析与 ModelReport.datasets 一致的输入；标签始终从特征中剔除。"""
    from ._contracts import evaluation_set_entries, extract_target

    model.eval_names_ = []
    if eval_set is None or (isinstance(eval_set, (list, tuple)) and not eval_set):
        return []
    result = []
    for name, item in evaluation_set_entries(eval_set):
        if isinstance(item, pd.DataFrame):
            X, y = item, None
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            X, y = item
        else:
            raise ValueError(f"评估集 {name!r} 必须是 (X, y) 或包含目标列的 DataFrame")
        X, y = extract_target(X, y, target or getattr(model, "target_", getattr(model, "target", None)))
        result.append((X, y))
        model.eval_names_.append(name)
    return result


def evaluate_probability_model(
    model,
    X,
    y=None,
    *,
    metrics=None,
    sample_weight=None,
    positive_class=None,
    score_direction="higher_risk",
    predict_params=None,
):
    """评分卡 predict 返回分数，统一评估必须使用概率生成类别，避免误作类别标签。"""
    from ._contracts import extract_target, positive_probability, validate_labels
    from .base import _evaluate_binary_predictions

    if not callable(getattr(X, "get_label", None)):
        X, y = extract_target(X, y, getattr(model, "target_", getattr(model, "target", None)))
    probabilities = np.asarray(model.predict_proba(X, **(predict_params or {})))
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if classes.shape != (2,):
        raise ValueError("统一模型评估仅支持二分类概率")
    positive = getattr(model, "positive_class_", classes[1]) if positive_class is None else positive_class
    labels = validate_labels(y, classes=classes, require_both=False)
    p = positive_probability(probabilities, classes, positive)
    label_method = getattr(model, "_predict_evaluation_labels", None)
    predicted = (
        (np.asarray(label_method(X)) == positive).astype(int) if callable(label_method) else (p >= 0.5).astype(int)
    )
    return _evaluate_binary_predictions(
        (labels == positive).astype(int),
        p,
        predicted,
        metrics=["auc", "ks", "logloss"] if metrics is None else metrics,
        sample_weight=sample_weight,
        score_direction=score_direction,
    )


def record_eval_results(model, eval_set, *, metrics=None, eval_sample_weight=None, predict_params=None, staged=False):
    """最终评估与逐轮曲线分别保留；没有 staged 接口时不伪造曲线。"""
    from .losses._sample_params import has_bound_sample_params
    from ._contracts import validate_sample_weight
    from .base import _evaluate_binary_predictions

    sets = eval_set or []
    if len(sets) > 1 and has_bound_sample_params(metrics):
        raise ValueError("绑定逐行金融参数的指标只能对应一个评估集；多数据集请分别 evaluate 或使用样本权重")
    names = getattr(model, "eval_names_", [])
    rows = []
    for index, (X, y) in enumerate(sets):
        name = names[index] if index < len(names) else f"数据集{index + 1}"
        weight = eval_sample_weight
        if isinstance(weight, dict):
            if name not in weight:
                raise ValueError(f"eval_sample_weight 缺少数据集 {name!r} 的权重")
            weight = weight[name]
        elif weight is not None and len(sets) > 1:
            raise ValueError("多个评估集的 eval_sample_weight 请使用 {数据集名称: 权重} 字典")
        weight = validate_sample_weight(weight, len(y))
        scores = evaluate_probability_model(
            model,
            X,
            y,
            metrics=metrics,
            sample_weight=weight,
            predict_params=predict_params,
        )
        rows.extend({"数据集": name, "指标": metric, "数值": value} for metric, value in scores.items())
        if type(model).__module__.startswith("hscredit.core.models.classical"):
            model._evals_result.setdefault(f"validation_{index}", scores)
        estimator = getattr(model, "_model", None)
        if staged and callable(getattr(estimator, "staged_predict_proba", None)):
            curves = {}
            for proba in estimator.staged_predict_proba(X):
                p = np.asarray(proba)[:, 1]
                values = _evaluate_binary_predictions(
                    y,
                    p,
                    (p >= 0.5).astype(int),
                    metrics=["auc", "ks", "logloss"] if metrics is None else metrics,
                    sample_weight=weight,
                )
                for metric, value in values.items():
                    curves.setdefault(metric, []).append(value)
            model._evals_result[f"validation_{index}"] = curves
    model.eval_results_ = pd.DataFrame(rows, columns=["数据集", "指标", "数值"])


def record_boosting_evaluation(model, eval_set, *, eval_metric=None, weights=None):
    """保存最终预测对应的评估，原生框架专用指标仍保留在原生曲线中。"""
    from .losses.base import BaseMetric

    if eval_set is None:
        return
    if callable(getattr(eval_set, "get_label", None)):
        eval_set = [eval_set]
    sets = []
    inferred_weights = {}
    names = getattr(model, "eval_names_", [])
    for index, item in enumerate(eval_set):
        name = names[index] if index < len(names) else f"数据集{index + 1}"
        if callable(getattr(item, "get_label", None)):
            sets.append((item, item.get_label()))
            values = item.get_weight()
            inferred_weights[name] = values if len(values) else None
        else:
            sets.append(item)
            inferred_weights[name] = None
        if weights is not None:
            inferred_weights[name] = weights[index]
    requested = eval_metric if isinstance(eval_metric, (list, tuple)) else [eval_metric]
    metrics = [item for item in requested if isinstance(item, BaseMetric)]
    record_eval_results(model, sets, metrics=metrics or None, eval_sample_weight=inferred_weights or None)


class TrainingProcessMixin:
    """训练器与概率/评分变换器共用过程查询，不隐含可训练的分类器语义。"""

    def get_training_history(self):
        """返回每次拟合的配置、内部转换、实际参数、状态和耗时。

        :return: 中文列名 DataFrame。训练修订从1开始，最多保留 max_history 次；
            不携带原始训练数据。未训练时返回空表，失败/中断的记录也保留。

        **参考样例**

        >>> history = model.get_training_history()  # model 已完成 fit 或 tune 后的最终重训
        >>> latest = history.iloc[-1] if not history.empty else None
        """
        from ._lifecycle import training_history_table

        return training_history_table(self)

    def get_learning_curve(self, revision=None):
        """查询某次模型训练的逐轮指标，不重新训练。

        :param revision: 训练修订号，None 为最近一次 fit；取值见 get_training_history()['训练修订']。
            历史超出 max_history 时无法恢复，会明确报错。

        :return: 含训练修订、数据集、原生数据集、迭代、指标、数值的中文长表。
            不产生逐轮曲线的算法返回空表，最终评估请用 get_eval_results()。

        **参考样例**

        >>> curve = model.get_learning_curve()
        >>> first_curve = model.get_learning_curve(revision=1)  # 前提：仍保留第一次拟合记录
        >>> trial_curve = model.tuner.get_learning_curve()  # 搜索最佳点曲线，与最终重训曲线分开
        """
        from ._lifecycle import model_learning_curve

        return model_learning_curve(self, revision)

    def get_eval_results(self, revision=None):
        """查询某次 fit 完成后，模型在各 eval_set 上的最终指标。

        :param revision: 训练修订号，None 为最近一次 fit；不是 Optuna Trial 编号。
        :return: 包含数据集、指标、数值的 DataFrame 副本；没有 eval_set 时返回空表。
            get_best_model() 默认全量重训，不会自动沿用搜索验证集；独立测试集用 evaluate() 计算。

        :raises ValueError: 指定修订已被历史保留策略移除。

        **参考样例**

        >>> final_validation = model.get_eval_results()
        >>> test_metrics = model.evaluate(X_test, y_test, metrics=['auc', 'ks'])
        """
        history = getattr(self, "training_history_", [])
        current = getattr(self, "fit_revision_", None)
        if revision is None or revision == current:
            return copy.deepcopy(getattr(self, "eval_results_", pd.DataFrame(columns=["数据集", "指标", "数值"])))
        for record in history:
            if record.get("训练修订") == revision:
                return pd.DataFrame(record.get("最终评估", []), columns=["数据集", "指标", "数值"])
        raise ValueError(f"未保留训练修订 {revision}，请检查 max_history 或历史制品")


class ModelWorkflowMixin(TrainingProcessMixin):
    """有 predict_proba 的模型共用金融指标与完整模型报告，保留各类专用报告。"""

    def evaluate(
        self,
        X,
        y=None,
        sample_weight=None,
        metrics=None,
        *,
        positive_class=None,
        score_direction="higher_risk",
        **predict_params,
    ):
        """按当前模型的概率计算效果，评分卡不会把信用分当作类别标签。

        :param X: 二维特征，或包含目标列的 DataFrame；目标列由模型 target 指定。
        :param y: 真实标签；None 时从 X 提取，显式 y 优先且目标列仍从特征剔除。
        :param sample_weight: 与评估行等长的有限非负权重；None 为等权，不继承训练权重。
        :param metrics: 名称、get_metric(...) / loss.metric(...) 对象或列表；None 使用 AUC、KS、LogLoss。
        :param positive_class: 正类标签，默认模型约定的正类；信贷模型以1表示坏样本。
        :param score_direction: AUC/Gini 的方向；默认 higher_risk 保留反向预测的低 AUC，
            higher_safe 用于高值低风险，auto 仅在显式指定时自动判断。

        :param predict_params: 传给 predict_proba 的参数；评分卡可用 input_type='woe'/'raw'，
            返回值必须仍是概率，不能请求 raw_score/叶节点矩阵。

        :return: 指标名到数值的字典，不改写模型、报告缓存或 fit 的评估记录。

        **参考样例**

        >>> from hscredit import get_loss
        >>> cost = get_loss('cost_sensitive', fn_cost=5, fp_cost=1).metric(kind='business')
        >>> values = model.evaluate(X_test, y_test, metrics=['auc', cost])
        """
        return evaluate_probability_model(
            self,
            X,
            y,
            metrics=metrics,
            sample_weight=sample_weight,
            positive_class=positive_class,
            score_direction=score_direction,
            predict_params=predict_params,
        )

    def model_report(self, datasets=None, *, target=None, metrics=None, sample_weight=None, **kwargs):
        """生成当前模型的完整报告，衔接搜索模型、效果评估及 Excel 输出。

        :param datasets: 与 eval_set 相同的具名字典/列表，元素为 (X, y) 或含目标列的 DataFrame。
            例如 {'训练': train_df, '验证': valid_df, '独立测试': test_df}；传完整数据时模型仅预测入模字段。

        :param target: 目标列名或报告多标签配置；None 继承模型目标列。overdue/dpds 模式不继承目标列。
        :param metrics: 统一指标名称/对象/列表；需与 evaluate 使用同一对象和参数。
            None 只输出报告默认指标，不自动把调参目标当作报告指标。

        :param sample_weight: 报告评估权重，可为列名、{数据集名: 权重}，或单数据集的权重数组。
            金额分析列 amount 与该参数作用不同，均不自动继承训练或搜索权重。

        :param kwargs: auto_model_report 的参数，如 excel_path、features、verbose、amount、date、
            prediction_method、prediction_kwargs、mode、return_result、with_plots、include_sample_records。
            默认返回 ModelReport；显式 return_result=True 改为 ReportResult 执行状态。

        :return: 默认 ModelReport，可调用 get_metrics()/get_training_process()/to_excel()；
            提供 excel_path 会同时导出。完整报告绑定模型状态，模型重训后需新建报告。

        **参考样例**

        >>> report = model.model_report(datasets={'独立测试': (X_test, y_test)}, metrics=['auc'], verbose=False)
        >>> values = report.get_metrics()
        >>> process = report.get_training_process()
        >>> status = report.to_excel('output/模型报告.xlsx', mode='strict', return_result=True,
        ...                          with_plots=False, include_sample_records=False)
        >>> assert status.complete
        """
        from ...report import auto_model_report

        if target is None and kwargs.get("overdue") is None:
            target = getattr(self, "target_", getattr(self, "target", None))
        return auto_model_report(
            model=self, datasets=datasets, target=target, metrics=metrics, sample_weight=sample_weight, **kwargs
        )

    def report(self, datasets=None, **kwargs):
        """兼容的完整模型报告入口。

        :param datasets: 具名字典或列表，元素为 (X, y) 或含目标列的 DataFrame。
        :param kwargs: model_report 的参数，如 metrics、target、excel_path、verbose。
        :return: 默认 ModelReport；return_result=True 返回执行状态。

        ProbabilityScoreCard 和 ProbabilityCalibrator 的 report 专用表保持原义；
        跨模型编写统一流程时使用 model_report()。

        **参考样例**

        >>> report = model.model_report(datasets={'测试': (X_test, y_test)}, verbose=False)
        """
        return self.model_report(datasets=datasets, **kwargs)

    def tune(
        self,
        X,
        y=None,
        *,
        search_space=None,
        metric="ks",
        cv=None,
        eval_set=None,
        n_trials=100,
        sample_weight=None,
        validation_weight=None,
        **kwargs,
    ):
        """搜索超参数并返回一个重新拟合的模型；源实例保留 tuner 供过程查询。

        :param X: 训练特征或包含目标列的 DataFrame；独立验证集和测试集不要拼入其中。
        :param y: 训练标签；None 按模型 target 提取。
        :param search_space: Optuna 兼容搜索空间；评分卡、规则和校准模型需显式提供，如 {'n_bins': [5, 10]}。
        :param metric: 选模指标名称、统一指标对象、带方向函数或列表；不是训练损失。
        :param cv: 默认 None 单次留出；显式整数>=2、分割器或索引对启用多折，最终仍返回单个模型。
        :param eval_set: 独立验证数据，支持 (X, y)、具名字典、列表、含目标列的 DataFrame。
            仅与 cv=None 配合，最后一集用于选参，其他集合仅监控；不并入最终重训。

        :param n_trials: 本次新增试验数量，默认100。
        :param sample_weight: 训练权重，按训练位置切分；不自动作为选模指标权重。
        :param validation_weight: 最后一个独立评估集的指标权重；内部分割请通过 evaluation_weight 配置。
        :param kwargs: ModelTuner 构造参数，如 direction、retention、fit_params、validation_fraction、
            random_state、n_jobs；timeout/show_progress_bar/evaluation_weight/groups 传给 fit。

        :return: 已拟合最佳模型，best.tuner 可查询参数点、各折效果、曲线和验证预测。
            没有可训练 fit 的组合对象应先对其基础模型或 ProbabilityCalibrator 调参。

        **参考样例**

        >>> from hscredit import ProbabilityScoreCard, LogisticRegression, get_metric
        >>> card = ProbabilityScoreCard(model=LogisticRegression(calculate_stats=False), prefit=False)
        >>> best = card.tune(X_train, y_train, search_space={'model__C': [0.5, 1.0]},
        ...                  metric=get_metric('auc'), eval_set={'验证': (X_valid, y_valid)}, n_trials=2)
        >>> test_values = best.evaluate(X_test, y_test, metrics=['auc', 'ks'])
        >>> report = best.model_report(datasets={'测试': (X_test, y_test)}, verbose=False)
        """
        from .tuning import ModelTuner

        if "validation_data" in kwargs:
            raise TypeError("验证集参数已统一为 eval_set，请使用 eval_set=[(X_valid, y_valid)]")
        if not callable(getattr(self, "fit", None)):
            raise ValueError("当前对象仅组合已训练模型；需要调参时请使用可训练的基础模型或 ProbabilityCalibrator")

        optimize = {
            key: kwargs.pop(key)
            for key in ("timeout", "show_progress_bar", "evaluation_weight", "groups", "catch", "gc_after_trial")
            if key in kwargs
        }
        optimize.update(kwargs.pop("optimize_kwargs", {}))
        tuner = ModelTuner(
            self,
            search_space=search_space,
            metric=metric,
            cv=cv,
            target=getattr(self, "target", None) or "target",
            **kwargs,
        )
        self.tuner = tuner
        tuner.fit(
            X,
            y,
            n_trials=n_trials,
            eval_set=eval_set,
            sample_weight=sample_weight,
            validation_weight=validation_weight,
            **optimize,
        )
        best = tuner.get_best_model()
        best.tuner = tuner
        return best
