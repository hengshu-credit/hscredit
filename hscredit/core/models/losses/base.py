"""损失、离线评估和训练回调的统一接口。"""

from abc import ABC, abstractmethod
from copy import deepcopy
from inspect import signature
from typing import Optional, Tuple

import numpy as np
from scipy.special import expit


def _validate_binary_inputs(y_true, y_pred, sample_weight=None, raw_score=False):
    """校验二分类标签、概率和权重；显式转换原始分数。"""
    try:
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("标签和预测值必须是数值数组") from exc
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if p.ndim == 2 and p.shape[1] == 1:
        p = p[:, 0]
    elif p.ndim == 2 and p.shape[1] == 2 and not raw_score:
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("两列概率必须对应类别[0, 1]，每行非负且合计为1")
        p = p[:, 1]
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("标签和预测值必须是一维、非空且等长的数组")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("标签必须是0（好样本）或1（坏样本）")
    if not np.isfinite(p).all():
        raise ValueError("预测值必须为有限数，不能包含缺失值或无穷大")
    if raw_score:
        p = expit(p)
    elif np.any((p < 0) | (p > 1)):
        raise ValueError("预测概率必须在[0, 1]内；原始分数请指定 raw_score=True")
    weight = None
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.ndim != 1 or len(weight) != len(y):
            raise ValueError("样本权重必须是一维且与标签等长")
        if not np.isfinite(weight).all() or np.any(weight < 0) or not np.isfinite(weight.sum()) or weight.sum() <= 0:
            raise ValueError("样本权重必须有限、非负且总和大于0")
    return y, p, weight


def _dataset_weight(data):
    """读取原生数据集权重；空数组表示未设置。"""
    weight = data.get_weight() if hasattr(data, "get_weight") else None
    return None if weight is None or len(weight) == 0 else weight


def _margin_derivatives(loss, y_true, probability) -> Tuple[np.ndarray, np.ndarray]:
    """将概率导数转换为原始分数导数；不在这里修改真实 Hessian。"""
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    grad_probability = np.asarray(loss.gradient(y_true, probability), dtype=float)
    hess_probability = loss.hessian(y_true, probability)
    link_grad = probability * (1.0 - probability)
    grad_margin = grad_probability * link_grad
    if hess_probability is None:
        hess_margin = np.maximum(link_grad, np.finfo(float).eps)
    else:
        link_hess = link_grad * (1.0 - 2.0 * probability)
        hess_margin = np.asarray(hess_probability, dtype=float) * link_grad**2 + grad_probability * link_hess
    return grad_margin, hess_margin


def _objective_derivatives(loss, y_true, margin, sample_weight=None):
    """训练入口：验证输入并为树模型提供正对角曲率近似。"""
    y, p, weight = _validate_binary_inputs(y_true, margin, sample_weight, raw_score=True)
    if weight is not None and type(loss).loss_values is BaseLoss.loss_values:
        raise ValueError("该损失依赖整体样本分布，尚不支持额外样本权重；请使用未加权目标或逐样本损失")
    grad, hess = _margin_derivatives(loss, y, p)
    if grad.shape != y.shape or hess.shape != y.shape or not np.isfinite(grad).all() or not np.isfinite(hess).all():
        raise ValueError("损失函数的梯度和二阶导必须与标签等长且为有限数")
    # 非凸损失使用正对角近似，不对真实曲率取绝对值。
    hess = np.maximum(hess, 1e-6)
    if weight is not None:
        normalizer = loss._external_weight_normalizer(y, weight)
        grad, hess = grad * weight / normalizer, hess * weight / normalizer
    return grad, hess


class BaseLoss(ABC):
    """二分类损失基类。

    **参数**

    :param name: 损失的标识名称。

    **属性**

    ``gradient`` / ``hessian`` 对概率求导，标度为 ``样本数 × 平均损失``；
    树框架适配器统一转换到原始分数。``is_additive`` 表示可安全分批计算。

    **参考样例**

    >>> from hscredit.core.models.losses import FocalLoss
    >>> loss = FocalLoss(alpha=0.75)
    >>> metric = loss.metric()
    >>> value = metric([0, 1], [0.2, 0.8])
    >>> scorer = metric.to_scorer()
    """

    is_additive = False

    def __init__(self, name: str = "custom_loss"):
        self.name = name

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算未加额外样本权重的平均损失。

        :param y_true: 一维 0/1 标签，1 表示坏样本，与预测等长。
        :param y_pred: 一维坏样本概率，取值在 [0, 1]；不接收原始分数。
        :return: float，损失越小越好；不对结果取负。

        原始分数、两列概率或额外样本权重请用 ``evaluate``。
        子类必须实现本方法，并让 ``gradient`` / ``hessian`` 与其数学定义一致。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> value = FocalLoss()([0, 1], [0.2, 0.8])
        """

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """返回对坏样本概率的一阶导数，用于实现训练目标。

        :param y_true: 形状 (n_samples,) 的 0/1 标签。
        :param y_pred: 同形状的坏样本概率，通常在数值计算前裁剪至 (0, 1)。
        :return: 同形状有限数组，定义为 ``d(n_samples * loss) / dp``；
            逐样本可加损失等于各行损失的导数，不再除以样本数。

        此处对概率 p 求导。框架适配器负责 sigmoid 链式求导与样本权重，
        自定义子类不应重复转换为原始分数导数。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> gradient = FocalLoss().gradient([0, 1], [0.2, 0.8])
        """

    def hessian(self, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[np.ndarray]:
        """返回对概率的二阶导数，或 None 以请求训练曲率近似。

        :param y_true: 一维 0/1 标签。
        :param y_pred: 同形状坏样本概率。
        :return: ``d²(n_samples * loss) / dp²`` 的对角项数组，或 None。
            原始数学曲率可为负；树框架适配器转换后会采用正对角近似。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> curvature = FocalLoss().hessian([0, 1], [0.2, 0.8])
        """
        return None

    def loss_values(self, y_true, y_pred):
        """计算逐样本损失贡献，用于加权评估或可安全分批的训练。

        :param y_true: 一维 0/1 标签。
        :param y_pred: 同形状坏样本概率。
        :return: 一维数组，其普通均值应等于 ``self(y_true, y_pred)``。
        :raises NotImplementedError: 损失依赖完整排序或整体样本分布，无法逐行分解。

        自定义损失只有在每行计算不依赖其他行时才应设置 ``is_additive=True``。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> values = FocalLoss().loss_values([0, 1], [0.2, 0.8])
        """
        raise NotImplementedError("该损失依赖整体样本分布，不支持逐样本损失或额外样本权重")

    def _external_weight_normalizer(self, y_true, sample_weight):
        """额外权重与损失内部归一化权重相乘时的校正量。"""
        return 1.0

    def metric(self, name=None, **sample_params):
        """生成与本损失公式一致的评估指标，越小越好。

        :param name: 可选的中文指标名称，省略时使用内置名称。
        :param sample_params: 替换指标副本中的逐行参数，例如
            ``AmountWeightedLoss.metric(amounts=valid_amounts)``。
            仅支持提供 ``set_sample_params`` 的损失，行顺序须与评估数据一致。
        :return: 独立的 ``LossMetric`` 参数快照，可离线调用、传入 ModelTuner，
            或转换为 sklearn 评分器、框架监控回调。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> loss = FocalLoss(alpha=0.75)
        >>> metric = loss.metric(name='验证集聚焦损失')
        >>> metric.direction
        'minimize'
        >>> value = metric([0, 1], [0.2, 0.8])
        """
        return LossMetric(self, name=name, **sample_params)

    def to_metric(self, name=None, **sample_params):
        """``metric`` 的同义方法，参数、快照与返回语义完全相同。

        :param name: 可选的中文指标名称。
        :param sample_params: 评估数据对应的样本金额或金融参数。
        :return: ``LossMetric`` 实例。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> metric = FocalLoss().to_metric()
        """
        return self.metric(name=name, **sample_params)

    def business_metric(self):
        """获取与损失目标对应的实际业务指标。

        :return: ``BaseMetric`` 实例，继承本损失的头部比例、成本等业务参数。
            排序目标对应 AUC，分离目标对应 KS，审批目标对应坏账率、通过率
            或利润等。没有专用映射时返回本损失的 ``LossMetric``，
            例如 FocalLoss 与 AmountWeightedLoss。详见 ``business_metric_for_loss``。

        损失值用于衡量代理优化目标，业务指标用于衡量实际模型表现，
        两者可能有不同的最优模型及优化方向，应通过 ``metric.direction`` 判断。

        >>> from hscredit.core.models.losses import RankingAUCProxyLoss
        >>> metric = RankingAUCProxyLoss().business_metric()
        >>> metric([0, 1], [0.2, 0.8])
        1.0
        """
        from .business_metrics import business_metric_for_loss

        return business_metric_for_loss(self)

    def evaluate(self, y_true, y_pred, sample_weight=None, *, raw_score=False, **sample_params):
        """校验输入并计算离线损失，建议作为统一评估入口。

        :param y_true: 一维 0/1 标签，1 为坏样本。
        :param y_pred: 一维坏样本概率，或类别顺序 [0, 1] 的两列概率。
            ``raw_score=True`` 时改为一维原始分数（logit）。
        :param sample_weight: 可选的一维非负样本权重，须有限、等长且总和大于 0。
            只有可逐行分解的损失支持额外权重；不得用金额数组替代样本行对齐。
        :param raw_score: 是否先将一维原始分数转换为坏样本概率，默认 False。
        :param sample_params: 当前评估数据的金额或金融参数，语义同 ``metric``。
        :return: float，原始损失值，越小越好。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> loss = FocalLoss()
        >>> value = loss.evaluate([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        >>> weighted = loss.evaluate([0, 1], [0.2, 0.8], sample_weight=[1, 2])
        """
        return self.metric(**sample_params).evaluate(y_true, y_pred, sample_weight=sample_weight, raw_score=raw_score)

    def to_scorer(self):
        """生成 sklearn 评分器，自动对损失取负以符合越大越好的约定。

        :return: ``scorer(estimator, X, y_true, sample_weight=None)`` 可调用对象。
            模型须有 ``predict_proba``，类别为 0 和 1。
        :raises ValueError: 已绑定逐行金额等数组，无法在 sklearn CV 中自动切分。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> from sklearn.model_selection import cross_val_score
        >>> scorer = FocalLoss().to_scorer()
        >>> scores = cross_val_score(model, X, y, scoring=scorer, cv=3)  # doctest: +SKIP
        """
        return self.metric().to_scorer()

    def to_xgboost(self, api="native"):
        """生成 XGBoost 自定义目标，自动处理原始分数、链接函数和权重。

        :param api: ``'native'`` 返回 ``objective(preds, DMatrix)``，用于
            ``xgboost.train(obj=...)``；``'sklearn'`` 返回
            ``objective(y_true, y_pred, sample_weight=None)``，用于 XGBClassifier。
        :return: 返回 ``(grad, hess)`` 两个等长数组的目标回调。
            输入预测始终为原始分数；非凸目标采用正对角 Hessian 近似。
        :raises ValueError: api 无效，或跨样本损失被传入额外权重。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> from xgboost import XGBClassifier
        >>> loss = FocalLoss(alpha=0.75)
        >>> model = XGBClassifier(objective=loss.to_xgboost(api='sklearn'))
        """
        if api == "native":

            def objective(preds, data):
                return _objective_derivatives(self, data.get_label(), preds, _dataset_weight(data))

        elif api == "sklearn":

            def objective(y_true, y_pred, sample_weight=None):
                return _objective_derivatives(self, y_true, y_pred, sample_weight)

        else:
            raise ValueError("api 必须是 'native' 或 'sklearn'")
        return objective

    def to_lightgbm(self, api="sklearn"):
        """生成 LightGBM 自定义目标。

        :param api: 默认 ``'sklearn'``，用于 ``LGBMClassifier(objective=...)``；
            ``'native'`` 用于 ``lightgbm.train`` 的 params['objective']。
        :return: 返回 (grad, hess) 的回调；原始分数、权重与曲率语义同 ``to_xgboost``。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> from lightgbm import LGBMClassifier
        >>> model = LGBMClassifier(objective=FocalLoss().to_lightgbm())
        """
        return self.to_xgboost(api=api)

    def to_catboost(self):
        """生成 CatBoost 自定义目标，自动提供 CatBoost 约定的负导数。

        :return: 带 ``calc_ders_range(approxes, targets, weights)`` 的目标对象。
            接收原始分数，返回逐行 (-grad, -hess)。
        :raises ValueError: 损失不能安全分批计算，例如全量排序或绑定金额数组。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> from catboost import CatBoostClassifier
        >>> model = CatBoostClassifier(loss_function=FocalLoss().to_catboost(), eval_metric='AUC')
        """
        from .adapters import CatBoostLossAdapter

        return CatBoostLossAdapter(self).objective()

    def to_ngboost(self):
        """生成 NGBoost Score；必须同时使用返回类的 distribution。

        :return: NGBoost Score 子类，``score_class.distribution`` 是配套的
            Bernoulli 分布子类；训练接收标签，内部自行读取坏样本概率。
        :raises ImportError: NGBoost 未安装。
        :raises ValueError: 损失不能安全地逐行、分批计算。

        推荐 ``ngboost_params`` 一并配置 Score 和 Dist，避免遗漏。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> from ngboost import NGBClassifier
        >>> score_class = FocalLoss().to_ngboost()
        >>> model = NGBClassifier(Score=score_class, Dist=score_class.distribution)
        """
        if not self.is_additive:
            raise ValueError("NGBoost仅支持可独立逐样本计算的损失；请使用固定权重的 FocalLoss 或 WeightedBCELoss")
        try:
            from ngboost.scores import Score
            from ngboost.distns import Bernoulli
        except ImportError as exc:
            raise ImportError("NGBoost未安装，请使用 pip install ngboost 安装") from exc
        loss = deepcopy(self)

        class CustomScore(Score):
            def score(self, y):
                return loss.loss_values(np.asarray(y), np.clip(self.probs[1], 1e-7, 1 - 1e-7))

            def d_score(self, y):
                p = np.clip(self.probs[1], 1e-7, 1 - 1e-7)
                return (loss.gradient(np.asarray(y), p) * p * (1 - p))[:, None]

            def metric(self):
                # Bernoulli 分布的 Fisher 信息用于自然梯度预条件。
                p = np.clip(self.probs[1], 1e-7, 1 - 1e-7)
                return (p * (1 - p))[:, None, None]

        class CustomBernoulli(Bernoulli):
            scores = [CustomScore]

            @classmethod
            def implementation(cls, score, scores=None):
                if score is CustomScore:
                    return CustomScore
                return Bernoulli.implementation(score, scores=scores)

        CustomScore.__name__ = f"NGBoost_{self.name}"
        CustomScore.distribution = CustomBernoulli
        return CustomScore

    def ngboost_params(self):
        """生成可直接解包给 NGBClassifier 的配套参数。

        :return: 字典，包含相互兼容的 ``Score`` 和 ``Dist`` 类。
        :raises ImportError: NGBoost 未安装。
        :raises ValueError: 损失不支持独立逐行计算。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> from ngboost import NGBClassifier
        >>> model = NGBClassifier(**FocalLoss().ngboost_params())
        """
        score = self.to_ngboost()
        return {"Score": score, "Dist": score.distribution}


class BaseMetric(ABC):
    """可用于离线评估、训练监控及调参的统一指标。

    **参数**

    :param name: 指标名称。
    :param greater_is_better: True 表示越大越好。

    **属性**

    direction 可直接用于 Optuna 的 maximize / minimize。

    **参考样例**

    >>> from hscredit.core.models.losses import FocalLoss
    >>> metric = FocalLoss().metric()
    >>> value = metric.evaluate([0, 1], [0.1, 0.9])
    >>> scorer = metric.to_scorer()

    自定义指标只需实现 ``__call__(y_true, y_pred, sample_weight=None)`` 并
    指定优化方向；不支持权重时可省略该参数，传权重会显式报错。
    离线场景优先调用 ``evaluate`` 以获取输入和返回值校验。
    """

    def __init__(self, name="自定义指标", greater_is_better=True):
        self.name = name
        self.greater_is_better = greater_is_better

    @property
    def direction(self):
        """返回 ``'maximize'`` 或 ``'minimize'``，可直接传给 Optuna。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> FocalLoss().metric().direction
        'minimize'
        """
        return "maximize" if self.greater_is_better else "minimize"

    @abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        """计算未改变符号的指标值，子类必须实现。

        :param y_true: 形状 (n_samples,) 的 0/1 标签，1 表示坏样本。
        :param y_pred: 同形状的坏样本概率，范围 [0, 1]。
        :return: 有限 float；优劣方向由 ``greater_is_better`` 指定。

        若子类支持权重，应增加 ``sample_weight=None`` 参数；输入原始分数
        或两列概率时统一使用 ``evaluate``，避免回调协议与离线协议混淆。
        """

    def evaluate(self, y_true, y_pred, sample_weight=None, *, raw_score=False):
        """离线评估统一入口，校验标签、概率、权重和返回值。

        :param y_true: 一维、非空 0/1 标签。
        :param y_pred: 一维坏样本概率，或列顺序 [0, 1] 的两列概率。
            ``raw_score=True`` 时接收一维 logit，并先做 sigmoid。
        :param sample_weight: 可选一维非负有限数组，等长且总和大于 0；
            指标子类必须声明支持此参数。
        :param raw_score: 是否将一维原始分数转为坏样本概率，默认 False。
        :return: float，原始指标值，不根据优化方向改变符号。
        :raises ValueError: 输入无效、权重不支持，或指标返回非有限值。

        >>> from hscredit.core.models.losses import AUCMetric
        >>> AUCMetric().evaluate([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        1.0
        >>> AUCMetric().evaluate([0, 1], [-2.0, 2.0], raw_score=True)
        1.0
        """
        y, p, weight = _validate_binary_inputs(y_true, y_pred, sample_weight, raw_score)
        if weight is None:
            value = self(y, p)
        elif "sample_weight" in signature(self.__call__).parameters:
            value = self(y, p, sample_weight=weight)
        else:
            raise ValueError(f"指标 {self.name} 不支持样本权重")
        if not np.isfinite(value):
            raise ValueError(f"指标 {self.name} 返回了非有限值，请检查样本和参数")
        return float(value)

    def to_scorer(self):
        """生成 sklearn CV 或 GridSearchCV 使用的评分器。

        :return: 独立参数快照，签名为
            ``scorer(estimator, X, y_true, sample_weight=None)``。
            根据 ``estimator.classes_`` 选择类别 1 的预测概率；仅在此处
            对越小越好的指标取负，保持 sklearn 越大越好的约定。

        >>> from hscredit.core.models.losses import AUCMetric
        >>> from sklearn.model_selection import GridSearchCV
        >>> search = GridSearchCV(model, {'C': [0.1, 1]}, scoring=AUCMetric().to_scorer())  # doctest: +SKIP
        """
        return _MetricScorer(deepcopy(self))

    def to_xgboost(self, api="native", *, raw_score=False):
        """生成 XGBoost 指标回调。

        :param api: ``'native'`` 用于 ``xgboost.train(custom_metric=...)``；
            ``'sklearn'`` 用于 ``XGBClassifier(eval_metric=...)``。
        :param raw_score: 使用自定义训练目标时设为 True；内置 logistic 目标
            通常传来概率，保持 False。转换只做一次，不能同时外部 sigmoid。
        :return: native 回调 ``metric(preds, DMatrix) -> (name, value)``；
            sklearn 回调 ``metric(y_true, y_pred, sample_weight=None) -> value``。

        原生 train 还需 maximize=metric.greater_is_better；sklearn 早停需配置
        相同方向的 EarlyStopping 回调。HSCredit 模型可自动完成这些配置。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> callback = FocalLoss().metric().to_xgboost(raw_score=True)
        >>> sklearn_callback = FocalLoss().metric().to_xgboost(api='sklearn', raw_score=True)
        """
        if api == "native":

            def metric(preds, data):
                return self.name, self.evaluate(data.get_label(), preds, _dataset_weight(data), raw_score=raw_score)

        elif api == "sklearn":

            def metric(y_true, y_pred, sample_weight=None):
                return self.evaluate(y_true, y_pred, sample_weight, raw_score=raw_score)

        else:
            raise ValueError("api 必须是 'native' 或 'sklearn'")
        metric.__name__ = self.name
        return metric

    def to_lightgbm(self, api="sklearn", *, raw_score=False):
        """生成 LightGBM 指标回调，返回值自动携带优化方向。

        :param api: ``'sklearn'`` 用于 ``LGBMClassifier.fit(eval_metric=...)``；
            ``'native'`` 用于 ``lightgbm.train(feval=...)``。
        :param raw_score: 与自定义目标一起使用时设 True，内置概率目标用 False。
        :return: 返回 ``(name, value, greater_is_better)`` 的回调。
            native 参数为 (preds, Dataset)，sklearn 为 (y_true, y_pred, weight=None)。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> callback = FocalLoss().metric().to_lightgbm(raw_score=True)
        >>> native_callback = FocalLoss().metric().to_lightgbm(api='native', raw_score=True)
        """
        if api == "native":

            def metric(preds, data):
                value = self.evaluate(data.get_label(), preds, _dataset_weight(data), raw_score=raw_score)
                return self.name, value, self.greater_is_better

        elif api == "sklearn":

            def metric(y_true, y_pred, weight=None):
                value = self.evaluate(y_true, y_pred, weight, raw_score=raw_score)
                return self.name, value, self.greater_is_better

        else:
            raise ValueError("api 必须是 'native' 或 'sklearn'")
        metric.__name__ = self.name
        return metric

    def to_catboost(self):
        """生成 CatBoost 指标对象，自动将原始分数转换为概率。

        :return: 提供 evaluate、get_final_error、is_max_optimal 的对象。
            evaluate 返回 (指标值 × 权重和, 权重和)，最终结果还原为指标值。
            标记为不可加，以保留 AUC、KS、排序和逐行金额指标的全量评估语义。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> callback = FocalLoss().metric().to_catboost()
        >>> callback.is_max_optimal()
        False
        """
        metric = self

        class CatBoostMetric:
            def is_additive(self):
                # 全量评价才能保持 AUC、KS、排序和绑定金额数组的正确语义。
                return False

            def evaluate(self, approxes, target, weight):
                if len(approxes) != 1:
                    raise ValueError("当前指标仅支持CatBoost二分类预测")
                value = metric.evaluate(target, approxes[0], weight, raw_score=True)
                mass = float(len(target) if weight is None else np.sum(weight))
                return value * mass, mass

            def get_final_error(self, error, weight):
                return error / weight if weight > 0 else 0.0

            def is_max_optimal(self):
                return metric.greater_is_better

        return CatBoostMetric()


_LOSS_LABELS = {
    "FocalLoss": "聚焦损失",
    "AsymmetricFocalLoss": "非对称聚焦损失",
    "BalancedFocalLoss": "平衡聚焦损失",
    "WeightedBCELoss": "加权交叉熵",
    "CostSensitiveLoss": "成本敏感损失",
    "BadDebtLoss": "坏账代理损失",
    "ApprovalRateLoss": "通过率代理损失",
    "ProfitMaxLoss": "利润代理损失",
    "ExpectedProfitLoss": "期望利润损失",
    "OrdinalRankLoss": "序数排序损失",
    "LiftFocusedLoss": "头部提升损失",
    "RankingAUCProxyLoss": "排序面积代理损失",
    "KSFocusedLoss": "分布分离损失",
    "TopKBadCaptureLoss": "头部捕获损失",
    "AmountWeightedLoss": "金额加权损失",
    "ExpectedValueLoss": "期望价值损失",
}


class LossMetric(BaseMetric):
    """每一种 BaseLoss 的配套评估指标。

    **参数**

    :param loss: 损失实例；创建时复制其参数，评估不会改变训练对象。
    :param name: 自定义指标名，默认采用中文名称。
    :param sample_params: 替换副本的样本参数，例如 ``amounts=valid_amounts``；
        须与验证集逐行对应，支持范围由损失的 ``set_sample_params`` 决定。

    **属性**

    loss 保存参数快照；greater_is_better=False。

    **参考样例**

    >>> from hscredit.core.models.losses import FocalLoss
    >>> loss = FocalLoss(alpha=0.75)
    >>> metric = loss.metric()
    >>> metric([0, 1], [0.2, 0.8]) == loss.evaluate([0, 1], [0.2, 0.8])
    True
    """

    def __init__(self, loss, name=None, **sample_params):
        if not isinstance(loss, BaseLoss):
            raise TypeError("loss 必须是 BaseLoss 实例")
        super().__init__(name or _LOSS_LABELS.get(type(loss).__name__, "自定义损失"), greater_is_better=False)
        self.loss = deepcopy(loss)
        if sample_params:
            if not hasattr(self.loss, "set_sample_params"):
                raise ValueError("该损失不支持样本参数；请在构造损失时设置模型参数")
            self.loss.set_sample_params(**deepcopy(sample_params))

    def __call__(self, y_true, y_pred, sample_weight=None):
        """计算与损失公式一致的离线指标，保持越小越好的原始数值。

        :param y_true: 一维 0/1 标签，1 表示坏样本。
        :param y_pred: 坏样本概率，或类别顺序 [0, 1] 的两列概率。
        :param sample_weight: 可选非负样本权重；跨样本损失不支持额外权重。
        :return: float，损失的普通或加权平均值，含内部权重的归一化校正。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> value = FocalLoss().metric()([0, 1], [0.2, 0.8], sample_weight=[1, 2])
        """
        y, p, weight = _validate_binary_inputs(y_true, y_pred, sample_weight)
        if weight is None:
            value = self.loss(y, p)
        else:
            values = np.asarray(self.loss.loss_values(y, p), dtype=float)
            if values.shape != y.shape:
                raise ValueError("逐样本损失必须与标签等长")
            value = np.average(values, weights=weight) / self.loss._external_weight_normalizer(y, weight)
        if not np.isfinite(value):
            raise ValueError(f"指标 {self.name} 返回了非有限值，请检查样本和参数")
        return float(value)

    def to_scorer(self):
        """生成 sklearn 评分器，并检查样本参数能否安全用于交叉验证。

        :return: ``scorer(estimator, X, y_true, sample_weight=None)``，输出负损失。
        :raises ValueError: 指标绑定 amounts、lgd、ead、rate 或 cost 的逐行数组；
            应在每折内切分对应数组并重新构造指标，或用 ModelTuner 的折内适配。

        >>> from hscredit.core.models.losses import FocalLoss
        >>> scorer = FocalLoss().metric().to_scorer()
        """
        for key in ("amounts_", "lgd", "ead", "rate", "cost"):
            value = getattr(self.loss, key, None)
            if value is not None and np.asarray(value).ndim > 0:
                raise ValueError("绑定样本金额或金融参数数组的指标不能直接用于交叉验证；请在每一折内按行切分并构造指标")
        return super().to_scorer()


class _MetricScorer:
    """可序列化、显式选择坏样本概率的 sklearn 评分器。"""

    def __init__(self, metric):
        self.metric = metric

    def __call__(self, estimator, X, y_true, sample_weight=None):
        classes = np.asarray(getattr(estimator, "classes_", []))
        if classes.shape != (2,) or set(classes.tolist()) != {0, 1}:
            raise ValueError("评分器要求模型类别为0（好样本）和1（坏样本）")
        prediction = np.asarray(estimator.predict_proba(X))
        if prediction.ndim != 2 or prediction.shape[1] != 2:
            raise ValueError("评分器要求 predict_proba 返回与类别对应的两列概率")
        prediction = prediction[:, np.argsort(classes)]
        value = self.metric.evaluate(y_true, prediction, sample_weight)
        return value if self.metric.greater_is_better else -value
