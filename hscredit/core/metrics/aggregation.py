"""固定分箱的可合并二分类充分统计，不保留逐行数据。"""

from dataclasses import dataclass
import numpy as np
from ._binning import _binary_stats_from_counts, _validate_bin_arrays


@dataclass(frozen=True)
class MetricSpec:
    """统计口径；本累加器按行计数，不将金额权重混称样本数。

    :param target_type: 目标类型标识；当前固定分箱累加器使用 binary 二分类。 默认 ``'binary'``。
    :param positive_label: 坏样本标签标识；当前二分类累加器使用 1。 默认 ``1``。
    :param grain: 统计粒度说明；当前累加器按行计件数。 默认 ``'行'``。
    :param feature: 特征名称标识，用于核对合并口径。 默认 ``None``。
    :param target: 目标名称标识，用于核对合并口径。 默认 ``None``。
    :param binning_version: 固定分箱语义版本，合并时必须一致。 默认 ``None``。
    """

    target_type: str = "binary"
    positive_label: int = 1
    grain: str = "行"
    feature: object = None
    target: object = None
    binning_version: object = None


class BinStatsAccumulator:
    """流式累计固定整数分箱的好坏样本数。

    每个批次必须使用同一套分箱规则；累计器不重新拟合切点，不保留逐行数据。

    **参数**

    :param feature: 特征名，默认 ``None`` ；严格模式下须为非空字符串。
    :param target: 目标名称，默认 ``None`` ；严格模式下须为非空字符串。
    :param binning_version: 分箱语义版本，默认 ``None`` ；严格模式下须为非空字符串。
    :param strict_identity: 默认 ``False`` ，由调用方保证同一口径； ``True`` 要求三个标识齐全。
        合并时无论是否严格模式，都必须具有相同统计口径。

    **属性**

    - ``rows_seen_`` ：已累计的总行数。
    - ``counts_`` ：每个整数箱号对应的好、坏样本计数。

    **参考样例**

    >>> from hscredit.core.metrics import BinStatsAccumulator
    >>> table = BinStatsAccumulator().update([0, 1, 0, 1], [0, 1, 1, 0]).finalize()
    """

    def __init__(self, *, feature=None, target=None, binning_version=None, strict_identity=False):
        if not isinstance(strict_identity, bool):
            raise ValueError("strict_identity 必须为布尔值")
        if strict_identity and any(not isinstance(v, str) or not v for v in (feature, target, binning_version)):
            raise ValueError("严格合并模式必须声明字段、目标和分箱语义版本")
        self.strict_identity = strict_identity
        self.spec = MetricSpec(feature=feature, target=target, binning_version=binning_version)
        self.counts_ = {}
        self.rows_seen_ = 0
        self.bin_dtype_ = None

    def update(self, bins, y):
        """累计一个数据批次。

        **参数**

        :param bins: 固定规则得到的一维整数箱号，与 ``y`` 等长。
        :param y: 一维二分类标签，只允许 ``0`` 和 ``1`` ， ``1`` 表示坏样本。

        :return: 累计器自身。
        """
        bins, y = _validate_bin_arrays(bins, y)
        self.bin_dtype_ = bins.dtype if self.bin_dtype_ is None else np.result_type(self.bin_dtype_, bins.dtype)
        codes, inverse = np.unique(bins, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(codes))
        bad = np.bincount(inverse, weights=y, minlength=len(codes)).astype(np.int64)
        for code, count, bad_count in zip(codes, counts, bad):
            good_old, bad_old = self.counts_.get(int(code), (0, 0))
            self.counts_[int(code)] = (good_old + int(count - bad_count), bad_old + int(bad_count))
        self.rows_seen_ += len(y)
        return self

    def merge(self, other):
        """合并同口径累计器。

        **参数**

        :param other: 具有相同 MetricSpec 的另一累计器；其统计加入当前累计器。

        :return: 累计器自身。
        """
        if not isinstance(other, BinStatsAccumulator) or self.spec != other.spec:
            raise ValueError("只能合并相同字段、目标和分箱版本的统计")
        if other.bin_dtype_ is not None:
            self.bin_dtype_ = (
                other.bin_dtype_ if self.bin_dtype_ is None else np.result_type(self.bin_dtype_, other.bin_dtype_)
            )
        for code, (good, bad) in list(other.counts_.items()):
            old_good, old_bad = self.counts_.get(code, (0, 0))
            self.counts_[code] = (old_good + good, old_bad + bad)
        self.rows_seen_ += other.rows_seen_
        return self

    def finalize(self, *, epsilon=1e-10, round_digits=True, woe_clip=None):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon 必须为有限正数")
        if woe_clip is not None and (not np.isfinite(woe_clip) or woe_clip <= 0):
            raise ValueError("woe_clip 必须为有限正数或None")
        codes = sorted(self.counts_, key=lambda code: (2 if code == -2 else 1 if code == -1 else 0, code))
        counts = np.asarray([self.counts_[code] for code in codes], dtype=np.float64).reshape(-1, 2)
        return _binary_stats_from_counts(
            np.asarray(codes, dtype=self.bin_dtype_ or np.int64),
            counts[:, 0],
            counts[:, 1],
            epsilon,
            None,
            round_digits,
            woe_clip,
        )
