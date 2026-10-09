"""固定分箱的可合并二分类充分统计，不保留逐行数据。"""
from dataclasses import dataclass
import numpy as np
from ._binning import _binary_stats_from_counts, _validate_bin_arrays


@dataclass(frozen=True)
class MetricSpec:
    """统计口径；本累加器按行计数，不将金额权重混称样本数。"""
    target_type: str = "binary"
    positive_label: int = 1
    grain: str = "行"
    feature: object = None
    target: object = None
    binning_version: object = None


class BinStatsAccumulator:
    """流式累计固定整数分箱的好坏样本数。

    **参数**
    feature, target, binning_version : str, optional
        字段、目标和分箱语义版本。update不重新拟合切点。
    strict_identity : bool, default=False
        True要求上述三个标识齐全，合并时拒绝不同分箱；False兼容由调用方保证口径的旧用法。

    **属性**
    rows_seen_ : int
        已累计行数。

    **参考样例**
    >>> table = BinStatsAccumulator().update([0, 1], [0, 1]).finalize()
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
        if not isinstance(other, BinStatsAccumulator) or self.spec != other.spec:
            raise ValueError("只能合并相同字段、目标和分箱版本的统计")
        if other.bin_dtype_ is not None:
            self.bin_dtype_ = other.bin_dtype_ if self.bin_dtype_ is None else np.result_type(self.bin_dtype_, other.bin_dtype_)
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
        return _binary_stats_from_counts(np.asarray(codes, dtype=self.bin_dtype_ or np.int64), counts[:, 0], counts[:, 1],
                                        epsilon, None, round_digits, woe_clip)
