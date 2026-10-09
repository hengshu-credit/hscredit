"""冻结基准分布，分别报告缺失和未知类别，不重新拟合监控窗口。"""
import copy
import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ...utils.serialization import ArtifactSerializableMixin


class MonitoringBaseline(ArtifactSerializableMixin, BaseEstimator):
    """单字段的版本化监控基准。

    **参数**
    max_n_bins : int, default=10
        数值字段的等频箱数，使用core.binning.QuantileBinning。
    epsilon : float, default=1e-10
        零占比平滑下限；原始占比仍在输出中保留。
    min_bin_size : float, default=0.01
        数值分箱的最小样本占比。
    binning_params : dict, optional
        传递给QuantileBinning的配置；不支持依赖真实标签的约束。

    **属性**
    version_ : str
        冻结箱定义和基准计数的摘要。

    **参考样例**
    >>> result = MonitoringBaseline().fit([1., 2., 3.]).evaluate([2., None, 5.])
    """
    artifact_kind = "监控基准"

    def __init__(self, max_n_bins=10, epsilon=1e-10, min_bin_size=0.01, binning_params=None, feature_name=None):
        self.max_n_bins = max_n_bins
        self.epsilon = epsilon
        self.min_bin_size = min_bin_size
        self.binning_params = binning_params
        self.feature_name = feature_name

    @staticmethod
    def _series(values):
        if np.ndim(values) != 1:
            raise ValueError("监控字段必须是一维数据")
        return pd.Series(values).reset_index(drop=True)

    @staticmethod
    def _validate_unknown_policy(policy):
        from ..binning._contracts import validate_handle_unknown
        policy = validate_handle_unknown(policy)
        if isinstance(policy, int) and policy < -3:
            raise ValueError("监控未知类别仅支持 -3未知箱、-2特殊箱、-1缺失箱、已有普通箱或raise")
        return policy

    def fit(self, expected, y=None):
        del y
        if isinstance(self.max_n_bins, bool) or not isinstance(self.max_n_bins, (int, np.integer)) or self.max_n_bins < 1:
            raise ValueError("max_n_bins 必须是正整数")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon 必须为有限正数")
        if not isinstance(self.min_bin_size, (int, float, np.number)) or not np.isfinite(self.min_bin_size) or self.min_bin_size <= 0:
            raise ValueError("min_bin_size 必须为有限正数")
        if self.binning_params is not None and not isinstance(self.binning_params, dict):
            raise ValueError("binning_params 必须是字典或None")
        if (self.binning_params or {}).get('min_bad_rate', 0) or (self.binning_params or {}).get('monotonic', False):
            raise ValueError("冻结分布基准不接受依赖目标标签的坏率或单调约束")
        values = self._series(expected)
        if len(values) == 0:
            raise ValueError("监控基准不能为空")
        # 候选状态全部完成后提交；失败重拟合不损坏已有基准。
        candidate = copy.copy(self)
        # 用真实基准重新拟合后，旧制品无法区分缺失/未知的迁移标记不再成立。
        candidate._legacy_missing_ambiguous_ = False
        candidate.epsilon_ = float(self.epsilon)
        candidate.feature_name_ = self.feature_name if self.feature_name is not None else getattr(expected, 'name', None)
        candidate.input_dtype_ = str(values.dtype)
        candidate.binner_ = None
        candidate.bin_specs_ = {}
        candidate.categories_ = None
        candidate.numeric_ = pd.api.types.is_numeric_dtype(values.dtype)
        nonmissing = values.dropna()
        if candidate.numeric_ and not np.isfinite(nonmissing.to_numpy(dtype=float)).all():
            raise ValueError("监控基准不能包含无穷值")
        # 常量/低基数数值保留原生类别，否则单个[-inf,+inf)会隐藏常量迁移。
        user_splits = (self.binning_params or {}).get('user_splits')
        has_splits = user_splits is not None and (not hasattr(user_splits, '__len__') or len(user_splits) > 0)
        if candidate.numeric_ and 0 < nonmissing.nunique() <= self.max_n_bins and not has_splits:
            candidate.numeric_ = False
        from ..binning import QuantileBinning
        options = dict(max_n_bins=self.max_n_bins, min_n_bins=1, force_numerical=candidate.numeric_,
                       n_jobs=1, min_bin_size=self.min_bin_size)
        options.update(self.binning_params or {})
        if isinstance(options.get('user_splits'), (list, tuple, np.ndarray)):
            options['user_splits'] = {'value': options['user_splits']}
        # 对类别/空基准也先验证构造配置，不能静默忽略拼写错误。
        binner = QuantileBinning(**options)
        candidate.unknown_policy_ = candidate._validate_unknown_policy(binner.handle_unknown)
        if (candidate.numeric_ or has_splits) and len(nonmissing):
            candidate.binner_ = binner
            # Quantile算法不使用标签拟合；现有fit契约要求二类标签，仅为适配该契约。
            # 这些标签生成的统计不作为监控指标，基准分布只由真实数据重新计数。
            # 保留真实缺失行，让显式未知→缺失箱策略有可验证的目标箱。
            fit_values = values if len(values) > 1 else pd.concat([values, values], ignore_index=True)
            labels = np.arange(len(fit_values)) % 2
            candidate.binner_.fit(pd.DataFrame({"value": fit_values.to_numpy()}), labels)
            from ..binning.spec import BinSpec
            candidate.bin_specs_ = BinSpec.from_binner(candidate.binner_, 'value')
        elif not candidate.numeric_:
            candidate.categories_ = pd.Index(pd.unique(nonmissing))
        known_values = list(nonmissing.unique()) if not candidate.numeric_ else []
        for spec in candidate.bin_specs_.values():
            if spec.kind == 'categories':
                known_values.extend(spec.values)
                known_values.extend(spec.include_values)
                known_values.extend(spec.known_values)
        candidate.known_categories_ = pd.Index(pd.unique(pd.Series(known_values, dtype=object)))
        if candidate.binner_ is None and candidate.unknown_policy_ != 'raise' and candidate.unknown_policy_ >= 0:
            if candidate.categories_ is None or candidate.unknown_policy_ >= len(candidate.categories_):
                raise ValueError("未知类别指定的普通箱不存在")
        candidate.reference_counts_ = candidate._counts(values)
        candidate.reference_size_ = len(values)
        candidate.reference_missing_count_ = int(values.isna().sum())
        candidate.schema_version_ = 2
        candidate.version_ = candidate._version_fingerprint()
        self.__dict__.update(candidate.__dict__)
        return self

    def _version_fingerprint(self):
        """字段、类型、完整变换和计数共同构成冻结身份，不只比较切点。"""
        inference = {name: getattr(self.binner_, name, None) for name in (
            'splits_', '_cat_bins_', '_category_code_maps_', '_category_orders_',
            '_missing_bin_targets_', '_user_missing_bin_targets_', 'handle_unknown',
            'missing_separate', 'special_codes',
        )}
        payload = (2, self.feature_name_, self.input_dtype_, self.numeric_, self.categories_,
                   self.known_categories_, self.unknown_policy_, self.epsilon_, self.bin_specs_,
                   inference, self.reference_counts_, self.reference_missing_count_, '监控缺失独立计数')
        return hashlib.sha256(pickle.dumps(payload)).hexdigest()[:16]

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault('feature_name', None)
        if hasattr(self, 'reference_counts_') and getattr(self, 'schema_version_', None) != 2:
            self.feature_name_ = getattr(self, 'feature_name_', None)
            self.input_dtype_ = getattr(self, 'input_dtype_', None)
            self.epsilon_ = getattr(self, 'epsilon_', self.epsilon)
            self.bin_specs_ = getattr(self, 'bin_specs_', {})
            self.unknown_policy_ = self._validate_unknown_policy(getattr(self.binner_, 'handle_unknown', -3))
            known = list(self.categories_) if self.categories_ is not None else []
            for spec in self.bin_specs_.values():
                if spec.kind == 'categories':
                    known.extend(spec.values + spec.include_values + spec.known_values)
            known.extend(getattr(self.binner_, '_category_code_maps_', {}).get('value', {}))
            self.known_categories_ = pd.Index(pd.unique(pd.Series(known, dtype=object)))
            self.previous_version_ = getattr(self, 'version_', None)
            self.reference_missing_count_ = self.reference_counts_.get(-1, 0)
            self._legacy_missing_ambiguous_ = self.unknown_policy_ == -1 and self.reference_missing_count_ > 0
            self.schema_version_ = 2
            self.version_ = self._version_fingerprint()

    def transform(self, values):
        """返回固定箱编号；-1为缺失，-3为未知类别或未见数值域。"""
        if not hasattr(self, "reference_counts_"):
            raise ValueError("MonitoringBaseline 尚未拟合")
        return self._codes(self._series(values))

    def _codes(self, values):
        missing = values.isna().to_numpy()
        default_code = -3 if self.unknown_policy_ == 'raise' else self.unknown_policy_
        codes = np.full(len(values), default_code, dtype=np.int64)
        if self.binner_ is not None:
            if self.numeric_:
                converted = pd.to_numeric(values, errors="raise")
                nonmissing = converted[~missing].to_numpy(dtype=float)
                if not np.isfinite(nonmissing).all():
                    raise ValueError("监控数据不能包含无穷值")
            else:
                nonmissing = values[~missing].to_numpy()
            if len(nonmissing):
                codes[~missing] = self.binner_.transform(pd.DataFrame({"value": nonmissing}), metric="indices")["value"].to_numpy(dtype=np.int64)
        elif not self.numeric_:
            known = self.categories_.get_indexer(values[~missing])
            if self.unknown_policy_ == 'raise' and (known < 0).any():
                raise ValueError("监控数据包含冻结词典之外的类别")
            unknown_code = -3 if self.unknown_policy_ == 'raise' else self.unknown_policy_
            codes[~missing] = np.where(known < 0, unknown_code, known)
        elif self.unknown_policy_ == 'raise' and (~missing).any():
            raise ValueError("全缺失基准没有可用于监控的非缺失值域")
        codes[missing] = -1
        return codes

    def _counts(self, values):
        codes, counts = np.unique(self._codes(values), return_counts=True)
        return {int(code): int(count) for code, count in zip(codes, counts)}

    def _label(self, code):
        if code == -1:
            return "缺失及并入类别" if self.unknown_policy_ == -1 else "缺失值"
        if code == -3:
            return "未知类别"
        if code == -2:
            return "特殊值及并入类别" if self.unknown_policy_ == -2 else "特殊值"
        if code < 0:
            raise ValueError(f"不支持的监控保留箱号: {code}")
        if not self.numeric_ and self.binner_ is None:
            return str(self.categories_[code])
        if code in self.bin_specs_:
            return self.bin_specs_[code].label()
        return f"分箱{code}"

    def _unknown_count(self, values):
        if self.numeric_:
            return int(values.notna().sum()) if self.binner_ is None else 0
        return int((values.notna() & ~values.isin(self.known_categories_)).sum())

    def evaluate(self, actual, *, include_missing=True):
        """返回总体/非缺失PSI、缺失与未知比例、状态和明细表。"""
        self.transform([])  # 显式检查拟合状态，不修改基准。
        values = self._series(actual)
        counts = self._counts(values)
        return self._evaluate_counts(counts, len(values), self._unknown_count(values), include_missing, int(values.isna().sum()))

    def evaluate_batches(self, batches, *, include_missing=True):
        """逐块累计监控分布，只保留各箱计数。"""
        self.transform([])
        counts, total, unknown, missing = {}, 0, 0, 0
        for batch in batches:
            values = self._series(batch)
            for code, count in self._counts(values).items():
                counts[code] = counts.get(code, 0) + count
            total += len(values)
            unknown += self._unknown_count(values)
            missing += int(values.isna().sum())
        return self._evaluate_counts(counts, total, unknown, include_missing, missing)

    def _evaluate_counts(self, counts, total, unknown_count=0, include_missing=True, missing_count=0):
        if not isinstance(include_missing, (bool, np.bool_)):
            raise ValueError("include_missing 必须为布尔值")
        if getattr(self, '_legacy_missing_ambiguous_', False):
            raise ValueError("旧基准合并了缺失和未知类别，无法恢复原始缺失计数；请用基准数据重新拟合")
        original_total = total
        original_missing = missing_count
        original_unknown_bin = counts.get(-3, 0)
        reference_counts = dict(self.reference_counts_)
        counts = dict(counts)
        reference_total = self.reference_size_
        if not include_missing:
            reference_total -= self.reference_missing_count_
            total -= original_missing
            for distribution, missing in ((reference_counts, self.reference_missing_count_), (counts, original_missing)):
                if -1 in distribution:
                    distribution[-1] -= missing
                    if distribution[-1] == 0:
                        distribution.pop(-1)
        codes = sorted(set(reference_counts) | set(counts), key=lambda x: (x < 0, x))
        reference = np.asarray([reference_counts.get(code, 0) for code in codes], dtype=float)
        current = np.asarray([counts.get(code, 0) for code in codes], dtype=float)
        p = reference / reference_total if reference_total else np.zeros_like(reference)
        q = current / total if total else np.zeros_like(current)
        safe_p, safe_q = np.maximum(p, self.epsilon_), np.maximum(q, self.epsilon_)
        contributions = (safe_q - safe_p) * np.log(safe_q / safe_p)
        sufficient = bool(total and reference_total)
        if not sufficient:
            contributions[:] = np.nan
        conditional_reference, conditional_current = reference.copy(), current.copy()
        if include_missing and -1 in codes:
            position = codes.index(-1)
            conditional_reference[position] -= self.reference_missing_count_
            conditional_current[position] -= original_missing
        ref_sum, act_sum = conditional_reference.sum(), conditional_current.sum()
        conditional = np.nan
        if ref_sum and act_sum:
            a = np.maximum(conditional_reference / ref_sum, self.epsilon_)
            b = np.maximum(conditional_current / act_sum, self.epsilon_)
            conditional = float(np.sum((b - a) * np.log(b / a)))
        table = pd.DataFrame({"分箱": [self._label(code) for code in codes], "期望样本数": reference.astype(np.int64),
            "实际样本数": current.astype(np.int64), "期望占比": p, "实际占比": q, "PSI贡献": contributions})
        table.attrs.update({"基准版本": self.version_, "分箱口径": "冻结基准分箱", "缺失策略": "独立分箱" if include_missing else "排除",
                            "状态": "成功" if sufficient else "数据不足"})
        return {"PSI": float(contributions.sum()) if sufficient else np.nan, "非缺失PSI": conditional,
            "基准缺失率": self.reference_missing_count_ / self.reference_size_,
            "实际缺失率": original_missing / original_total if original_total else np.nan,
            "未知类别率": unknown_count / original_total if original_total else np.nan,
            "未知箱占比": original_unknown_bin / original_total if original_total else np.nan,
            "有效样本数": total, "总样本数": original_total, "基准版本": self.version_, "状态": table.attrs["状态"], "分箱明细": table}
