"""MDLP前缀计数与旧逐候选扫描的独立差分基线。"""

import numpy as np
import pandas as pd
import pytest

from hscredit.core.binning import MDLPBinning


class LegacyScanMDLP(MDLPBinning):
    """仅用于差分/基准，冻结优化前的候选评分顺序。"""

    def _find_best_split_v3(self, x, y, candidates):
        best_split, best_idx, best_score = None, None, -np.inf
        total_bad, total_good = y.sum(), len(y) - y.sum()
        for index in candidates:
            if index < self.min_samples_leaf or len(x) - index < self.min_samples_leaf:
                continue
            score = self._calculate_iv_gain_v3(y[:index], y[index:], total_bad, total_good)
            if score < self.min_iv_gain:
                continue
            if score > best_score:
                best_score, best_idx = score, index
                best_split = (x[index - 1] + x[index]) / 2
        return best_split, best_idx

    def _force_additional_splits_v3(self, x, y, existing_splits, all_candidates):
        splits = list(existing_splits)
        total_bad, total_good = y.sum(), len(y) - y.sum()
        minimum = self._get_min_samples(len(x))
        candidates = []
        for index in all_candidates:
            if index < minimum or len(x) - index < minimum:
                continue
            score = self._calculate_iv_gain_v3(y[:index], y[index:], total_bad, total_good)
            candidates.append(((x[index - 1] + x[index]) / 2, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        while len(splits) < self.max_n_bins - 1:
            best_split, best_iv = None, -np.inf
            for split, _ in candidates:
                if split in splits:
                    continue
                bins = np.digitize(x, sorted(splits + [split]))
                value = 0.0
                for bin_id in range(len(splits) + 2):
                    mask = bins == bin_id
                    bad, good = y[mask].sum(), mask.sum() - y[mask].sum()
                    if bad > 0 and good > 0:
                        br, gr = bad / total_bad, good / total_good
                        value += (gr - br) * np.log((gr + 1e-10) / (br + 1e-10))
                if value > best_iv:
                    best_iv, best_split = value, split
            if best_split is None:
                for split in np.linspace(x.min(), x.max(), self.max_n_bins + 1)[1:-1]:
                    if split not in splits:
                        splits.append(split)
                        if len(splits) >= self.max_n_bins - 1:
                            break
                break
            splits.append(best_split)
        return sorted(set(splits))


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("discrete", [False, True])
def test_mdlp_prefix_recursion_and_force_match_legacy(seed, discrete):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.integers(0, 20, 120).astype(float) if discrete else rng.normal(size=120))
    y = rng.integers(0, 2, len(x))
    new = MDLPBinning(max_n_bins=5, n_jobs=1)
    old = LegacyScanMDLP(max_n_bins=5, n_jobs=1)
    np.testing.assert_array_equal(new._mdlp_split_v3(x, y), old._mdlp_split_v3(x, y))
    candidates = new._find_all_candidates_v3(x, y)
    np.testing.assert_array_equal(new._force_additional_splits_v3(x, y, [], candidates), old._force_additional_splits_v3(x, y, [], candidates))


@pytest.mark.parametrize("target", [np.zeros(30), np.ones(30), np.tile([0.0, 1.0], 15)])
def test_mdlp_pure_and_tied_targets_keep_old_semantics(target):
    x = np.repeat(np.arange(10.0), 3)
    kwargs = dict(max_n_bins=4, min_iv_gain=0.0, n_jobs=1)
    old, new = LegacyScanMDLP(**kwargs), MDLPBinning(**kwargs)
    np.testing.assert_array_equal(old._mdlp_split_v3(x, target), new._mdlp_split_v3(x, target))


def test_mdlp_fitted_tables_and_transform_match_legacy():
    rng = np.random.default_rng(913)
    X = pd.DataFrame({"分数": rng.normal(size=400)})
    y = (rng.random(400) < 0.3).astype(int)
    old = LegacyScanMDLP(max_n_bins=5, n_jobs=1).fit(X, y)
    new = MDLPBinning(max_n_bins=5, n_jobs=1).fit(X, y)
    np.testing.assert_array_equal(old.splits_["分数"], new.splits_["分数"])
    pd.testing.assert_frame_equal(old.bin_tables_["分数"], new.bin_tables_["分数"])
    pd.testing.assert_frame_equal(old.transform(X, metric="woe"), new.transform(X, metric="woe"))
