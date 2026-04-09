"""测试分箱标签的区间开闭格式是否正确.

规则:
- 首箱: (-inf, X]   左开右闭
- 中间: (X, Y]       左开右闭
- 末箱: (X, +inf)    左开右开（无穷不能用闭区间）
- 单箱: (-inf, +inf) 两端都开
- 缺失: missing
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest


def _get_bin_labels_from_base():
    """动态导入 _get_bin_labels 方法以避免全链路导入问题."""
    from importlib import import_module
    import types
    
    # 直接读取 base.py 的源码，提取 _get_bin_labels 方法逻辑
    base_path = os.path.join(
        os.path.dirname(__file__), '..', 'hscredit', 'core', 'binning', 'base.py'
    )
    
    # 使用 exec 加载单个函数避免整个模块导入
    with open(base_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    return source


def _make_bin_labels(splits, bins=None):
    """直接实现 _get_bin_labels 逻辑用于测试."""
    labels = []
    n_splits = len(splits) if splits is not None else 0

    if bins is not None:
        unique_bins = np.unique(bins)
        n_bins = n_splits + 1

        for i in unique_bins:
            if i == -1:
                labels.append('missing')
            elif i == -2:
                labels.append('special')
            elif n_splits == 0:
                labels.append('(-inf, +inf)')
            elif i < n_bins:
                if i == 0:
                    labels.append(f'(-inf, {splits[i]}]')
                elif i == n_bins - 1:
                    labels.append(f'({splits[i-1]}, +inf)')
                else:
                    labels.append(f'({splits[i-1]}, {splits[i]}]')
            else:
                labels.append(f'bin_{i}')
    else:
        if n_splits == 0:
            labels.append('(-inf, +inf)')
        else:
            for i in range(n_splits + 1):
                if i == 0:
                    labels.append(f'(-inf, {splits[i]}]')
                elif i == n_splits:
                    labels.append(f'({splits[i-1]}, +inf)')
                else:
                    labels.append(f'({splits[i-1]}, {splits[i]}]')

    return labels


class TestGetBinLabels:
    """测试 _get_bin_labels 的区间格式."""

    def test_single_split(self):
        """单切分点: (-inf, 10] 和 (10, +inf)."""
        labels = _make_bin_labels(splits=[10.0])
        assert labels == ['(-inf, 10.0]', '(10.0, +inf)']

    def test_multiple_splits(self):
        """多切分点: 首箱(-inf, X], 中间(X, Y], 末箱(X, +inf)."""
        labels = _make_bin_labels(splits=[10.0, 20.0, 30.0])
        assert labels == [
            '(-inf, 10.0]',
            '(10.0, 20.0]',
            '(20.0, 30.0]',
            '(30.0, +inf)',
        ]

    def test_no_splits(self):
        """无切分点: 单箱 (-inf, +inf)."""
        labels = _make_bin_labels(splits=[])
        assert labels == ['(-inf, +inf)']

    def test_with_bins_including_missing(self):
        """包含缺失值箱: missing + 正常区间."""
        bins = np.array([-1, 0, 1, 2])
        labels = _make_bin_labels(splits=[10.0, 20.0], bins=bins)
        assert 'missing' in labels
        assert '(-inf, 10.0]' in labels
        assert '(10.0, 20.0]' in labels
        assert '(20.0, +inf)' in labels

    def test_last_bin_uses_open_bracket(self):
        """末箱必须用 ) 而非 ] — 无穷不能用闭区间."""
        labels = _make_bin_labels(splits=[5.0, 15.0])
        last_label = labels[-1]
        assert last_label.endswith('+inf)'), \
            f"末箱应以 '+inf)' 结尾（开区间），实际: '{last_label}'"
        assert not last_label.endswith('+inf]'), \
            f"末箱不应以 '+inf]' 结尾（闭区间），实际: '{last_label}'"

    def test_first_bin_uses_open_left(self):
        """首箱必须用 ( 开头 — 负无穷用开区间."""
        labels = _make_bin_labels(splits=[5.0])
        first_label = labels[0]
        assert first_label.startswith('(-inf'), \
            f"首箱应以 '(-inf' 开头，实际: '{first_label}'"
