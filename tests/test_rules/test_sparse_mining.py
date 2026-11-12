"""高缺失字段不应因为非缺失子集只有一类标签而中断批量挖掘。"""

import numpy as np
import pandas as pd

from hscredit.report.mining import SingleFeatureRuleMiner


def test_sparse_single_class_keeps_missingness_rules():
    frame = pd.DataFrame({"稀疏变量": [np.nan, np.nan, np.nan, 10, 20], "标签": [0, 0, 1, 1, 1]})
    miner = SingleFeatureRuleMiner(target="标签", features=["稀疏变量"], method="quantile", min_lift=0, min_samples=1, n_jobs=1).fit(frame)
    rules = miner.get_top_rules(top_n=10)
    present = rules.loc[rules["规则表达式"].str.contains("notna")].iloc[0]
    missing = rules.loc[~rules["规则表达式"].str.contains("notna")].iloc[0]
    assert present["命中样本数"] == 2
    assert present["命中坏样本数"] == 2
    assert missing["命中样本数"] == 3
    assert missing["命中坏样本数"] == 1


def test_boolean_feature_is_mined_as_category():
    frame = pd.DataFrame({"布尔变量": [True, True, False, False], "常量布尔": [False] * 4, "标签": [1, 1, 0, 1]})
    miner = SingleFeatureRuleMiner(target="标签", method="quantile", min_lift=0, min_samples=1, n_jobs=1).fit(frame)
    rules = miner.get_top_rules(top_n=10)
    hit = rules.loc[rules["规则表达式"].str.contains("布尔变量.*True")].iloc[0]
    assert hit["命中样本数"] == 2
    assert hit["命中坏样本数"] == 2
    assert any("常量布尔" in expr for expr in rules["规则表达式"])
