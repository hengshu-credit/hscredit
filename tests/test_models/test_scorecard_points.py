"""ScoreCard / RoundScoreCard 的 scorecard_points 行为测试.

回归覆盖：离线规则加载（无 WOE 值）后，scorecard_points 不应因 zip 截断
而丢失分箱行（历史 bug：加载后仅剩“基础分”一行）。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import ScoreCard, RoundScoreCard
from hscredit.utils.datasets import germancredit


def _train_scorecard(cls=ScoreCard, direction: str = 'descending'):
    df = germancredit().copy()
    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    binner = OptimalBinning(method='target_bad_rate', max_n_bins=5)
    binner.fit(X_train, y_train)

    scorecard = cls(
        pdo=60, rate=2, base_odds=35, base_score=750,
        direction=direction, binner=binner,
    )
    scorecard.fit(binner.transform(X_train, metric='woe'), y_train, input_type='woe')
    return scorecard, binner


def _expected_bin_rows(scorecard) -> int:
    """评分卡规则中所有特征的分箱总数（不含基础分行）."""
    return sum(len(scorecard.rules_[col]['scores']) for col in scorecard.feature_names_)


def test_fitted_scorecard_points_has_base_plus_all_bins():
    scorecard, _ = _train_scorecard()
    points = scorecard.scorecard_points()

    # 第一行为基础分
    assert points.iloc[0]['变量名称'] == '基础分'
    # 行数 = 1（基础分） + 所有分箱数
    assert len(points) == 1 + _expected_bin_rows(scorecard)
    # 拟合得到的评分卡应当带有 WOE 值
    assert points['WOE值'].notna().any()


@pytest.mark.parametrize('cls', [ScoreCard, RoundScoreCard])
def test_scorecard_points_preserved_after_export_load(cls, tmp_path):
    """回归：导出 + 加载后 scorecard_points 不丢失分箱行."""
    scorecard, binner = _train_scorecard(cls=cls)
    fitted_points = scorecard.scorecard_points()
    fitted_rows = len(fitted_points)
    assert fitted_rows > 1  # 至少包含基础分以外的分箱

    json_path = tmp_path / 'rules.json'
    scorecard.export(to_json=str(json_path), include_meta=True)

    loaded = cls(pdo=60, rate=2, base_odds=35, base_score=750)
    loaded.load(str(json_path))
    loaded_points = loaded.scorecard_points()

    # 关键断言：加载后行数与拟合时一致（历史 bug 会退化为仅 1 行）
    assert len(loaded_points) == fitted_rows
    assert loaded_points.iloc[0]['变量名称'] == '基础分'
    # 每个特征都应出现在分箱表中
    for col in scorecard.feature_names_:
        assert (loaded_points['变量名称'] == col).any()


def test_scorecard_points_with_rules_missing_woe_keeps_all_bins():
    """精确回归：rules_ 中无 'woe' 键时仍输出全部分箱行，WOE 值为空."""
    card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
    rules = {
        '__meta__': {
            'intercept_score': 600.0,
            'base_score': 750,
            'direction': 'descending',
            'pdo': 60, 'rate': 2, 'base_odds': 35,
            'A': 442.0, 'B': 86.0,
            'feature_names': ['age', 'amount'],
            'coef': [0.5, 0.3],
        },
        'age': {'[-inf, 25)': 10.0, '[25, 40)': 20.0, '[40, +inf)': 30.0},
        'amount': {'[-inf, 1000)': 5.0, '[1000, +inf)': 15.0},
    }
    card.load(rules)

    # 加载得到的规则确实不含 WOE
    assert 'woe' not in card.rules_['age']

    points = card.scorecard_points()
    # 1 基础分 + 3（age） + 2（amount） = 6 行
    assert len(points) == 6
    assert (points['变量名称'] == 'age').sum() == 3
    assert (points['变量名称'] == 'amount').sum() == 2
    # 无 WOE 时该列应为空
    feature_rows = points[points['变量名称'] != '基础分']
    assert feature_rows['WOE值'].isna().all()
    # 分数应与加载的规则一致
    age_scores = points[points['变量名称'] == 'age']['对应分数'].tolist()
    assert age_scores == [10.0, 20.0, 30.0]
