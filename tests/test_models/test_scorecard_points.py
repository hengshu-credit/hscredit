"""ScoreCard / RoundScoreCard 的 scorecard_points 行为测试.

回归覆盖：离线规则加载（无 WOE 值）后，scorecard_points 不应因 zip 截断
而丢失分箱行（历史 bug：加载后仅剩“基础分”一行）。
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from hscredit.core.binning import OptimalBinning
from hscredit.core.models import ScoreCard, RoundScoreCard, LogisticRegression
from hscredit.exceptions import NotFittedError
from hscredit.utils.datasets import germancredit


def _train_lr_on_woe():
    """获取 WOE 特征上训练好的逻辑回归 + 对应 binner（模拟 04_models.ipynb 流程）.

    复用 _train_scorecard（method='target_bad_rate'，可稳定拟合）得到的
    底层 LR 与 binner，避免直接对含缺失 WOE 的数据训练 sklearn LR 失败。
    """
    scorecard, binner = _train_scorecard()
    return scorecard.lr_model_, binner


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
@pytest.mark.parametrize(
    'method_call',
    [
        lambda card: card.predict(pd.DataFrame({'x': [1.0]})),
        lambda card: card.predict_score(proba=[0.1, 0.2]),
        lambda card: card.transform([0.1, 0.2]),
        lambda card: card.scorecard_points(),
        lambda card: card.get_feature_importances(),
    ],
)
def test_unfitted_scorecard_methods_raise_not_fitted(cls, method_call):
    """ScoreCard 构造期有 A_/B_，不能让 sklearn 默认检查误判为已拟合."""
    card = cls()

    with pytest.raises(NotFittedError, match='尚未拟合'):
        method_call(card)


def test_round_scorecard_rejects_unfitted_source_scorecard():
    with pytest.raises(NotFittedError, match='尚未拟合'):
        RoundScoreCard(scorecard=ScoreCard())


def test_scorecard_reports_keep_single_bin_for_constant_scores():
    card = ScoreCard(pdo=60, rate=2, base_odds=35, base_score=750)
    card.load({
        '__meta__': {
            'intercept_score': 600.0,
            'base_score': 750,
            'direction': 'descending',
            'pdo': 60,
            'rate': 2,
            'base_odds': 35,
            'feature_names': ['x'],
            'coef': [1.0],
        },
        'x': {'[-inf, +inf)': 0.0},
    })

    scores = np.array([600.0, 600.0, 600.0])
    y = np.array([0, 1, 0])

    bad_rate = card.score_to_bad_rate_table(scores, y, n_bins=5)
    assert len(bad_rate) == 1
    assert bad_rate.loc[0, '评分区间'] == '[600.0000, 600.0000]'
    assert bad_rate.loc[0, '样本数'] == 3
    assert bad_rate.loc[0, '坏样本率'] == '33.33%'

    probability = card.score_to_probability_table(scores=scores, y=y, n_bins=5)
    assert len(probability) == 1
    assert probability.loc[0, '评分区间'] == '[600.00, 600.00]'
    assert probability.loc[0, '样本数'] == 3
    assert probability.loc[0, '实际逾期率'] == '33.33%'


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


def test_scorecard_points_lr_model_only_shows_all_features():
    """回归（04_models.ipynb scorecard_tuned）：仅传 lr_model（无 binner、未 fit）时，
    scorecard_points 必须展示每个变量，而非仅基础分."""
    lr, _ = _train_lr_on_woe()
    n_features = len(lr.coef_[0])

    card = ScoreCard(lr_model=lr, base_score=600, pdo=20)
    points = card.scorecard_points()

    # 历史 bug：只返回 1 行（基础分）。修复后应为 基础分 + 每个特征一行回退规则
    assert len(points) == 1 + n_features
    assert points.iloc[0]['变量名称'] == '基础分'
    feature_rows = points[points['变量名称'] != '基础分']
    assert len(feature_rows) == n_features
    # 无分箱信息时，分箱标签标注为「每单位WOE(系数=...)」
    assert feature_rows['变量分箱'].str.contains('每单位WOE').all()


def test_scorecard_points_lr_model_with_binner_recovers_real_bins():
    """传入 lr_model + binner 时，scorecard_points 还原真实分箱区间（推荐用法）."""
    lr, binner = _train_lr_on_woe()
    n_features = len(lr.coef_[0])

    card = ScoreCard(lr_model=lr, binner=binner, base_score=600, pdo=20)
    points = card.scorecard_points()

    # 每个特征都应有多个分箱（远多于回退的 1 行/特征）
    assert len(points) > 1 + n_features
    feature_rows = points[points['变量名称'] != '基础分']
    # 出现真实区间标签（含区间括号），而非回退标签
    assert feature_rows['变量分箱'].str.contains(r'[\[(]').any()
    assert not feature_rows['变量分箱'].str.contains('每单位WOE').any()


def test_scorecard_scale_includes_formula_matching_score_formula():
    """scorecard_scale 应包含 formula 行，且与 score_formula() 的公式一致."""
    lr, binner = _train_lr_on_woe()
    card = ScoreCard(lr_model=lr, binner=binner, base_score=600, pdo=20)

    scale = card.scorecard_scale()
    assert 'formula' in scale['刻度项'].values

    formula_value = scale.loc[scale['刻度项'] == 'formula', '刻度值'].iloc[0]
    # 与 score_formula 的 A、B 一致
    info = card.score_formula()
    assert str(round(info['A'], 4)) in formula_value
    assert str(round(info['B'], 4)) in formula_value
    assert 'ln(odds)' in formula_value
