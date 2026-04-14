"""WOE 截断功能演示.

演示如何使用 woe_clip 参数限制 WOE 值的范围，防止因分箱中无坏样本
或无好样本导致的极端 WOE 值，避免评分卡分数异常。
"""
import numpy as np
import pandas as pd
from hscredit.core.binning import OptimalBinning
from hscredit.core.encoders import WOEEncoder
from hscredit.core.models import ScoreCard
from hscredit.core.models.classical import LogisticRegression


def main():
    print("=" * 70)
    print("WOE 截断功能演示")
    print("=" * 70)
    print()

    # 创建测试数据：极端情况，一个分箱完全没有坏样本
    np.random.seed(42)
    train = pd.DataFrame({
        'age': list(range(100)) + list(range(100, 200)),
        'target': [0] * 100 + [1] * 100,  # 前100个全为好样本，后100个全为坏样本
    })

    X = train.drop(columns=['target'])
    y = train['target']

    print("数据分布:")
    print(f"  总样本数: {len(train)}")
    print(f"  好样本数: {(y == 0).sum()}")
    print(f"  坏样本数: {(y == 1).sum()}")
    print()

    # 场景1: 不使用 WOE 截断
    print("-" * 70)
    print("场景1: 不使用 WOE 截断 (woe_clip=None)")
    print("-" * 70)

    binner_no_clip = OptimalBinning(
        user_splits={'age': [50]},
        woe_clip=None
    )
    binner_no_clip.fit(X, y)
    bin_table_no_clip = binner_no_clip.get_bin_table('age')
    woe_values_no_clip = bin_table_no_clip['分档WOE值'].tolist()

    print(f"WOE 值: {woe_values_no_clip}")
    print(f"WOE 值范围: [{min(woe_values_no_clip):.2f}, {max(woe_values_no_clip):.2f}]")
    print("注意: WOE 值达到 ±27.6，这会导致评分卡分数异常!")
    print()

    # 场景2: 使用 WOE 截断
    print("-" * 70)
    print("场景2: 使用 WOE 截断 (woe_clip=3.0)")
    print("-" * 70)

    binner_clip = OptimalBinning(
        user_splits={'age': [50]},
        woe_clip=3.0  # 将 WOE 限制在 [-3.0, 3.0] 范围内
    )
    binner_clip.fit(X, y)
    bin_table_clip = binner_clip.get_bin_table('age')
    woe_values_clip = bin_table_clip['分档WOE值'].tolist()

    print(f"WOE 值: {woe_values_clip}")
    print(f"WOE 值范围: [{min(woe_values_clip):.2f}, {max(woe_values_clip):.2f}]")
    print("注意: WOE 值被限制在 [-3.0, 3.0]，评分卡分数更合理")
    print()

    # 场景3: 完整评分卡流程对比
    print("-" * 70)
    print("场景3: 完整评分卡流程对比")
    print("-" * 70)

    for clip_value, label in [(None, "无截断"), (3.0, "截断=3.0")]:
        print(f"\n{label}:")

        # 分箱
        binner = OptimalBinning(user_splits={'age': [50]}, woe_clip=clip_value)
        binner.fit(X, y)

        # WOE 编码
        X_bins = binner.transform(X)
        woe = WOEEncoder(woe_clip=clip_value)
        woe.fit(X_bins, y)
        X_woe = woe.transform(X_bins)

        # 训练 LR
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_woe, y)

        # 创建评分卡
        scorecard = ScoreCard(
            binner=binner,
            encoder=woe,
            lr_model=lr,
            target='target',
            base_score=500,
            base_odds=1.0,
            pdo=50,
        )
        scorecard.fit(X_woe, y)

        # 查看评分卡规则
        rule = scorecard.rules_['age']
        scores = rule['scores']

        print(f"  评分范围: [{scores.min():.2f}, {scores.max():.2f}]")
        print(f"  评分离差: {scores.max() - scores.min():.2f}")

    print()
    print("=" * 70)
    print("结论:")
    print("=" * 70)
    print("使用 woe_clip 参数可以有效限制极端 WOE 值，防止评分卡出现")
    print("异常分数。建议在以下情况使用:")
    print("  - 分箱中存在无坏样本或无好样本的情况")
    print("  - 评分卡分数出现异常大的正值或负值")
    print("  - 需要控制评分卡各分箱分数的合理范围")
    print()
    print("推荐设置: woe_clip=3.0 到 5.0 之间")


if __name__ == '__main__':
    main()
