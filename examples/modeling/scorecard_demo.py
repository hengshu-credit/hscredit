"""
ScoreCard 完整演示样例

展示 ScoreCard 的各种使用方式：
1. 从零开始训练
2. 使用预训练 LR 模型
3. 使用 lr_kwargs 参数
4. 使用已训练的 pipeline
"""

import sys
sys.path.insert(0, '/Users/xiaoxi/CodeBuddy/hscredit/hscredit')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from hscredit.core.models import ScoreCard, LogisticRegression
from hscredit.core.binning import OptimalBinning
from hscredit.utils.datasets import germancredit

print("=" * 80)
print("ScoreCard 完整演示样例")
print("=" * 80)
print()

# 加载数据
print("1. 加载 German Credit 数据集")
df = germancredit()
df['class'] = df['class'].astype(int)

# 分离特征和目标
target = 'class'
feature_cols = [c for c in df.columns if c != target]

print(f"   数据形状: {df.shape}")
print(f"   特征数: {len(feature_cols)}")
print(f"   目标分布:\n{df[target].value_counts()}")
print()

# 划分训练集和测试集
train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])
print(f"   训练集: {train.shape[0]} 样本")
print(f"   测试集: {test.shape[0]} 样本")
print()

# ============================================================================
# 方式1: 从零开始训练（推荐）
# ============================================================================
print("=" * 80)
print("方式1: 从零开始训练 ScoreCard")
print("=" * 80)
print()

# 步骤1: 分箱
print("   步骤1: 最优分箱")
binner = OptimalBinning(max_n_bins=5, method='chi_merge')
binner.fit(train[feature_cols], train[target])

# 步骤2: WOE转换
train_woe = binner.transform(train[feature_cols])
test_woe = binner.transform(test[feature_cols])
print(f"   WOE转换后特征数: {train_woe.shape[1]}")
print()

# 步骤3: 创建并训练 ScoreCard
print("   步骤2: 创建并训练 ScoreCard")
scorecard1 = ScoreCard(
    pdo=50,
    rate=2,
    base_odds=1/19,  # 对应约 5% 的坏样本率
    base_score=750,
    calculate_stats=True
)

scorecard1.fit(train_woe, train[target])
print(f"   ScoreCard 训练完成！")
print(f"   入模特征数: {len(scorecard1.feature_names_)}")
print(f"   模型截距: {scorecard1.intercept_:.4f}")
print()

# 步骤4: 预测评分
print("   步骤3: 预测评分")
train_scores1 = scorecard1.predict(train_woe)
test_scores1 = scorecard1.predict(test_woe)

print(f"   训练集分数范围: [{train_scores1.min():.2f}, {train_scores1.max():.2f}]")
print(f"   训练集分数均值: {train_scores1.mean():.2f}")
print(f"   测试集分数范围: [{test_scores1.min():.2f}, {test_scores1.max():.2f}]")
print(f"   测试集分数均值: {test_scores1.mean():.2f}")
print()

# 步骤5: 查看评分卡规则
print("   步骤4: 查看评分卡规则（前3个特征）")
for i, (feature, rule) in enumerate(list(scorecard1.rules_.items())[:3]):
    print(f"   \n   特征 {i+1}: {feature}")
    print(f"      系数: {rule['coef']:.4f}")
    if rule['bins'] is not None:
        print(f"      分箱数: {len(rule['bins'])}")
print()

# ============================================================================
# 方式2: 使用预训练 LR 模型
# ============================================================================
print("=" * 80)
print("方式2: 使用预训练 LR 模型")
print("=" * 80)
print()

# 步骤1: 训练逻辑回归模型
print("   步骤1: 训练逻辑回归模型")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(train_woe, train[target])
print(f"   逻辑回归模型训练完成！")
print(f"   模型系数: {lr_model.coef_[0][:3]}...")  # 只显示前3个
print(f"   模型截距: {lr_model.intercept_[0]:.4f}")
print()

# 步骤2: 使用预训练模型创建 ScoreCard（不调用 fit）
print("   步骤2: 使用预训练模型创建 ScoreCard")
scorecard2 = ScoreCard(
    binner=binner,
    lr_model=lr_model,
    pdo=50,
    rate=2,
    base_odds=1/19,
    base_score=750
)
print(f"   ScoreCard 创建完成！")
print()

# 步骤3: 直接预测（不调用 fit）
print("   步骤3: 直接预测（不调用 fit）")
train_scores2 = scorecard2.predict(train_woe)
test_scores2 = scorecard2.predict(test_woe)

print(f"   ✅ 预测成功！")
print(f"   训练集分数范围: [{train_scores2.min():.2f}, {train_scores2.max():.2f}]")
print(f"   训练集分数均值: {train_scores2.mean():.2f}")
print(f"   测试集分数范围: [{test_scores2.min():.2f}, {test_scores2.max():.2f}]")
print(f"   测试集分数均值: {test_scores2.mean():.2f}")
print()

# 验证两种方式结果一致
print("   验证: 两种方式预测结果是否一致")
diff = np.abs(train_scores1 - train_scores2).max()
print(f"   最大差异: {diff:.6f}")
if diff < 1e-6:
    print(f"   ✅ 结果完全一致！")
else:
    print(f"   ⚠️ 结果有差异，但可能在可接受范围内")
print()

# ============================================================================
# 方式3: 使用 lr_kwargs 参数
# ============================================================================
print("=" * 80)
print("方式3: 使用 lr_kwargs 参数")
print("=" * 80)
print()

print("   使用 lr_kwargs 传入逻辑回归参数")
scorecard3 = ScoreCard(
    pdo=50,
    rate=2,
    base_odds=1/19,
    base_score=750,
    lr_kwargs={'C': 0.1, 'max_iter': 500, 'random_state': 42}
)

scorecard3.fit(train_woe, train[target])
print(f"   ScoreCard 训练完成！")

train_scores3 = scorecard3.predict(train_woe)
print(f"   训练集分数均值: {train_scores3.mean():.2f}")
print()

# ============================================================================
# 评分卡刻度表
# ============================================================================
print("=" * 80)
print("评分卡刻度表")
print("=" * 80)
print()

scale_df = scorecard1.scorecard_scale()
print("   评分与理论逾期率对应关系（前10行）：")
print(scale_df.head(10).to_string(index=False))
print()

# ============================================================================
# 模型评估
# ============================================================================
print("=" * 80)
print("模型评估")
print("=" * 80)
print()

from hscredit.core.metrics import KS, AUC

# 预测概率
train_proba = scorecard1.predict_proba(train_woe)[:, 1]
test_proba = scorecard1.predict_proba(test_woe)[:, 1]

# 计算 KS 和 AUC
train_ks = KS(train[target].values, train_proba)
test_ks = KS(test[target].values, test_proba)

print(f"   训练集 KS: {train_ks:.4f}")
print(f"   测试集 KS: {test_ks:.4f}")
print()

train_auc = AUC(train[target].values, train_proba)
test_auc = AUC(test[target].values, test_proba)

print(f"   训练集 AUC: {train_auc:.4f}")
print(f"   测试集 AUC: {test_auc:.4f}")
print()

# ============================================================================
# 保存和导出
# ============================================================================
print("=" * 80)
print("保存和导出")
print("=" * 80)
print()

# 查看评分卡规则
print(f"   评分卡规则数: {len(scorecard1.rules_)}")
print(f"   入模特征: {scorecard1.feature_names_[:5]}...")  # 只显示前5个
print()

print("=" * 80)
print("演示完成！")
print("=" * 80)
