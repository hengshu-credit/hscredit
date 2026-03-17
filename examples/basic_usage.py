"""
hscredit 基础使用示例

展示如何使用hscredit进行完整的评分卡建模流程。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 注意: 以下代码在完整实现后可用
# import hscredit as hsc


def load_german_credit():
    """
    加载德国信贷数据集示例
    
    Returns
    -------
    data : DataFrame
        德国信贷数据集
    """
    # 这里使用模拟数据作为示例
    # 实际实现中会加载真实数据集
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(10000, 100000, n_samples),
        'debt_ratio': np.random.uniform(0, 1, n_samples),
        'credit_history': np.random.choice(['good', 'bad', 'unknown'], n_samples),
        'employment_years': np.random.randint(0, 30, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    return data


def example_basic_pipeline():
    """
    基础Pipeline示例
    展示如何使用hscredit构建完整的评分卡建模流程
    """
    print("=" * 60)
    print("hscredit 基础Pipeline示例")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data = load_german_credit()
    print(f"数据集形状: {data.shape}")
    print(f"目标变量分布:\n{data['target'].value_counts()}")
    
    # 2. 数据集划分
    print("\n2. 划分数据集...")
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 3. 构建Pipeline
    print("\n3. 构建Pipeline...")
    # 注意: 以下代码在完整实现后可用
    # pipeline = Pipeline([
    #     ('binning', hsc.OptimalBinning(method='tree', max_n_bins=5)),
    #     ('encoding', hsc.WOEEncoder()),
    #     ('selection', hsc.FeatureSelector(iv_threshold=0.02)),
    #     ('model', hsc.LogisticRegression())
    # ])
    print("Pipeline包含以下步骤:")
    print("  - 分箱: OptimalBinning (决策树分箱)")
    print("  - 编码: WOEEncoder (WOE编码)")
    print("  - 筛选: FeatureSelector (IV筛选)")
    print("  - 建模: LogisticRegression (逻辑回归)")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    # pipeline.fit(X_train, y_train)
    print("模型训练完成")
    
    # 5. 模型评估
    print("\n5. 模型评估...")
    # y_pred = pipeline.predict_proba(X_test)[:, 1]
    # from sklearn.metrics import roc_auc_score, roc_curve
    # auc = roc_auc_score(y_test, y_pred)
    # print(f"测试集AUC: {auc:.4f}")
    
    # 6. 创建评分卡
    print("\n6. 创建评分卡...")
    # scorecard = hsc.ScoreCard(
    #     pdo=60,
    #     base_score=750,
    #     combiner=pipeline.named_steps['binning'],
    #     encoder=pipeline.named_steps['encoding'],
    #     model=pipeline.named_steps['model']
    # )
    # scores = scorecard.transform(X_test)
    print("评分卡创建完成")
    
    # 7. 生成报告
    print("\n7. 生成Excel报告...")
    # report = hsc.ExcelReport('model_report.xlsx')
    # report.add_model_summary(scorecard, X_test, y_test)
    # report.add_ks_plot(y_test, y_pred)
    # report.save()
    print("报告已保存到 model_report.xlsx")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


def example_feature_selection():
    """
    特征筛选示例
    展示如何使用不同的特征筛选方法
    """
    print("\n" + "=" * 60)
    print("特征筛选示例")
    print("=" * 60)
    
    # 加载数据
    data = load_german_credit()
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 1. IV值筛选
    print("\n1. IV值筛选...")
    # selector_iv = hsc.IVSelector(threshold=0.02)
    # selector_iv.fit(X, y)
    # print(f"通过IV筛选的特征: {selector_iv.selected_features_}")
    
    # 2. 相关性筛选
    print("\n2. 相关性筛选...")
    # selector_corr = hsc.CorrelationSelector(threshold=0.7)
    # selector_corr.fit(X, y)
    # print(f"通过相关性筛选的特征: {selector_corr.selected_features_}")
    
    # 3. 综合筛选
    print("\n3. 综合筛选...")
    # selector = hsc.FeatureSelector(
    #     iv_threshold=0.02,
    #     corr_threshold=0.7,
    #     vif_threshold=10
    # )
    # X_selected = selector.fit_transform(X, y)
    # print(f"筛选后特征数量: {X_selected.shape[1]}")
    # print(f"被移除的特征: {selector.removed_features_}")


def example_binning_methods():
    """
    分箱方法示例
    展示不同的分箱算法
    """
    print("\n" + "=" * 60)
    print("分箱方法示例")
    print("=" * 60)
    
    # 加载数据
    data = load_german_credit()
    X = data.drop('target', axis=1)
    y = data['target']
    
    # 1. 决策树分箱
    print("\n1. 决策树分箱...")
    # binner_tree = hsc.TreeBinning(max_depth=3, min_samples_leaf=0.05)
    # binner_tree.fit(X, y)
    # print("决策树分箱完成")
    # print(f"age特征的分箱表:\n{binner_tree.get_bin_table('age')}")
    
    # 2. 卡方分箱
    print("\n2. 卡方分箱...")
    # binner_chi = hsc.ChiMergeBinning(max_n_bins=5)
    # binner_chi.fit(X, y)
    # print("卡方分箱完成")
    
    # 3. 最优分箱
    print("\n3. 最优分箱...")
    # binner_opt = hsc.OptimalBinning(
    #     method='optimal',
    #     monotonic_trend='ascending',
    #     solver='cp'
    # )
    # binner_opt.fit(X, y)
    # print("最优分箱完成")


def example_metrics_calculation():
    """
    指标计算示例
    展示如何计算各种评估指标
    """
    print("\n" + "=" * 60)
    print("指标计算示例")
    print("=" * 60)
    
    # 模拟预测结果
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_score = np.random.uniform(0, 1, 1000)
    
    # 1. KS计算
    print("\n1. KS计算...")
    # ks_value = hsc.KS(y_score, y_true)
    # print(f"KS值: {ks_value:.4f}")
    
    # 2. AUC计算
    print("\n2. AUC计算...")
    # auc_value = hsc.AUC(y_score, y_true)
    # print(f"AUC值: {auc_value:.4f}")
    
    # 3. PSI计算
    print("\n3. PSI计算...")
    # train_score = np.random.uniform(0, 1, 1000)
    # test_score = np.random.uniform(0, 1, 1000)
    # psi_value = hsc.PSI(train_score, test_score)
    # print(f"PSI值: {psi_value:.4f}")
    
    # 4. IV计算
    print("\n4. IV计算...")
    # data = load_german_credit()
    # X = data.drop('target', axis=1)
    # y = data['target']
    # iv_df = hsc.IV(X, y, return_dataframe=True)
    # print(f"各特征IV值:\n{iv_df}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("hscredit 使用示例")
    print("=" * 60)
    
    # 运行各个示例
    example_basic_pipeline()
    example_feature_selection()
    example_binning_methods()
    example_metrics_calculation()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
