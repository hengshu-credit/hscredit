"""数据集工具.

提供示例数据集的加载功能。
"""

import os
import pandas as pd
from pandas.api.types import CategoricalDtype


def germancredit():
    """加载德国信贷数据集 German Credit Data。

    数据来源：https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

    :return: pd.DataFrame

    示例:
        >>> df = germancredit()
        >>> print(df.shape)
        (1000, 21)
    """
    # 方案1：优先读取仓库内置数据（离线可用）
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'resources',
        'datasets',
        'germancredit.csv',
    )
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # 本地文件目标列为 creditability：good/bad -> 0/1
        if 'creditability' in df.columns:
            df['class'] = df['creditability'].map({'good': 0, 'bad': 1})
            df = df.drop(columns=['creditability'])
        elif 'class' in df.columns:
            # 兼容 class 列为字符串或数值的情况
            if df['class'].dtype == 'O':
                df['class'] = df['class'].map({'good': 0, 'bad': 1})
            else:
                df['class'] = df['class'].map({1: 0, 2: 1}).fillna(df['class'])
        return df

    # 方案2：使用 sklearn OpenML（在线）
    try:
        from sklearn.datasets import fetch_openml
        try:
            data = fetch_openml(name='credit-g', version=1, as_frame=True, parser='auto')
        except TypeError:
            # 兼容旧版本 sklearn（无 parser 参数）
            data = fetch_openml(name='credit-g', version=1, as_frame=True)

        df = data.frame
        df['class'] = df['class'].map({'good': 0, 'bad': 1})
        return df
    except Exception:
        # 方案3：从 UCI 加载
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

        column_names = [
            "status_of_existing_checking_account", "duration_in_month", "credit_history",
            "purpose", "credit_amount", "savings_account_and_bonds", "present_employment_since",
            "installment_rate_in_percentage_of_disposable_income", "personal_status_and_sex",
            "other_debtors_or_guarantors", "present_residence_since", "property", "age_in_years",
            "other_installment_plans", "housing", "number_of_existing_credits_at_this_bank",
            "job", "number_of_people_being_liable_to_provide_maintenance_for", "telephone",
            "foreign_worker", "class"
        ]

        df = pd.read_csv(url, sep=' ', names=column_names, header=None)

        # 转换目标变量 (1=good, 2=bad -> 0=good, 1=bad)
        df['class'] = df['class'].map({1: 0, 2: 1})

        return df


def load_titanic():
    """加载泰坦尼克号数据集。

    :return: pd.DataFrame

    示例:
        >>> df = load_titanic()
    """
    try:
        import seaborn as sns
        return sns.load_dataset('titanic')
    except Exception:
        url = "https://raw.githubusercontent.com/datasciencedoc/data/master/titanic.csv"
        return pd.read_csv(url)


def load_iris():
    """加载鸢尾花数据集。

    :return: pd.DataFrame

    示例:
        >>> df = load_iris()
    """
    from sklearn.datasets import load_iris as sklearn_load_iris
    iris = sklearn_load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df
