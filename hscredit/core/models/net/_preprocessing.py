"""仅在训练子集拟合的数值填补、标准化与类别编码。"""

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ....exceptions import ValidationError


class TabularPreprocessor:
    """数值列保留全空列；类别编码 0 专供缺失及未知类别。"""

    @staticmethod
    def _frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def fit(self, X, categorical_features=None):
        """记录训练数据中的填充值、尺度与类别字典。

        :param X: 非空二维训练特征，可包含数值、类别及缺失值。
        :param categorical_features: 字段名或列位置列表；None 自动识别。
        :return: 当前预处理器。

        >>> preprocessor = TabularPreprocessor().fit(X_train)
        """
        frame = self._frame(X)
        self.n_features_ = frame.shape[1]
        if categorical_features is None:
            self.cat_indices_ = [
                i for i, dtype in enumerate(frame.dtypes) if not is_numeric_dtype(dtype) or is_bool_dtype(dtype)
            ]
        else:
            if isinstance(categorical_features, (str, bytes)):
                raise ValidationError("categorical_features 必须是字段名或列位置的列表")
            self.cat_indices_ = []
            for value in categorical_features:
                if isinstance(value, str) and value in frame.columns:
                    index = frame.columns.get_loc(value)
                elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
                    index = int(value)
                else:
                    raise ValidationError(f"未知类别字段: {value!r}")
                if not 0 <= index < self.n_features_ or index in self.cat_indices_:
                    raise ValidationError("类别字段位置越界或重复")
                self.cat_indices_.append(index)
        self.num_indices_ = [i for i in range(self.n_features_) if i not in self.cat_indices_]
        self.categories_ = [pd.Index(frame.iloc[:, i].dropna().unique()) for i in self.cat_indices_]
        self.cardinalities_ = [len(values) + 1 for values in self.categories_]
        values = self._numeric(frame)
        self.medians_ = np.array(
            [np.median(col[~np.isnan(col)]) if np.any(~np.isnan(col)) else 0.0 for col in values.T]
        )
        filled = np.where(np.isnan(values), self.medians_, values)
        self.means_ = filled.mean(axis=0)
        self.scales_ = filled.std(axis=0)
        self.scales_[self.scales_ == 0] = 1.0
        return self

    def _numeric(self, frame):
        try:
            values = frame.iloc[:, self.num_indices_].to_numpy(dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValidationError("数值特征包含非数值，请通过 categorical_features 指定类别列") from exc
        if np.isinf(values).any():
            raise ValidationError("数值特征不能包含正负无穷；缺失值请使用 NaN")
        return values

    def transform(self, X):
        frame = self._frame(X)
        values = self._numeric(frame)
        values = (np.where(np.isnan(values), self.medians_, values) - self.means_) / self.scales_
        with np.errstate(over="ignore"):
            numeric = values.astype(np.float32)
        if not np.isfinite(numeric).all():
            raise ValidationError("标准化后的数值超出 float32 范围，请先缩放异常值")
        categorical = np.empty((len(frame), len(self.cat_indices_)), dtype=np.int64)
        for j, (i, categories) in enumerate(zip(self.cat_indices_, self.categories_)):
            categorical[:, j] = pd.Categorical(frame.iloc[:, i], categories=categories).codes.astype(np.int64) + 1
        return numeric, categorical
