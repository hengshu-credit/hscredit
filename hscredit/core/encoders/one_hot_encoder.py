"""One-Hot Encoder (独热编码器).

将类别特征转换为独热编码形式，支持数值型和类别型数据。
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd

from .base import BaseEncoder
from ._category_protocol import CategoryToken, MISSING, INFREQUENT
from ...exceptions import NotFittedError


# 保留旧类路径别名，使此前完整对象制品可以加载。
_OneHotBucket = CategoryToken
_MISSING = MISSING
_INFREQUENT = INFREQUENT


class OneHotEncoder(BaseEncoder):
    """独热编码器.

    将每个类别转换为一个二进制列，适用于类别数量不多的特征。
    支持数值型和类别型数据。

    **参数**

    :param cols: 需要编码的列名列表。如果为None，则编码所有列
    :param drop: 是否删除某一列以避免多重共线性，默认为None
        - None: 保留所有列
        - 'first': 删除第一列
        - 'if_binary': 二值特征时删除一列
    :param handle_unknown: 处理未知类别的方式，默认为'ignore'
        - 'error': 抛出错误
        - 'ignore': 忽略（所有编码列为0）
    :param handle_missing: 处理缺失值的方式，默认为'value'
        - 'value': 单独编码为'missing'列
        - 'error': 抛出错误
    :param use_cat_names: 是否使用类别值作为列名后缀，默认为True
    :param return_df: 是否返回DataFrame，默认为True
    :param sparse_output: 是否原生构造稀疏输出；return_df=True 返回稀疏DataFrame，
        False 返回CSR（此时未编码透传列必须为数值），默认False
    :param min_frequency: 低频类别合并门槛：正整数人数或 (0,1) 内比例
    :param max_categories: 每列最多保留的非缺失输出类别数，包含稀有类别桶；默认不限制
    :param max_output_bytes: 编码输出预计字节预算，超限在构造输出前报错；
        不包含上游输入、进程或全部运行时内存，默认不限制

    **属性**

    - categories_: 各列的类别列表，格式为 {col: [category1, category2, ...]}
    - feature_names_: 编码后的特征名列表

    **参考样例**

    >>> from hscredit.core.encoders import OneHotEncoder
    >>> encoder = OneHotEncoder(cols=['color'])
    >>> X_encoded = encoder.fit_transform(X)
    >>>
    >>> # 删除第一列避免多重共线性
    >>> encoder = OneHotEncoder(cols=['color'], drop='first')
    >>> X_encoded = encoder.fit_transform(X)

    **注意**

    独热编码为无监督方法，列数随类别基数线性增长，仅适合低基数特征；用于线性/逻辑回归时
    建议 ``drop='first'`` 以消除虚拟变量陷阱（多重共线性），用于树模型可保留全部列。

    **引用**

    虚拟变量（dummy variables）/ one-hot 编码是统计建模标准做法，参见 sklearn
    ``OneHotEncoder``：
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    """

    # categories_ 是 transform 生成独热列所必需的状态（_transform 依赖它而非 mapping_）；
    # feature_names_ / _other_cols_ 供 get_feature_names(_out) 使用，三者须一并序列化
    _EXTRA_STATE_ATTRS = ["categories_", "feature_names_", "_other_cols_", "known_categories_", "category_groups_"]

    def __init__(
        self,
        cols: Optional[List[str]] = None,
        drop: Optional[str] = None,
        handle_unknown: str = "ignore",
        handle_missing: str = "value",
        use_cat_names: bool = True,
        return_df: bool = True,
        target: Optional[str] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        sparse_output: bool = False,
        min_frequency: Optional[Union[int, float]] = None,
        max_categories: Optional[int] = None,
        max_output_bytes: Optional[int] = None,
        passthrough_target: bool = False,
    ):
        """初始化独热编码器。

        :param cols: 需要编码的列名列表
        :param drop: 是否删除某一列以避免多重共线性
        :param handle_unknown: 处理未知类别的方式
        :param handle_missing: 处理缺失值的方式
        :param use_cat_names: 是否使用类别值作为列名后缀
        :param return_df: 是否返回DataFrame
        :param target: scorecardpipeline风格的目标列名
        """
        super().__init__(
            cols=cols,
            drop_invariant=False,
            return_df=return_df,
            handle_unknown=handle_unknown,
            handle_missing=handle_missing,
            target=target,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            passthrough_target=passthrough_target,
        )
        self.drop = drop
        self.use_cat_names = use_cat_names
        self.sparse_output = sparse_output
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.max_output_bytes = max_output_bytes

        self.categories_: Dict[str, List] = {}
        self.feature_names_: List[str] = []
        self._other_cols_: List[str] = []
        self.known_categories_: Dict[str, List] = {}
        self.category_groups_: Dict[str, Dict] = {}

    def _get_category_cols(self, X: pd.DataFrame) -> List[str]:
        """获取需要编码的列。

        OneHotEncoder支持数值型和类别型列。

        :param X: 输入数据
        :return: 列名列表
        """
        if self.cols is not None:
            return [c for c in self.cols if c in X.columns]
        return X.columns.tolist()

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """拟合独热编码器。

        :param X: 输入数据
        :param y: 目标变量（可选）
        """
        # 保留未编码列，与 _transform 输出顺序保持一致（未编码列在前）
        if self.drop not in (None, "first", "if_binary"):
            raise ValueError("drop 必须为 None、first 或 if_binary")
        if self.handle_unknown not in ("ignore", "error"):
            raise ValueError("OneHot 的 handle_unknown 必须为 ignore 或 error")
        if self.handle_missing not in ("value", "error"):
            raise ValueError("OneHot 的 handle_missing 必须为 value 或 error")
        if self.min_frequency is not None:
            value = self.min_frequency
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or value <= 0:
                raise ValueError("min_frequency 必须为正整数或 (0, 1) 内的比例")
            if isinstance(value, (float, np.floating)) and value >= 1:
                raise ValueError("浮点 min_frequency 必须在 (0, 1) 内")
        if self.max_categories is not None and (isinstance(self.max_categories, (bool, np.bool_)) or not isinstance(self.max_categories, (int, np.integer)) or self.max_categories < 1):
            raise ValueError("max_categories 必须为正整数")
        if self.max_output_bytes is not None and (isinstance(self.max_output_bytes, (bool, np.bool_)) or not isinstance(self.max_output_bytes, (int, np.integer)) or self.max_output_bytes < 1):
            raise ValueError("max_output_bytes 必须为正整数字节数")
        self._other_cols_ = [c for c in X.columns if c not in self.cols_]
        self._fit_columns(X, y, state_attrs=("mapping_", "categories_", "known_categories_", "category_groups_"))
        # 跨原字段及透传字段也可能发生同名；保持原有无冲突列名，冲突时加稳定后缀。
        used = set(self._other_cols_)
        for column in self.cols_:
            for category in self.categories_[column]:
                preferred = self.mapping_[column][category]
                name, suffix = preferred, 2
                while name in used:
                    name = f"{preferred}__{suffix}"
                    suffix += 1
                self.mapping_[column][category] = name
                used.add(name)
        self.feature_names_ = [
            self.mapping_[column][category]
            for column in self.cols_
            for category in self.categories_[column]
        ]

    def _fit_column(self, column, values, y=None):
        # 获取唯一值（包括缺失值）
        categories = values.unique()

        # 分离缺失值和正常值
        has_missing = any(pd.isna(c) for c in categories)
        normal_categories = self._sort_categories([c for c in categories if not pd.isna(c)])

        counts = values.value_counts(dropna=True)
        minimum = self.min_frequency or 0
        if isinstance(minimum, (float, np.floating)):
            minimum *= len(values)
        frequent = [category for category in normal_categories if counts.get(category, 0) >= minimum]
        if self.max_categories is not None and (len(frequent) > self.max_categories or len(frequent) < len(normal_categories)):
            ranked = sorted(frequent, key=lambda category: -counts.get(category, 0))
            keep = set(ranked[: max(0, self.max_categories - 1)])
            frequent = [category for category in frequent if category in keep]
        frequent_set = set(frequent)
        groups = {category: category if category in frequent_set else _INFREQUENT for category in normal_categories}
        output_categories = frequent + ([_INFREQUENT] if len(frequent) < len(normal_categories) else [])

        # 处理drop参数
        if self.drop == "first" and len(output_categories) > 0:
            categories_to_use = output_categories[1:]
        elif self.drop == "if_binary" and len(output_categories) == 2:
            categories_to_use = output_categories[:1]
        else:
            categories_to_use = output_categories[:]

        # 如果有缺失值且handle_missing='value'，添加missing
        if has_missing and self.handle_missing == "value":
            categories_to_use = categories_to_use + [_MISSING]

        # 构建mapping_（与其他编码器保持一致）
        col_mapping = {}
        for cat in categories_to_use:
            if cat == _MISSING:
                col_name = f"{column}_missing"
            elif cat == _INFREQUENT:
                col_name = f"{column}_infrequent"
            elif self.use_cat_names:
                safe_cat = str(cat).replace(" ", "_").replace("-", "_")
                col_name = f"{column}_{safe_cat}"
            else:
                col_name = f"{column}_{cat}"
            col_mapping[cat] = col_name
        return {"mapping_": col_mapping, "categories_": categories_to_use, "known_categories_": normal_categories, "category_groups_": groups}

    def _transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """转换数据。

        :param X: 输入数据
        :param y: 目标变量（可选）
        :return: 编码后的数据
        """
        from scipy import sparse

        other_cols = [column for column in X.columns if column not in self.cols_]
        columns = len(self.feature_names_) + len(other_cols)
        # 编码输出的保守预算；不声称覆盖上游 DataFrame 或进程的全部内存。
        estimate = (len(X) * (len(self.cols_) + len(other_cols)) * 16 + (len(X) + 1) * 8) if self.sparse_output else len(X) * columns * np.dtype(int).itemsize
        self.estimated_output_bytes_ = estimate
        if self.max_output_bytes is not None and estimate > self.max_output_bytes:
            raise ValueError(f"独热编码预计输出 {estimate} 字节，超过 max_output_bytes={self.max_output_bytes}")
        output = self._transform_columns(X, y, passthrough=True)
        if self.sparse_output and not self.return_df:
            blocks = []
            if other_cols:
                try:
                    blocks.append(sparse.csr_matrix(output[other_cols].to_numpy(dtype=float)))
                except (ValueError, TypeError) as exc:
                    raise ValueError("返回 CSR 时未编码透传列必须为数值") from exc
            if self.feature_names_:
                blocks.append(output[self.feature_names_].sparse.to_coo().tocsr())
            return sparse.hstack(blocks, format="csr") if blocks else sparse.csr_matrix((len(X), 0))
        return output

    def _transform_column(self, column, values, y=None, context=None):
        categories = self.categories_[column]

        # 检查未知类别
        if self.handle_unknown == "error":
            unique_vals = set(values.dropna().unique())
            known_vals = set(self.known_categories_.get(column, categories))
            unknown = unique_vals - known_vals
            if unknown:
                raise ValueError(f"列'{column}'包含未知类别: {unknown}")

        from scipy import sparse

        positions = {category: position for position, category in enumerate(categories)}
        groups = self.category_groups_.get(column, {})
        category_positions = {category: positions.get(group, -1) for category, group in groups.items()}
        if not groups:  # 兼容旧映射制品
            category_positions = positions
        indices = values.map(category_positions).fillna(-1).to_numpy(dtype=int)
        indices[values.isna().to_numpy()] = positions.get(_MISSING, positions.get("missing", -1) if not groups else -1)
        valid = indices >= 0
        matrix = sparse.csr_matrix((np.ones(valid.sum(), dtype=int), (np.flatnonzero(valid), indices[valid])), shape=(len(values), len(categories)))
        names = [self.mapping_[column][category] for category in categories]
        if self.sparse_output:
            return pd.DataFrame.sparse.from_spmatrix(matrix, index=values.index, columns=names)
        return pd.DataFrame(matrix.toarray(), index=values.index, columns=names)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆编码，将独热编码列还原为原始类别列。

        对每个原始列，取值为 1 的独热列对应类别即为原始类别；
        若所有独热列均为 0（如 drop 删除的参考类别或未知类别），则还原为 NaN。
        缺失列（``{col}_missing``）激活时还原为 NaN。

        :param X: 编码后的数据
        :return: 逆编码后的数据
        :raises NotFittedError: 当编码器尚未拟合时抛出
        """
        if not hasattr(self, "mapping_") or self.mapping_ is None or len(self.mapping_) == 0:
            raise NotFittedError("OneHotEncoder 尚未拟合，请先调用 fit 方法")

        X = self._check_input(X).copy()

        reconstructed = {}
        consumed = set()
        for col in self.cols_ or []:
            col_map = self.mapping_.get(col, {})  # {category: col_name}
            name_to_cat = {name: cat for cat, name in col_map.items()}
            present = [n for n in col_map.values() if n in X.columns]
            if not present:
                continue
            consumed.update(present)

            sub = X[present]

            def _pick(row):
                for name in present:
                    if row[name] == 1:
                        cat = name_to_cat[name]
                        return np.nan if cat == _MISSING else ("稀有类别" if cat == _INFREQUENT else cat)
                return np.nan

            reconstructed[col] = sub.apply(_pick, axis=1)

        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            if c not in consumed:
                out[c] = X[c]
        for col, series in reconstructed.items():
            out[col] = series

        return out

    def get_feature_names(self) -> List[str]:
        """获取独热编码生成的特征名（不含未编码的透传列）。

        :return: 独热编码后的特征名列表
        """
        return self.feature_names_

    @staticmethod
    def _pack_category(value):
        """JSON 类别值；桶与业务字符串使用不同结构，不依赖保留字符串。"""
        if isinstance(value, _OneHotBucket):
            return {"bucket": value.kind}
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError("该类别类型不支持映射 JSON，请使用 save_artifact 保存完整编码器")
        return {"value": value}

    @staticmethod
    def _unpack_category(record):
        if "bucket" in record:
            if record["bucket"] not in ("missing", "infrequent"):
                raise ValueError("独热映射包含未知类别桶")
            return _OneHotBucket(record["bucket"])
        return record["value"]

    def export_mapping(self, *, legacy=False):
        """导出带类型记录的 JSON 映射，避免 JSON 字典键丢失类别类型。"""
        if not legacy:
            return super().export_mapping()
        payload = super().export_mapping(legacy=True)
        payload["onehot_version"] = 2
        payload["mapping_"] = {
            column: {str(i): self.mapping_[column][category] for i, category in enumerate(self.categories_[column])}
            for column in self.cols_ or []
        }
        payload["extra_state"] = {"feature_names_": self.feature_names_, "_other_cols_": self._other_cols_}
        payload["onehot_columns"] = {
            column: {
                "output": [[self._pack_category(category), self.mapping_[column][category]] for category in self.categories_[column]],
                "known": [self._pack_category(category) for category in self.known_categories_.get(column, [])],
                "groups": [[self._pack_category(category), self._pack_category(group)] for category, group in self.category_groups_.get(column, {}).items()],
            } for column in self.cols_ or []
        }
        return payload

    def import_mapping(self, mapping):
        """加载当前带类型记录映射或旧版映射。"""
        if mapping.get("format") == "hscredit-encoder-mapping":
            return super().import_mapping(mapping)
        version = mapping.get("onehot_version")
        if version is not None and version != 2:
            raise ValueError("不支持的独热映射协议版本")
        super().import_mapping(mapping)
        if version == 2:
            self.mapping_, self.categories_, self.known_categories_, self.category_groups_ = {}, {}, {}, {}
            for column, state in mapping["onehot_columns"].items():
                self.mapping_[column] = {self._unpack_category(category): name for category, name in state["output"]}
                self.categories_[column] = list(self.mapping_[column])
                self.known_categories_[column] = [self._unpack_category(category) for category in state["known"]]
                self.category_groups_[column] = {self._unpack_category(category): self._unpack_category(group) for category, group in state["groups"]}
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """获取转换后的全部输出列名（sklearn 兼容接口）。

        输出顺序与 transform 一致：未编码透传列在前，独热编码列在后。

        :param input_features: 兼容 sklearn 接口的占位参数，未使用
        :return: 输出列名数组
        """
        other_cols = getattr(self, "_other_cols_", [])
        columns = list(other_cols) + list(self.feature_names_)
        if getattr(self, "passthrough_target", False) and getattr(self, "_target_in_fit_", False):
            columns.append(self.target)
        return np.asarray(columns, dtype=object)
