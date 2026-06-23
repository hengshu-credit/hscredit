"""生产规则流转一致性校验."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame

from ...exceptions import FeatureNotFoundError, InputTypeError
from .rule import Rule


RuleInput = Union[Rule, Sequence[Rule]]


class RuleFlow:
    """生产规则流转校验器。

    按给定规则顺序计算每笔样本的规则命中结果，支持串行和并行两种执行模式。

    **参数**

    :param rules: 单条 :class:`Rule` 或规则列表，规则顺序即生产执行顺序
    :param mode: 执行模式，``"serial"``/``"串行"`` 表示命中第一条规则后停止，
        ``"parallel"``/``"并行"`` 表示所有规则都参与判断
    :param name: 规则流名称，默认 ``"RuleFlow"``

    **属性**

    :ivar rules: 规则列表
    :ivar mode: 标准化后的执行模式，取值为 ``"serial"`` 或 ``"parallel"``
    :ivar feature_names_in_: 全部规则表达式引用的字段列表

    **参考样例**

    >>> from hscredit.core.rules import Rule, RuleFlow
    >>> flow = RuleFlow([Rule("score < 500", name="低分"), Rule("cnt > 5", name="多头")])
    >>> result = flow.predict(data)
    >>> report = flow.report(data, date_col="放款时间", freq="M", group_cols="商品类别")
    >>> summary = flow.summary(data, group_cols="商品类别")
    """

    _SERIAL_MODES = {"serial", "sequential", "串行"}
    _PARALLEL_MODES = {"parallel", "并行"}

    def __init__(self, rules: RuleInput, mode: str = "serial", name: Optional[str] = None):
        self.rules = self._normalize_rules(rules)
        self.mode = self._normalize_mode(mode)
        self.name = name or "RuleFlow"
        self.feature_names_in_ = sorted({feature for rule in self.rules for feature in rule.feature_names_in_})
        self.rule_labels_ = self._build_rule_labels()

    @staticmethod
    def _normalize_rules(rules: RuleInput) -> List[Rule]:
        if isinstance(rules, Rule):
            return [rules]
        if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
            raise InputTypeError("rules 必须是 Rule 或 Rule 列表")
        rules = list(rules)
        if not rules:
            raise ValueError("rules 不能为空")
        invalid = [type(rule).__name__ for rule in rules if not isinstance(rule, Rule)]
        if invalid:
            raise InputTypeError(f"rules 中只能包含 Rule 对象，发现: {invalid}")
        return rules

    @classmethod
    def _normalize_mode(cls, mode: str) -> str:
        if not isinstance(mode, str):
            raise InputTypeError("mode 必须是字符串")
        normalized = mode.lower()
        if normalized in cls._SERIAL_MODES:
            return "serial"
        if normalized in cls._PARALLEL_MODES:
            return "parallel"
        raise ValueError("mode 仅支持 'serial'/'parallel' 或 '串行'/'并行'")

    def _build_rule_labels(self) -> List[str]:
        counts: Dict[str, int] = {}
        labels = []
        for rule in self.rules:
            base = rule.name or rule.expr
            counts[base] = counts.get(base, 0) + 1
            labels.append(base if counts[base] == 1 else f"{base}#{counts[base]}")
        return labels

    def _validate_data(self, data: DataFrame) -> None:
        if not isinstance(data, DataFrame):
            raise InputTypeError("RuleFlow 只能处理 DataFrame")
        missing_cols = set(self.feature_names_in_) - set(data.columns)
        if missing_cols:
            raise FeatureNotFoundError(f"数据集缺少规则字段: {missing_cols}")

    @staticmethod
    def _predict_rule(rule: Rule, data: DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(False, index=data.index, dtype=bool)
        result = rule.predict(data)
        return pd.Series(result, index=data.index).fillna(False).astype(bool)

    @staticmethod
    def _join_rule_names(row: pd.Series, labels: Sequence[str]) -> str:
        matched = [label for label in labels if bool(row.get(label, False))]
        return "|".join(matched)

    @staticmethod
    def _join_rule_indices(row: pd.Series, labels: Sequence[str]) -> str:
        matched = [str(i + 1) for i, label in enumerate(labels) if bool(row.get(label, False))]
        return "|".join(matched)

    @staticmethod
    def _is_missing(value) -> bool:
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            return False
        return pd.isna(value)

    def _rule_alias_map(self) -> Dict[str, str]:
        aliases = {}
        for rule, label in zip(self.rules, self.rule_labels_):
            aliases[label] = label
            aliases[rule.name] = label
            aliases[rule.expr] = label
        return aliases

    def _normalize_hit_value(self, value, aliases: Dict[str, str]) -> List[str]:
        if self._is_missing(value):
            return []
        if isinstance(value, str):
            values = [item.strip() for item in value.replace(",", "|").split("|") if item.strip()]
        elif isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            values = [item for item in value if not self._is_missing(item)]
        else:
            values = [value]

        labels = []
        for item in values:
            key = str(item).strip()
            if key in aliases:
                labels.append(aliases[key])
        return list(dict.fromkeys(labels))

    def _align_production_hits(
        self,
        data: DataFrame,
        production_hits: Union[pd.DataFrame, pd.Series],
        order_id_col: Optional[str],
    ) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(production_hits, (pd.DataFrame, pd.Series)):
            raise InputTypeError("production_hits 必须是 DataFrame 或 Series")

        if order_id_col is None:
            if len(production_hits) != len(data):
                raise ValueError("未指定 order_id_col 时，production_hits 行数必须与 data 一致")
            if production_hits.index.equals(data.index):
                return production_hits
            aligned = production_hits.copy()
            aligned.index = data.index
            return aligned

        if order_id_col not in data.columns:
            raise ValueError(f"data 缺少订单字段: {order_id_col}")
        if isinstance(production_hits, pd.DataFrame):
            if order_id_col not in production_hits.columns:
                raise ValueError(f"production_hits 缺少订单字段: {order_id_col}")
            keyed = production_hits.set_index(order_id_col, drop=False)
        else:
            keyed = production_hits

        order_ids = data[order_id_col]
        missing_orders = order_ids[~order_ids.isin(keyed.index)]
        if not missing_orders.empty:
            sample = missing_orders.drop_duplicates().head(5).tolist()
            raise ValueError(f"production_hits 缺少以下订单: {sample}")

        aligned = keyed.loc[order_ids].copy()
        aligned.index = data.index
        return aligned

    def _production_hits_to_matrix(
        self,
        data: DataFrame,
        production_hits: Union[pd.DataFrame, pd.Series],
        hit_col: Optional[str],
        order_id_col: Optional[str],
    ) -> pd.DataFrame:
        aligned = self._align_production_hits(data, production_hits, order_id_col)
        aliases = self._rule_alias_map()
        matrix = pd.DataFrame(False, index=data.index, columns=self.rule_labels_)

        if isinstance(aligned, pd.Series):
            hit_values = aligned
        else:
            candidate_cols = [col for col in aligned.columns if col != order_id_col]
            matched_cols = {col: aliases[str(col)] for col in candidate_cols if str(col) in aliases}

            if hit_col is not None:
                if hit_col not in aligned.columns:
                    raise ValueError(f"production_hits 缺少命中规则列: {hit_col}")
                hit_values = aligned[hit_col]
            elif "命中规则" in aligned.columns:
                hit_values = aligned["命中规则"]
            elif matched_cols:
                for col, label in matched_cols.items():
                    matrix[label] = aligned[col].fillna(False).astype(bool).values
                return matrix
            elif len(candidate_cols) == 1:
                hit_values = aligned[candidate_cols[0]]
            else:
                raise ValueError("无法识别 production_hits 格式，请传入 hit_col 或规则命中矩阵")

        for index, value in hit_values.items():
            labels = self._normalize_hit_value(value, aliases)
            for label in labels:
                matrix.at[index, label] = True
        return matrix

    def compare(
        self,
        data: DataFrame,
        production_hits: Union[pd.DataFrame, pd.Series],
        hit_col: Optional[str] = None,
        order_id_col: Optional[str] = None,
        include_data: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """对比线上生产命中结果与线下规则执行结果。

        ``production_hits`` 支持三种格式：

        - 单列命中规则：每行是 ``rule.name``、``[rule.name]``、空值或多个规则名列表
        - 指定 ``hit_col``：从 DataFrame 的该列读取命中规则
        - 规则命中矩阵：列名为规则名称/表达式/展示名，值为 ``0/1`` 或 ``True/False``

        :param data: 生产订单原始字段数据 DataFrame
        :param production_hits: 每笔订单线上生产命中规则
        :param hit_col: 命中规则列名；不传时优先识别 ``"命中规则"``，再识别规则命中矩阵
        :param order_id_col: 订单 ID 字段；传入时按订单 ID 对齐，否则按行顺序/索引对齐
        :param include_data: 差异明细是否拼接原始订单字段，默认为 True
        :return: ``(report, diff_detail)``。report 为规则一致性报告，diff_detail 为差异订单明细
        """
        self._validate_data(data)
        offline_prediction = self.predict(data)
        offline_matrix = offline_prediction[self.rule_labels_].fillna(False).astype(bool)
        online_matrix = self._production_hits_to_matrix(data, production_hits, hit_col, order_id_col)

        rows = []
        total = len(data)
        for i, (rule, label) in enumerate(zip(self.rules, self.rule_labels_), start=1):
            consistent = offline_matrix[label].eq(online_matrix[label])
            consistent_count = int(consistent.sum())
            diff_count = total - consistent_count
            rows.append(
                {
                    "规则序号": i,
                    "规则名称": rule.name,
                    "规则表达式": rule.expr,
                    "样本总数": total,
                    "一致样本": consistent_count,
                    "差异样本": diff_count,
                    "一致率": self._rate(consistent_count, total),
                    "差异率": self._rate(diff_count, total),
                }
            )
        report = pd.DataFrame(rows)

        online_names = online_matrix.apply(lambda row: self._join_rule_names(row, self.rule_labels_), axis=1)
        offline_names = offline_matrix.apply(lambda row: self._join_rule_names(row, self.rule_labels_), axis=1)
        diff_mask = offline_matrix.ne(online_matrix).any(axis=1)
        detail = pd.DataFrame(index=data.index)
        detail["线下命中规则"] = offline_names
        detail["线上命中规则"] = online_names
        offline_only = offline_matrix & ~online_matrix
        online_only = online_matrix & ~offline_matrix
        detail["差异规则"] = offline_matrix.ne(online_matrix).apply(
            lambda row: self._join_rule_names(row, self.rule_labels_), axis=1
        )
        detail["线下独有规则"] = offline_only.apply(lambda row: self._join_rule_names(row, self.rule_labels_), axis=1)
        detail["线上独有规则"] = online_only.apply(lambda row: self._join_rule_names(row, self.rule_labels_), axis=1)

        for label in self.rule_labels_:
            detail[f"线下_{label}"] = offline_matrix[label]
            detail[f"线上_{label}"] = online_matrix[label]
            detail[f"是否一致_{label}"] = offline_matrix[label].eq(online_matrix[label])

        detail = detail.loc[diff_mask]
        if include_data:
            detail = pd.concat([data.loc[diff_mask].copy(), detail], axis=1)
        return report, detail

    def predict(self, data: DataFrame) -> pd.DataFrame:
        """计算每笔样本在每条规则上的命中结果。

        串行模式下，前序规则已命中的样本不会继续流转到后续规则，后续规则列填充
        ``<NA>``；并行模式下，每条规则都会对全量样本计算命中结果。返回结果保留
        输入数据索引。

        :param data: 生产拼表后的变量取值明细
        :return: 每条规则命中明细以及规则集汇总列
        """
        self._validate_data(data)

        result = pd.DataFrame(index=data.index)
        hit_any = pd.Series(False, index=data.index, dtype=bool)

        if self.mode == "serial":
            active = pd.Series(True, index=data.index, dtype=bool)
            first_rule_name = pd.Series("", index=data.index, dtype=object)
            first_rule_index = pd.Series("", index=data.index, dtype=object)

            for i, (rule, label) in enumerate(zip(self.rules, self.rule_labels_), start=1):
                rule_hit = pd.Series(pd.NA, index=data.index, dtype="boolean")
                if active.any():
                    evaluated = self._predict_rule(rule, data.loc[active])
                    rule_hit.loc[active] = evaluated
                    hit_index = evaluated[evaluated].index
                    first_rule_name.loc[hit_index] = label
                    first_rule_index.loc[hit_index] = str(i)

                result[label] = rule_hit
                hit = rule_hit.fillna(False).astype(bool)
                hit_any = hit_any | hit
                active = active & ~hit

            result["命中规则序号"] = first_rule_index
            result["命中规则"] = first_rule_name
        else:
            for rule, label in zip(self.rules, self.rule_labels_):
                hit = self._predict_rule(rule, data)
                result[label] = hit
                hit_any = hit_any | hit

            result["命中规则序号"] = result.apply(lambda row: self._join_rule_indices(row, self.rule_labels_), axis=1)
            result["命中规则"] = result.apply(lambda row: self._join_rule_names(row, self.rule_labels_), axis=1)

        result["是否命中"] = hit_any
        result["是否通过"] = ~hit_any
        return result

    def _group_slices(
        self,
        data: DataFrame,
        date_col: Optional[str],
        freq: str,
        group_cols: Optional[Union[str, Sequence[str]]],
        dropna: bool,
    ) -> List[Tuple[Dict[str, object], pd.Index]]:
        group_keys: Dict[str, pd.Series] = {}

        if date_col is not None:
            if date_col not in data.columns:
                raise ValueError(f"数据集缺少日期列: {date_col}")
            parsed = pd.to_datetime(data[date_col], errors="coerce")
            try:
                period = parsed.dt.to_period(freq).astype(str)
            except ValueError as exc:
                raise ValueError(f"freq={freq!r} 无法转换为 pandas Period 频率") from exc
            group_keys["统计周期"] = period.where(parsed.notna(), other=np.nan)

        if group_cols is not None:
            if isinstance(group_cols, str):
                group_cols = [group_cols]
            else:
                group_cols = list(group_cols)
            missing = [col for col in group_cols if col not in data.columns]
            if missing:
                raise ValueError(f"数据集缺少分组字段列: {missing}")
            for col in group_cols:
                group_keys[col] = data[col]

        if not group_keys:
            return [({}, data.index)]

        groups = pd.DataFrame(group_keys, index=data.index)
        if dropna:
            valid = groups.notna().all(axis=1)
            groups = groups.loc[valid]
        else:
            groups = groups.where(groups.notna(), other="缺失")

        if groups.empty:
            return []

        slices = []
        group_names = list(groups.columns)
        grouped = groups.groupby(group_names, sort=True, dropna=False)
        for values, group in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            prefix = dict(zip(group_names, values))
            slices.append((prefix, group.index))
        return slices

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def _report_one(self, data: DataFrame) -> pd.DataFrame:
        prediction = self.predict(data)
        total = len(data)
        rows = []

        for i, (rule, label) in enumerate(zip(self.rules, self.rule_labels_), start=1):
            values = prediction[label]
            if self.mode == "serial":
                current_mask = values.notna()
            else:
                current_mask = pd.Series(True, index=prediction.index, dtype=bool)

            current_total = int(current_mask.sum())
            hit_count = int(values.fillna(False).astype(bool).sum())
            pass_count = current_total - hit_count
            rows.append(
                {
                    "规则序号": i,
                    "规则名称": rule.name,
                    "规则表达式": rule.expr,
                    "样本总数": current_total,
                    "规则命中": hit_count,
                    "命中率(绝对值)": self._rate(hit_count, total),
                    "命中率(相对值)": self._rate(hit_count, current_total),
                    "规则通过": pass_count,
                    "通过率(绝对值)": self._rate(pass_count, total),
                    "通过率(相对值)": self._rate(pass_count, current_total),
                }
            )

        return pd.DataFrame(rows)

    def report(
        self,
        data: DataFrame,
        date_col: Optional[str] = None,
        freq: str = "M",
        group_cols: Optional[Union[str, Sequence[str]]] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """输出每条规则的流转命中报表。

        :param data: 生产拼表后的变量取值明细
        :param date_col: 日期列名，传入后按 ``freq`` 生成 ``统计周期`` 分组
        :param freq: pandas Period 频率，默认 ``"M"`` 月
        :param group_cols: 类别分组字段，支持单列或多列；可与 ``date_col`` 同时使用
        :param dropna: 是否丢弃分组字段缺失样本，默认为 True；False 时归入 ``"缺失"``
        :return: 规则级流转命中报表
        """
        self._validate_data(data)
        rows = []
        for prefix, index in self._group_slices(data, date_col, freq, group_cols, dropna):
            table = self._report_one(data.loc[index])
            for key, value in reversed(prefix.items()):
                table.insert(0, key, value)
            rows.append(table)

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def _summary_one(self, data: DataFrame) -> Dict[str, Union[int, float]]:
        prediction = self.predict(data)
        total = len(data)
        hit_count = int(prediction["是否命中"].sum())
        pass_count = int(prediction["是否通过"].sum())
        return {
            "样本总数": total,
            "通过样本": pass_count,
            "命中样本": hit_count,
            "通过率": self._rate(pass_count, total),
            "命中率": self._rate(hit_count, total),
        }

    def summary(
        self,
        data: DataFrame,
        date_col: Optional[str] = None,
        freq: str = "M",
        group_cols: Optional[Union[str, Sequence[str]]] = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """输出规则集整体通过与命中汇总。

        分组参数与 :meth:`report` 一致；未传入分组时返回一行整体汇总。
        """
        self._validate_data(data)
        rows = []
        for prefix, index in self._group_slices(data, date_col, freq, group_cols, dropna):
            row = {**prefix, **self._summary_one(data.loc[index])}
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"RuleFlow(name={self.name!r}, mode={self.mode!r}, n_rules={len(self.rules)})"
