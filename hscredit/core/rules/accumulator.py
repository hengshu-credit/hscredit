"""固定规则的可合并流式计数，不保留训练数据或命中明细。"""

import numpy as np

from ...utils.data_contracts import validate_target


class RuleAccumulator:
    """update/merge/finalize 协议；分批与整表计数严格相等。"""

    def __init__(self, rule, target=None):
        from .rule import Rule

        self.rule = Rule(rule.expr if isinstance(rule, Rule) else rule, n_jobs=1)
        self.target = target
        self.total = self.matched = self.bad = self.matched_bad = 0

    def update(self, batch):
        labels = (
            None
            if self.target is None
            else validate_target(
                batch[self.target],
                target_type="binary",
                allow_empty=True,
            )
        )
        mask = self.rule.predict(batch).to_numpy(dtype=bool)
        # 成功计算之后提交，任何异常不留下部分计数。
        matched_bad = int(np.asarray(labels)[mask].sum()) if labels is not None else 0
        bad = int(np.asarray(labels).sum()) if labels is not None else 0
        self.total += len(batch)
        self.matched += int(mask.sum())
        self.bad += bad
        self.matched_bad += matched_bad
        self.rule.result_ = None
        return self

    def merge(self, other):
        if not isinstance(other, RuleAccumulator) or (self.rule.expr, self.target) != (other.rule.expr, other.target):
            raise ValueError("只能合并相同规则和目标口径的统计")
        self.total += other.total
        self.matched += other.matched
        self.bad += other.bad
        self.matched_bad += other.matched_bad
        return self

    def finalize(self):
        result = {"样本总数": self.total, "命中样本数": self.matched, "未命中样本数": self.total - self.matched}
        if self.target is not None:
            result.update(
                {"坏样本数": self.bad, "命中坏样本数": self.matched_bad, "未命中坏样本数": self.bad - self.matched_bad}
            )
        return result
