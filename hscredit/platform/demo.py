"""复用统一样例字段加载完整现金贷数据，并显式构造演示标签。"""

import numpy as np
import pandas as pd

CONTEXT_COLUMNS = {"用户编码", "授信时间", "loan_amt", "periods", "流量渠道", "MOB1", "MOB3", "MOB6"}
DEMO_TARGET = "MOB6大于30天"


def load_demo(workspace):
    """导入统一 88 列，不抽样；额外添加一个保留无效值为缺失的二分类标签。"""
    from hscredit.utils.datasets import HSCREDIT_EXAMPLE_COLUMNS, hscredit_dataset

    with workspace.lock:
        existing = next((d for d in workspace.list("datasets") if d.get("source") == "hscredit_dataset"), None)
        if existing:
            return existing
        data = hscredit_dataset(columns=HSCREDIT_EXAMPLE_COLUMNS)
        overdue = pd.to_numeric(data["MOB6"], errors="coerce")
        valid = overdue.notna() & np.isfinite(overdue) & overdue.ge(0)
        data[DEMO_TARGET] = (overdue > 30).astype(float).where(valid)
        metadata = workspace.add_dataset(data, "hscredit_dataset · 统一演示字段")
        return workspace.update("datasets", metadata["id"], source="hscredit_dataset", default_target=DEMO_TARGET, default_features=[c for c in HSCREDIT_EXAMPLE_COLUMNS if c not in CONTEXT_COLUMNS], default_date="授信时间", default_amount="loan_amt", default_overdue=["MOB1", "MOB3", "MOB6"], source_columns=list(HSCREDIT_EXAMPLE_COLUMNS), note="完整订单数据，88 个原始样例字段及 MOB6 > 30 标签；负值、非数值和缺失逾期均保留为缺失标签。TOP32 特征来自全量摘要预选，本数据用于功能演示，不代表独立泛化评估。")
