"""调参结果保留策略：完整、预测、摘要、最佳试验和逐折磁盘制品。"""

from hashlib import sha256
from pathlib import Path

from .._lifecycle import atomic_save_pickle

RETENTION_MODES = frozenset({"full", "predictions", "summary", "best", "disk"})


def resolve_retention(retention, store_models, artifact_dir):
    """显式策略优先；未指定时保持 store_models 的历史行为。"""
    mode = retention if retention is not None else ("full" if store_models else "predictions")
    if mode not in RETENTION_MODES:
        raise ValueError(f"retention 必须是 {sorted(RETENTION_MODES)} 之一")
    if mode == "disk" and artifact_dir is None:
        raise ValueError("retention='disk' 必须设置 artifact_dir")
    return mode


def trial_path(tuner, number, fold=None):
    """同一 Study 使用稳定的制品目录，磁盘档每折独立写入。"""
    key = sha256(tuner.study_.study_name.encode()).hexdigest()[:16]
    directory = Path(tuner.artifact_dir).resolve() / key
    name = f"trial_{number}.pkl" if fold is None else f"trial_{number}_fold_{fold}.pkl"
    return directory / name


def compact_training_record(record):
    """摘要不携带验证矩阵、样本参数、回调及其捕获对象。"""
    keys = {"开始时间", "样本数", "依赖版本", "状态", "错误类型", "错误信息", "耗时秒", "最佳迭代", "最佳得分"}
    return {name: value for name, value in record.items() if name in keys}


def compact_fold(record):
    """保留指标/曲线/异常与样本数，不保留逐样本数组和折模型。"""
    keys = {"折编号", "状态", "指标", "评估曲线", "最佳迭代", "启用早停", "错误类型", "错误信息", "制品路径", "评估口径", "补充LIFT口径"}
    result = {name: value for name, value in record.items() if name in keys}
    for name, size_name in (("训练位置", "训练样本数"), ("验证位置", "验证样本数")):
        result[size_name] = len(record[name]) if name in record else record.get(size_name, 0)
    result["训练记录"] = compact_training_record(record.get("训练记录", {}))
    return result


def retain_fold(tuner, number, record):
    """在一折结束后应用策略；磁盘写入成功后才释放该折的内存对象。"""
    mode = tuner.retention_
    if mode == "disk":
        path = trial_path(tuner, number, record["折编号"])
        atomic_save_pickle(record, path, engine="cloudpickle")
        retained = compact_fold(record)
        retained["制品路径"] = str(path)
    elif mode == "summary":
        retained = compact_fold(record)
    elif mode == "predictions":
        record.pop("模型", None)
        record["训练记录"] = compact_training_record(record.get("训练记录", {}))
        return
    else:
        return
    record.clear()
    record.update(retained)


def load_fold(record, tuner=None):
    """读取单折制品，不把模型重新放入调参器常驻缓存。"""
    if "制品路径" not in record:
        return record
    from ....utils import load_pickle

    path = Path(record["制品路径"])
    if tuner is not None and tuner.artifact_dir is not None:
        relocated = trial_path(tuner, 0).parent / path.name
        if relocated.is_file():
            path = relocated
    if not path.is_file():
        raise ValueError(f"调参折制品不存在: {path}，请一起保留 artifact_dir")
    return load_pickle(path, engine="cloudpickle")
