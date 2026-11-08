"""数据集工具.

提供示例数据集的加载功能。
"""

import hashlib
import os
import re
import tempfile
from contextlib import closing, contextmanager
from http.client import HTTPException
from importlib.resources import as_file, files
from pathlib import Path
from time import monotonic
from typing import Optional, Sequence, Tuple, Union
from urllib.request import Request, urlopen
import pandas as pd
from pandas.api.types import CategoricalDtype

_DATASET_URL = "https://github.com/hengshu-credit/hscredit/releases/download/v0.1.3/hscredit.parquet"
_GITEE_RAW_URL = "https://gitee.com/itlubber/hscredit/raw/v0.1.3/hscredit/resources/datasets/hscredit.parquet"
_MIN_DOWNLOAD_SPEED = 100 * 1024
_SPEED_WINDOW = 5
_DATASET_SHA256 = "6c4e4790f3a56472d36e58517df349aeb18d3ca709d6b424f51a6556fa8d14f9"
_DATASET_BYTES = 375272455


def _verified_dataset(path):
    if not path.is_file() or path.stat().st_size != _DATASET_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == _DATASET_SHA256


def _dataset_blocks(url):
    """按网络实际到达的数据块读取，低速连接也能及时测速。"""
    request = Request(url, headers={"User-Agent": "hscredit-dataset"})
    try:
        with urlopen(request, timeout=15) as response:
            while block := response.read1(256 * 1024):
                yield block
    except (OSError, HTTPException) as exc:
        raise ConnectionError(f"下载 {url} 失败：{exc}") from exc


def _write_dataset(blocks, path, *, monitor_speed=False, append=False):
    """写入并校验数据；GitHub 的测速窗口低于阈值时暂停并保留迭代位置。"""
    with path.open("ab" if append else "wb") as output:
        total = output.tell()
        bytes_at_window_start = total
        started = monotonic()
        for block in blocks:
            total += len(block)
            if total > _DATASET_BYTES:
                raise ValueError("下载的 hscredit 数据集超出预期大小")
            output.write(block)
            if monitor_speed and total < _DATASET_BYTES:
                now = monotonic()
                elapsed = now - started
                if elapsed >= _SPEED_WINDOW:
                    if total - bytes_at_window_start < _MIN_DOWNLOAD_SPEED * elapsed:
                        return False
                    started = now
                    bytes_at_window_start = total
    if not _verified_dataset(path):
        raise ValueError("下载的 hscredit 数据集大小或 SHA-256 校验失败")
    return True


def _download_dataset(directory):
    """优先下载 GitHub，镜像不可用时可继续尚未完成的低速连接。"""
    primary = directory / "github.part"
    mirror = directory / "gitee.part"
    github_error = None
    with closing(_dataset_blocks(_DATASET_URL)) as github:
        try:
            if _write_dataset(github, primary, monitor_speed=True):
                return primary
        except (ConnectionError, ValueError) as exc:
            github_error = exc

        try:
            with closing(_dataset_blocks(_GITEE_RAW_URL)) as blocks:
                _write_dataset(blocks, mirror)
            return mirror
        except (ConnectionError, ValueError) as exc:
            mirror_error = exc

        if github_error is None:
            try:
                _write_dataset(github, primary, append=True)
                return primary
            except (ConnectionError, ValueError) as exc:
                github_error = exc

    message = f"GitHub 下载失败：{github_error}。Gitee LFS 下载失败：{mirror_error}"
    if isinstance(github_error, ValueError) and isinstance(mirror_error, ValueError):
        raise ValueError(message) from mirror_error
    raise ConnectionError(message) from mirror_error


@contextmanager
def _hscredit_dataset_path():
    """优先使用源码中的 Parquet，安装包通过校验后原子发布到本地缓存。"""
    resource = files("hscredit").joinpath("resources").joinpath("datasets").joinpath("hscredit.parquet")
    if resource.is_file():
        with as_file(resource) as path:
            with path.open("rb") as stream:
                is_pointer = stream.read(42).startswith(b"version https://git-lfs.github.com/spec/v1")
            if not is_pointer:
                yield path
                return

    root = Path(os.environ.get("HSCREDIT_DATA_HOME") or Path.home() / ".cache" / "hscredit").expanduser()
    target = root / "datasets" / _DATASET_SHA256 / "hscredit.parquet"
    if not _verified_dataset(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=target.parent) as directory:
                verified = _download_dataset(Path(directory))
                try:
                    os.replace(verified, target)
                except PermissionError:
                    # Windows 上另一进程可能已完成下载，并打开了经过校验的文件。
                    if not _verified_dataset(target):
                        raise
        except OSError as exc:
            raise OSError(f"无法缓存 hscredit 数据集：{exc}。也可手动从 {_DATASET_URL} 下载至 {target}") from exc
    yield target


_HSCREDIT_ORDER_COLUMNS = frozenset(
    {
        "放款编码",
        "用户编码",
        "loan_amt",
        "periods",
        "放款银行",
        "流量渠道",
        "授信时间",
        "放款时间",
        "etl_date",
        "etl_time",
    }
)


def hscredit_dataset(
    split: bool = False,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """加载 hscredit 现金贷单笔单批放款数据集。PyPI 安装后首次加载会下载并校验缓存。

    **参数**

    :param split: 默认返回全部原始数据；为 True 时依次返回 ``(订单信息, 逾期数据, X)``。
        订单信息包含订单/用户编号、审批额度与期数、资金方、渠道、授信/借款申请时间、
        抽数日期。逾期数据包含 DPD、CURRENT_DPD 和 MOBn；其余授信时点字段归入 X。
    :type split: bool
    :param columns: 仅读取指定原始字段，降低内存开销；拆分时各组也只包含所选字段。
        省略时读取全部 8,516 列。完整 DataFrame 约占 7.16 GiB，加载与拆分另需临时空间。

    :type columns: Optional[Sequence[str]]

    **返回**

    DataFrame 或三个 DataFrame 的元组
        保留源数据的行索引、字段名、类型和值，不填补缺失、不生成标签、不抽样。
        三组内字段保持读取顺序，互不重叠且覆盖全部读取字段。

    **参考样例**

    >>> data = hscredit_dataset()
    >>> order, overdue, X = hscredit_dataset(split=True)
    >>> sample = hscredit_dataset(columns=["用户编码", "age_year", "MOB1"])

    标签需根据表现期和逾期阈值显式构造；原始数据不包含预设的二分类标签。
    详细业务口径见 ``resources/datasets/README.md``。
    """
    if not isinstance(split, bool):
        raise TypeError("split 必须为布尔值")
    if columns is not None:
        if isinstance(columns, (str, bytes)):
            raise TypeError("columns 必须为字段名序列，不能是单个字符串")
        columns = list(columns)
        if not all(isinstance(column, str) for column in columns):
            raise TypeError("columns 中的字段名必须为字符串")
        if len(columns) != len(set(columns)):
            raise ValueError("columns 不能包含重复字段")
    with _hscredit_dataset_path() as path:
        if columns is not None:
            from pyarrow.parquet import ParquetFile

            with ParquetFile(path) as parquet:
                available = set(parquet.schema_arrow.names)
                row_count = parquet.metadata.num_rows
            missing = [column for column in columns if column not in available]
            if missing:
                raise ValueError(f"hscredit 数据集不存在字段：{missing}")
        # pandas/pyarrow 对零列投影会丢失 RangeIndex，显式保留原始行数。
        data = pd.DataFrame(index=pd.RangeIndex(row_count)) if columns == [] else pd.read_parquet(path, columns=columns, engine="pyarrow")
    if not split:
        return data
    order_columns = [column for column in data if column in _HSCREDIT_ORDER_COLUMNS]
    overdue_columns = [column for column in data if column in {"DPD", "CURRENT_DPD"} or re.fullmatch(r"MOB[1-9][0-9]*", column)]
    excluded = set(order_columns) | set(overdue_columns)
    feature_columns = [column for column in data if column not in excluded]
    return data[order_columns].copy(), data[overdue_columns].copy(), data[feature_columns].copy()


def germancredit():
    """加载德国信贷数据集 German Credit Data。

    数据来源：https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

    :return: pd.DataFrame

    **参考样例**

    >>> df = germancredit()
    >>> print(df.shape)
    (1000, 21)
    """
    # 方案1：优先读取仓库内置数据（离线可用）
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "resources",
        "datasets",
        "germancredit.csv",
    )
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        # 本地文件目标列为 creditability：good/bad -> 0/1
        if "creditability" in df.columns:
            df["class"] = df["creditability"].map({"good": 0, "bad": 1})
            df = df.drop(columns=["creditability"])
        elif "class" in df.columns:
            # 兼容 class 列为字符串或数值的情况
            if df["class"].dtype == "O":
                df["class"] = df["class"].map({"good": 0, "bad": 1})
            else:
                df["class"] = df["class"].map({1: 0, 2: 1}).fillna(df["class"])
        return df

    # 方案2：使用 sklearn OpenML（在线）
    try:
        from sklearn.datasets import fetch_openml

        try:
            data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
        except TypeError:
            # 兼容旧版本 sklearn（无 parser 参数）
            data = fetch_openml(name="credit-g", version=1, as_frame=True)

        df = data.frame
        df["class"] = df["class"].map({"good": 0, "bad": 1})
        return df
    except Exception:
        # 方案3：从 UCI 加载
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

        column_names = [
            "status_of_existing_checking_account",
            "duration_in_month",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings_account_and_bonds",
            "present_employment_since",
            "installment_rate_in_percentage_of_disposable_income",
            "personal_status_and_sex",
            "other_debtors_or_guarantors",
            "present_residence_since",
            "property",
            "age_in_years",
            "other_installment_plans",
            "housing",
            "number_of_existing_credits_at_this_bank",
            "job",
            "number_of_people_being_liable_to_provide_maintenance_for",
            "telephone",
            "foreign_worker",
            "class",
        ]

        df = pd.read_csv(url, sep=" ", names=column_names, header=None)

        # 转换目标变量 (1=good, 2=bad -> 0=good, 1=bad)
        df["class"] = df["class"].map({1: 0, 2: 1})

        return df


def load_titanic():
    """加载泰坦尼克号数据集。

    :return: pd.DataFrame

    **参考样例**

    >>> df = load_titanic()
    """
    try:
        import seaborn as sns

        return sns.load_dataset("titanic")
    except Exception:
        url = "https://raw.githubusercontent.com/datasciencedoc/data/master/titanic.csv"
        return pd.read_csv(url)


def load_iris():
    """加载鸢尾花数据集。

    :return: pd.DataFrame

    **参考样例**

    >>> df = load_iris()
    """
    from sklearn.datasets import load_iris as sklearn_load_iris

    iris = sklearn_load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df
