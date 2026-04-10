"""数据IO工具.

提供 pickle 文件的读写功能，支持多种序列化引擎和压缩格式。
"""

import gzip
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Union

import joblib

from ..exceptions import DependencyError, ValidationError


def _open_file(
    file: Union[str, Path],
    mode: str = 'rb',
    compression: Optional[str] = None
):
    """根据压缩格式打开文件."""
    if compression == 'gzip' or str(file).lower().endswith('.gz'):
        return gzip.open(file, mode)
    elif compression == 'bz2' or str(file).lower().endswith('.bz2'):
        import bz2
        return bz2.open(file, mode)
    elif compression == 'xz' or str(file).lower().endswith('.xz'):
        try:
            import lzma
            return lzma.open(file, mode)
        except ImportError:
            raise DependencyError("缺少可选依赖 lzma，请改用 gzip 或 bz2 压缩格式")
    elif compression == 'lz4' or str(file).lower().endswith('.lz4'):
        try:
            import lz4.frame
            return lz4.frame.open(file, mode)
        except ImportError:
            raise DependencyError("缺少可选依赖 lz4，请先安装: pip install lz4")
    elif compression in ('zstd', 'zstandard') or str(file).lower().endswith(('.zst', '.zstd')):
        try:
            import zstandard
            return zstandard.open(file, mode)
        except ImportError:
            raise DependencyError("缺少可选依赖 zstandard，请先安装: pip install zstandard")
    else:
        return open(file, mode)


def load_pickle(
    file: Union[str, Path],
    engine: str = "auto",
    compression: Optional[str] = None
) -> Any:
    """导入 pickle 文件。

    支持多种序列化引擎（joblib/dill/cloudpickle/pickle）和压缩格式
    （gzip/bz2/xz/lz4/zstd）。支持根据文件扩展名自动检测。

    :param file: pickle 文件路径，支持 .pkl, .pkl.gz, .joblib, .dill 等格式
    :param engine: 使用的序列化引擎，可选：
        - 'auto': 自动检测（根据文件内容和扩展名推断，默认）
        - 'joblib': 使用 joblib（推荐用于 numpy/scipy/sklearn 对象）
        - 'dill': 使用 dill（支持 lambda、嵌套函数等复杂对象）
        - 'cloudpickle': 使用 cloudpickle（常用于分布式计算如 PyTorch/Spark）
        - 'pickle': 使用标准库 pickle
    :param compression: 压缩格式，可选：
        - None: 根据文件扩展名自动检测（.gz/.bz2/.xz/.lz4/.zst）
        - 'gzip'/'gz': gzip 压缩
        - 'bz2': bzip2 压缩
        - 'xz': xz/lzma 压缩
        - 'lz4': lz4 压缩（需安装 lz4）
        - 'zstd'/'zstandard': zstd 压缩（需安装 zstandard）
    :return: 反序列化后的对象

    示例:
        >>> # 自动检测
        >>> data = load_pickle('model.pkl')
        >>> data = load_pickle('model.pkl.gz')  # 自动解压
        >>>
        >>> # 指定引擎
        >>> data = load_pickle('model.dill', engine='dill')
        >>> data = load_pickle('model.pkl', engine='cloudpickle')
        >>>
        >>> # 指定压缩
        >>> data = load_pickle('model.pkl', compression='gzip')
    """
    file_str = str(file).lower()

    # 自动检测压缩格式
    comp = compression
    if comp is None:
        if file_str.endswith('.gz') or file_str.endswith('.gzip'):
            comp = 'gzip'
        elif file_str.endswith('.bz2'):
            comp = 'bz2'
        elif file_str.endswith('.xz'):
            comp = 'xz'
        elif file_str.endswith('.lz4'):
            comp = 'lz4'
        elif file_str.endswith('.zst') or file_str.endswith('.zstd'):
            comp = 'zstd'

    # 自动检测引擎
    eng = engine
    if eng == "auto":
        # 根据文件扩展名推断
        if file_str.endswith('.joblib') or file_str.endswith('.joblib.gz'):
            eng = 'joblib'
        elif file_str.endswith('.dill') or file_str.endswith('.dill.gz'):
            eng = 'dill'
        elif file_str.endswith('.cloudpickle'):
            eng = 'cloudpickle'
        else:
            # 默认使用 joblib（业内最常用）
            eng = 'joblib'

    # 使用指定引擎加载
    if eng == "joblib":
        if comp:
            # joblib 需要特殊处理压缩文件
            with _open_file(file, 'rb', comp) as f:
                data: bytes = f.read()  # type: ignore
                buf = BytesIO(data)
                return joblib.load(buf)
        return joblib.load(file)

    elif eng == "dill":
        try:
            import dill
            with _open_file(file, "rb", comp) as f:
                return dill.load(f)
        except ImportError:
            raise DependencyError("缺少可选依赖 dill，请先安装: pip install dill")

    elif eng == "cloudpickle":
        try:
            import cloudpickle
            with _open_file(file, "rb", comp) as f:
                return cloudpickle.load(f)
        except ImportError:
            raise DependencyError("缺少可选依赖 cloudpickle，请先安装: pip install cloudpickle")

    elif eng == "pickle":
        with _open_file(file, "rb", comp) as f:
            return pickle.load(f)

    else:
        raise ValidationError(
            f"engine 目前只支持 ['auto', 'joblib', 'dill', 'cloudpickle', 'pickle'], "
            f"不支持 {eng}"
        )


def save_pickle(
    obj: Any,
    file: Union[str, Path],
    engine: str = "joblib",
    compression: Optional[str] = None,
    compression_level: Optional[int] = None,
    protocol: Optional[int] = None
) -> str:
    """保存数据至 pickle 文件。

    支持多种序列化引擎（joblib/dill/cloudpickle/pickle）和压缩格式
    （gzip/bz2/xz/lz4/zstd），可处理大型模型和复杂对象。

    :param obj: 需要保存的数据对象
    :param file: 文件路径，建议扩展名 .pkl, .joblib, .dill 等
    :param engine: 使用的序列化引擎，可选：
        - 'joblib': joblib（默认，推荐用于 numpy/scipy/sklearn 对象）
        - 'dill': dill（支持 lambda、嵌套函数等复杂对象）
        - 'cloudpickle': cloudpickle（常用于分布式计算）
        - 'pickle': 标准库 pickle
    :param compression: 压缩格式，可选：
        - None: 不压缩（默认）
        - 'gzip'/'gz': gzip 压缩（兼容性好）
        - 'bz2': bzip2 压缩（压缩率高但较慢）
        - 'xz': xz/lzma 压缩（最高压缩率）
        - 'lz4': lz4 压缩（速度最快，需安装 lz4）
        - 'zstd'/'zstandard': zstd 压缩（速度与压缩率平衡，需安装 zstandard）
        - 'auto': 根据文件扩展名自动选择
    :param compression_level: 压缩级别（1-9，数字越大压缩率越高，默认取决于压缩算法）
    :param protocol: pickle 协议版本（默认使用最高可用版本）
    :return: 保存的文件路径

    示例:
        >>> # 基本用法（默认 joblib）
        >>> save_pickle(model, 'model.pkl')
        >>>
        >>> # 使用 dill 保存复杂对象
        >>> save_pickle(lambda_func, 'func.dill', engine='dill')
        >>>
        >>> # 使用 cloudpickle（适用于分布式计算）
        >>> save_pickle(model, 'model.pkl', engine='cloudpickle')
        >>>
        >>> # gzip 压缩（自动根据扩展名）
        >>> save_pickle(model, 'model.pkl.gz')
        >>>
        >>> # 显式指定压缩
        >>> save_pickle(model, 'model.pkl', compression='zstd', compression_level=3)
        >>>
        >>> # 最高压缩率
        >>> save_pickle(model, 'model.pkl.xz', compression='xz')
    """
    file_str = str(file).lower()

    # 自动检测压缩格式
    comp = compression
    if comp == 'auto' or comp is None:
        if file_str.endswith('.gz') or file_str.endswith('.gzip'):
            comp = 'gzip'
        elif file_str.endswith('.bz2'):
            comp = 'bz2'
        elif file_str.endswith('.xz'):
            comp = 'xz'
        elif file_str.endswith('.lz4'):
            comp = 'lz4'
        elif file_str.endswith('.zst') or file_str.endswith('.zstd'):
            comp = 'zstd'

    # 设置 pickle protocol
    proto = protocol
    if proto is None:
        proto = pickle.HIGHEST_PROTOCOL

    # 序列化
    if engine == "joblib":
        if comp:
            # joblib 对压缩支持有限，先序列化到内存再压缩
            buf = BytesIO()
            joblib.dump(obj, buf)
            buf.seek(0)
            with _open_file(file, "wb", comp) as f:
                f.write(buf.getvalue())  # type: ignore
        else:
            joblib.dump(obj, file)

    elif engine == "dill":
        try:
            import dill
            with _open_file(file, "wb", comp) as f:
                dill.dump(obj, f, protocol=proto)
        except ImportError:
            raise DependencyError("缺少可选依赖 dill，请先安装: pip install dill")

    elif engine == "cloudpickle":
        try:
            import cloudpickle
            with _open_file(file, "wb", comp) as f:
                cloudpickle.dump(obj, f, protocol=proto)
        except ImportError:
            raise DependencyError("缺少可选依赖 cloudpickle，请先安装: pip install cloudpickle")

    elif engine == "pickle":
        with _open_file(file, "wb", comp) as f:
            pickle.dump(obj, f, protocol=proto)

    else:
        raise ValidationError(
            f"engine 目前只支持 ['joblib', 'dill', 'cloudpickle', 'pickle'], "
            f"不支持 {engine}"
        )

    return str(file)
