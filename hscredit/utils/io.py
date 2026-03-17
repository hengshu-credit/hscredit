"""数据IO工具.

提供 pickle 文件的读写功能。
"""

import pickle
import joblib


def load_pickle(file, engine: str = "joblib"):
    """导入 pickle 文件。

    :param file: pickle 文件路径
    :param engine: 使用的引擎，可选 'joblib', 'dill', 'pickle'，默认为 'joblib'
    :return: pickle 文件的内容

    示例:
        >>> data = load_pickle('model.pkl')
        >>> data = load_pickle('model.pkl', engine='dill')
    """
    if engine == "joblib":
        return joblib.load(file)
    elif engine == "dill":
        try:
            import dill
            with open(file, "rb") as f:
                return dill.load(f)
        except ImportError:
            raise ImportError("dill is not installed. Install it with: pip install dill")
    elif engine == "pickle":
        with open(file, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"engine 目前只支持 ['joblib', 'dill', 'pickle'], 不支持 {engine}")


def save_pickle(obj, file, engine: str = "joblib"):
    """保存数据至 pickle 文件。

    :param obj: 需要保存的数据
    :param file: 文件路径
    :param engine: 使用的引擎，可选 'joblib', 'dill', 'pickle'，默认为 'joblib'

    示例:
        >>> save_pickle(model, 'model.pkl')
        >>> save_pickle(model, 'model.pkl', engine='dill')
    """
    if engine == "joblib":
        return joblib.dump(obj, file)
    elif engine == "dill":
        try:
            import dill
            with open(file, "wb") as f:
                return dill.dump(obj, f)
        except ImportError:
            raise ImportError("dill is not installed. Install it with: pip install dill")
    elif engine == "pickle":
        with open(file, "wb") as f:
            return pickle.dump(obj, f)
    else:
        raise ValueError(f"engine 目前只支持 ['joblib', 'dill', 'pickle'], 不支持 {engine}")
