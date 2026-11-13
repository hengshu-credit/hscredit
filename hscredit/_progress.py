"""按运行环境创建文本或 Jupyter 控件进度条。"""

import sys


def create_progress_bar(**kwargs):
    """仅在活动的 Jupyter 内核中尝试加载控件，缺少控件时使用文本进度条。"""
    from tqdm import tqdm

    get_ipython = getattr(sys.modules.get("IPython"), "get_ipython", None)
    shell = get_ipython() if get_ipython else None
    if "IPKernelApp" in getattr(shell, "config", {}):
        try:
            from tqdm.notebook import IProgress, tqdm as notebook_tqdm
        except ImportError:
            IProgress = None
        if IProgress is not None:
            tqdm = notebook_tqdm
    return tqdm(**kwargs)
