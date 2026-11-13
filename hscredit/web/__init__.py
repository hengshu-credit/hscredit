"""可通过 Python 启动的本地风控工作台，Web 依赖按需加载。"""

from .launcher import launch

__all__ = ["launch", "create_app"]


def create_app(data_dir=None, *, token=None, max_concurrent_jobs=None):
    """创建用于自定义部署或接口测试的 ASGI 应用。

    :param data_dir: 数据集、任务与报告的存储目录，默认使用用户主目录下的 ``.hscredit/platform``。
    :param token: 保护平台 API 的访问令牌，默认仅允许本机访问。
    :param max_concurrent_jobs: 同时运行的任务数，范围为 1 至 10，默认 2。
    :return: 配置好工作区和平台路由的 FastAPI 应用。
    """
    try:
        from .server import create_app as factory
    except ImportError as exc:
        raise ImportError('风控平台需要可选依赖，请运行 pip install "hscredit[web]"') from exc
    return factory(data_dir=data_dir, token=token, max_concurrent_jobs=max_concurrent_jobs)
