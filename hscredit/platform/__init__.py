"""可通过 Python 启动的本地风控工作台，Web 依赖按需加载。"""

from .launcher import launch

__all__ = ["launch", "create_app"]


def create_app(data_dir=None, *, token=None):
    """创建 ASGI 应用；用于自定义部署或接口测试。"""
    try:
        from .server import create_app as factory
    except ImportError as exc:
        raise ImportError('风控平台需要可选依赖，请运行 pip install "hscredit[platform]"') from exc
    return factory(data_dir=data_dir, token=token)
