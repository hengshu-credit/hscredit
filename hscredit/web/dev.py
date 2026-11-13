"""为本地热更新服务创建带实例标识的平台应用。"""

import os
from uuid import uuid4

from . import create_app


def create_dev_app():
    """每次重载后提供新标识，供开发页面重新读取后端配置。"""
    app = create_app(data_dir=os.environ.get("HSCREDIT_PLATFORM_DATA_DIR"), token=os.environ.get("HSCREDIT_PLATFORM_TOKEN"))
    app.state.dev_instance = uuid4().hex
    return app
