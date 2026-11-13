"""兼容旧版风控平台入口和已保存模型中的模块名。"""

import sys

from .web import create_app, launch

__all__ = ["create_app", "launch"]

if __name__ == "__main__":
    from .cli import main

    raise SystemExit(main(["web", *sys.argv[1:]]))

# 旧版流水线序列化了预处理函数与平台估计器的完整导入路径。
from .web import analysis, models

sys.modules[__name__ + ".analysis"] = analysis
sys.modules[__name__ + ".models"] = models
