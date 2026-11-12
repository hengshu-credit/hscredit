"""允许 python -m hscredit.platform 启动工作台。"""

import sys
from ..cli import main

raise SystemExit(main(["platform", *sys.argv[1:]]))
