"""允许 python -m hscredit.web 启动工作台。"""

import sys
from ..cli import main

raise SystemExit(main(["web", *sys.argv[1:]]))
