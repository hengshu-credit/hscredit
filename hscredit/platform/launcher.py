"""风控平台的前台、Notebook 后台启动与停止接口。"""

import ipaddress
import socket
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote


class PlatformServer:
    """后台服务句柄。

    **参数**

    :param server: Uvicorn 服务。
    :param sock: 已绑定的监听套接字。
    :param url: 工作台访问链接。

    **属性**

    :ivar url: 可直接在浏览器访问的链接。

    **参考样例**

    >>> server = launch(background=True)  # doctest: +SKIP
    >>> server.stop()  # doctest: +SKIP
    """

    def __init__(self, server, sock, url):
        self.server, self.sock, self.url = server, sock, url
        self.thread = None

    def stop(self):
        """停止服务和未完成任务，保留配置供重新执行。"""
        self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("后台任务仍在结束中，请稍后再次调用 stop()")
        self.sock.close()

    def add_dataset(self, data, name="Python 数据集"):
        """将内存中的 DataFrame 注册到工作台。"""
        return self.server.config.app.state.workspace.add_dataset(data, name)

    def __repr__(self):
        return f"hscredit 风控平台：{self.url}"


def launch(host="127.0.0.1", port=8765, *, data_dir=None, background=False, open_browser=True, token=None, demo=False):
    """启动风控平台，返回带有 ``url``、``add_dataset`` 和 ``stop`` 的服务句柄。

    默认只监听本机；局域网部署必须显式传入访问令牌 ``token``。
    ``port=0`` 自动选择空闲端口；Notebook 中使用 ``background=True``。
    数据默认保存在用户主目录的 ``.hscredit/platform``，不会写入项目目录。
    """
    try:
        import uvicorn
        from . import create_app
    except ImportError as exc:
        raise ImportError('请先运行 pip install "hscredit[platform]"') from exc
    if not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("端口必须在 0 到 65535 之间")
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and (not token or len(token) < 16):
        raise ValueError("非本机监听必须设置至少 16 位的访问令牌 token")
    app = create_app(data_dir=data_dir, token=token)
    if demo:
        from .demo import load_demo

        load_demo(app.state.workspace)
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    try:
        sock.bind((host, port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    bound_port = sock.getsockname()[1]
    address = "127.0.0.1" if host == "0.0.0.0" else "[::1]" if host == "::" else f"[{host}]" if ":" in host else host
    url = f"http://{address}:{bound_port}/" + (f"#token={quote(token, safe='')}" if token else "")
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    handle = PlatformServer(server, sock, url)

    def show_when_ready():
        for _ in range(200):
            if server.started:
                if open_browser:
                    webbrowser.open(url)
                return
            if server.should_exit:
                return
            time.sleep(0.05)

    if background:
        handle.thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        handle.thread.start()
        show_when_ready()
        if not server.started:
            handle.stop()
            raise RuntimeError("平台未能正常启动，请检查服务日志")
    else:
        threading.Thread(target=show_when_ready, daemon=True).start()
    print(f"hscredit 风控平台：{url}")
    print(f"数据目录：{Path(app.state.workspace.root)}")
    if not background:
        try:
            server.run(sockets=[sock])
        finally:
            sock.close()
    return handle
