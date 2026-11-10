"""为 Optuna Dashboard 提供 hscredit 品牌主题与命令行服务。"""

import gzip
import hashlib
import json
import re
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Optional
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server
from wsgiref.util import setup_testing_defaults

from .utils.colors import CONDITIONAL_BLUE, EXTENDED_COLORS, METRIC_ACCENT_COLORS, PRIMARY_COLORS
from .utils.fonts import FONT_NAME, get_bundled_font_path

_ASSET_PREFIX = "/_hscredit_dashboard/"
_RESOURCE_DIR = Path(__file__).parent / "resources" / "dashboard"
_PLOTLY_EXPORT = re.compile(rb"(?P<exports>[\w$]+)\._doPlot=(?P<core>[\w$]+)\._doPlot,(?P=exports)\.newPlot=(?P=core)\.newPlot")


def _theme_config():
    """从公共颜色定义生成浏览器主题，避免维护第二份品牌色板。"""
    return {
        "font": FONT_NAME,
        "primary": PRIMARY_COLORS[0],
        "secondary": PRIMARY_COLORS[1],
        "lightBlue": CONDITIONAL_BLUE,
        "series": list(EXTENDED_COLORS),
        "metrics": dict(METRIC_ACCENT_COLORS),
    }


def _patch_plotly_bundle(bundle: bytes) -> bytes:
    """在上游 Plotly 公共导出位置安装绘图主题适配器。"""
    matches = list(_PLOTLY_EXPORT.finditer(bundle))
    if not 1 <= len(matches) <= 8:
        raise RuntimeError("当前 Optuna Dashboard 的前端结构不受 hscredit 主题适配器支持，请使用已验证的 optuna-dashboard 0.20.0，或更新 hscredit。")
    return _PLOTLY_EXPORT.sub(lambda match: b"window.__hscreditDashboard.wrapPlotly(" + match.group("core") + b")," + match.group(0), bundle)


def _read_response(app, environ):
    """读取需要改写的页面响应；其他接口保留原生流式处理。"""
    response = {}
    chunks = []

    def capture(status, headers, exc_info=None):
        response.update(status=status, headers=headers)
        return chunks.append

    iterator = app(environ, capture)
    try:
        chunks.extend(iterator)
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()
    body = b"".join(chunks)
    if any(name.lower() == "content-encoding" and value.lower() == "gzip" for name, value in response["headers"]):
        body = gzip.decompress(body)
    return response["status"], response["headers"], body


class _ThemedDashboard:
    """装饰 Dashboard 页面和静态资源，API 与下载接口直接交给上游。"""

    def __init__(self, app):
        self.app = app
        config = _theme_config()
        javascript = "window.__hscreditDashboardConfig = " + json.dumps(config, ensure_ascii=False) + ";\n" + (_RESOURCE_DIR / "theme.js").read_text(encoding="utf-8")
        stylesheet = (_RESOURCE_DIR / "theme.css").read_text(encoding="utf-8")
        for name, value in {"FONT": FONT_NAME, "PRIMARY": config["primary"], "SECONDARY": config["secondary"], "LIGHT_BLUE": CONDITIONAL_BLUE}.items():
            stylesheet = stylesheet.replace("__" + name + "__", value)
        self.assets = {"theme.js": ("application/javascript; charset=utf-8", javascript.encode("utf-8")), "theme.css": ("text/css; charset=utf-8", stylesheet.encode("utf-8")), "font.ttf": ("font/ttf", get_bundled_font_path().read_bytes())}
        # 提前校验实际安装版本，避免浏览器打开后才发现主题无法加载。
        environ = {}
        setup_testing_defaults(environ)
        environ.update(PATH_INFO="/static/bundle.js", HTTP_ACCEPT_ENCODING="identity")
        status, _, bundle = _read_response(app, environ)
        if not status.startswith("200"):
            raise RuntimeError("无法读取 Optuna Dashboard 前端资源，不能启动 hscredit 主题。")
        self.assets["bundle.js"] = ("application/javascript; charset=utf-8", _patch_plotly_bundle(bundle))

    def _asset(self, name, environ, start_response):
        if name not in self.assets:
            start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
            return [] if environ.get("REQUEST_METHOD") == "HEAD" else ["资源不存在".encode("utf-8")]
        content_type, body = self.assets[name]
        etag = '"' + hashlib.sha256(body).hexdigest() + '"'
        headers = [("Content-Type", content_type), ("Cache-Control", "no-cache"), ("ETag", etag)]
        if environ.get("HTTP_IF_NONE_MATCH") == etag:
            start_response("304 Not Modified", headers)
            return []
        start_response("200 OK", [*headers, ("Content-Length", str(len(body)))])
        return [] if environ.get("REQUEST_METHOD") == "HEAD" else [body]

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        method = environ.get("REQUEST_METHOD", "GET")
        if method not in {"GET", "HEAD"}:
            return self.app(environ, start_response)
        if path == "/static/bundle.js":
            return self._asset("bundle.js", environ, start_response)
        if path.startswith(_ASSET_PREFIX):
            return self._asset(path[len(_ASSET_PREFIX) :], environ, start_response)
        if path != "/dashboard" and not path.startswith("/dashboard/"):
            return self.app(environ, start_response)
        request = {key: value for key, value in environ.items() if key not in {"HTTP_IF_NONE_MATCH", "HTTP_IF_MODIFIED_SINCE", "HTTP_RANGE"}}
        request.update(REQUEST_METHOD="GET", HTTP_ACCEPT_ENCODING="identity")
        status, headers, body = _read_response(self.app, request)
        if status.startswith("200") and any(name.lower() == "content-type" and "text/html" in value for name, value in headers):
            tags = ('<link rel="stylesheet" href="' + _ASSET_PREFIX + 'theme.css">' + '<script src="' + _ASSET_PREFIX + 'theme.js"></script>').encode("utf-8")
            body = body.replace(b"<head>", b"<head>" + tags, 1)
        headers = [(name, value) for name, value in headers if name.lower() not in {"content-length", "content-encoding", "etag", "last-modified", "accept-ranges", "cache-control"}]
        start_response(status, [*headers, ("Content-Length", str(len(body))), ("Cache-Control", "no-cache")])
        return [] if method == "HEAD" else [body]


def create_dashboard_app(storage, *, artifact_store=None, llm_provider=None):
    """创建使用 hscredit 配色和内置字体的 Optuna WSGI 应用。

    :param storage: Optuna 存储 URL 或 BaseStorage 实例。
    :param artifact_store: 可选的 Optuna 产物存储。
    :param llm_provider: 可选的 Optuna Dashboard LLM 接口。
    :return: 可传给 WSGI 服务器的应用。
    """
    try:
        from optuna_dashboard import wsgi
    except ImportError as exc:
        raise ImportError('启动主题看板需要可选依赖，请运行 pip install "hscredit[tune]"。') from exc
    return _ThemedDashboard(wsgi(storage, artifact_store=artifact_store, llm_provider=llm_provider))


def run_dashboard(storage: str, *, host: str = "127.0.0.1", port: int = 8080, artifact_dir: Optional[str] = None, quiet: bool = False):
    """运行 hscredit 主题看板，按 Ctrl+C 关闭服务。"""
    artifact_store = None
    if artifact_dir:
        from optuna.artifacts import FileSystemArtifactStore

        artifact_store = FileSystemArtifactStore(artifact_dir)
    app = create_dashboard_app(storage, artifact_store=artifact_store)

    class Server(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    class Handler(WSGIRequestHandler):
        def log_message(self, format, *args):
            if not quiet:
                super().log_message(format, *args)

    with make_server(host, port, app, server_class=Server, handler_class=Handler) as server:
        print(f"hscredit 调参看板已启动：http://{host}:{server.server_port}/dashboard/", flush=True)
        print("按 Ctrl+C 停止服务。", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
