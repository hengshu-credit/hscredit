"""数据库 SSH 认证、转发和同步连接的生命周期。"""

import asyncio
import hmac
import os
import re
import ssl
from concurrent.futures import Future
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from threading import Event, Thread
from typing import Optional, Union
from urllib.parse import urlsplit

from ..exceptions import DependencyError
from .exceptions import DatabaseConnectionError

_AUTH_METHODS = {"password": ("password",), "key": ("publickey",), "key_password": ("publickey", "password")}
_TCP_PORTS = {"mysql": 3306, "starrocks": 9030, "clickhouse": 8123, "hive": 10000, "impala": 21050}
TCP_DATABASES = frozenset((*_TCP_PORTS, "oracle"))
LOCAL_HOST = "127.0.0.1"


@dataclass(frozen=True, repr=False)
class SSHTunnelConfig:
    host: str
    username: str
    port: int = 22
    auth: str = "password"
    password: Optional[str] = None
    private_key: Union[str, bytes] = ""
    private_key_path: Optional[Union[str, os.PathLike]] = None
    passphrase: Optional[str] = None
    host_fingerprint: str = ""
    known_hosts: Optional[Union[str, os.PathLike]] = None

    def ssh_options(self, ssh):
        if not all(isinstance(value, str) for value in (self.host, self.username, self.auth, self.host_fingerprint)):
            raise ValueError("SSH 主机、用户名、认证方式和指纹须为文本")
        if not all(value is None or isinstance(value, str) for value in (self.password, self.passphrase)):
            raise ValueError("SSH 密码和私钥口令须为文本")
        if self.auth not in _AUTH_METHODS:
            raise ValueError("SSH 认证方式须为 password、key 或 key_password")
        if not self.host.strip() or not self.username.strip() or (self.auth != "key" and not self.password):
            raise ValueError("请补全 SSH 主机、用户名和所选方式的登录凭据")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("SSH 端口须为 1 到 65535 的整数")
        key = None
        if self.auth != "password":
            if self.private_key and self.private_key_path is not None:
                raise ValueError("请选择 SSH 私钥内容或私钥路径其中一项")
            try:
                passphrase = self.passphrase or None
                if self.private_key_path is not None:
                    key = ssh.read_private_key(os.path.expanduser(os.fsdecode(self.private_key_path)), passphrase)
                else:
                    key = ssh.import_private_key(self.private_key.strip(), passphrase)
            except (AttributeError, OSError, TypeError, ValueError, ssh.KeyImportError, ssh.KeyEncryptionError):
                raise ValueError("无法读取 SSH 私钥，请检查私钥文件及其解密口令") from None

        options = dict(
            host=self.host.strip(),
            port=self.port,
            username=self.username.strip(),
            password=self.password if self.auth != "key" else None,
            client_keys=[key] if key is not None else None,
            preferred_auth=list(_AUTH_METHODS[self.auth]),
            agent_path=None,
            config=None,
            connect_timeout=15,
            disable_trivial_auth=True,
        )
        fingerprint = self.host_fingerprint.strip()
        if fingerprint:
            if self.known_hosts is not None:
                raise ValueError("请在 SSH 主机指纹和 known_hosts 文件中选择一种主机身份校验方式")
            if not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint):
                raise ValueError("SSH 主机指纹应为 OpenSSH 的 SHA256 指纹")

            class PinnedHost(ssh.SSHClient):
                def validate_host_public_key(self, host, addr, port, key):
                    return hmac.compare_digest(key.get_fingerprint("sha256"), fingerprint)

            options.update(client_factory=PinnedHost, known_hosts=ssh.import_known_hosts(""))
        elif self.known_hosts is not None:
            try:
                options["known_hosts"] = os.fsdecode(os.path.expanduser(os.fspath(self.known_hosts)))
            except TypeError:
                raise ValueError("known_hosts 须为文件路径") from None
        return options


@asynccontextmanager
async def open_ssh(config):
    try:
        import asyncssh
    except ImportError:
        raise DependencyError("无法导入 asyncssh，请检查 hscredit 的基础依赖安装") from None
    try:
        async with asyncssh.connect(**config.ssh_options(asyncssh)) as ssh:
            yield ssh
    except asyncssh.HostKeyNotVerifiable:
        raise ValueError("SSH 主机身份校验失败，请核对主机指纹或运行账号的 known_hosts") from None
    except asyncssh.PermissionDenied:
        raise ValueError("SSH 登录失败，请核对认证方式和跳板机凭据") from None
    except (asyncssh.Error, asyncio.TimeoutError, OSError):
        raise ValueError("SSH 隧道连接失败，请检查跳板机地址、端口及转发权限") from None


class _DatabaseTLSContext(ssl.SSLContext):
    """通过转发端口连接时继续使用原数据库主机名和驱动的 TLS 配置。"""

    def __new__(cls, context, host):
        return super().__new__(cls, context.protocol)

    def __init__(self, context, host):
        self.context = context
        self.host = host

    def wrap_socket(self, sock, **kwargs):
        kwargs["server_hostname"] = self.host
        return self.context.wrap_socket(sock, **kwargs)


def _mysql_tls(options, host):
    tls_names = ("ssl", "ssl_ca", "ssl_cert", "ssl_key", "ssl_key_password", "ssl_verify_cert", "ssl_verify_identity")
    if options.get("ssl_disabled") or not any(options.get(name) for name in tls_names):
        return
    tls = {name: options[name] for name in tls_names if name in options}
    if isinstance(tls.get("ssl"), ssl.SSLContext) and not any(tls.get(name) for name in tls_names[1:]):
        context = tls["ssl"]
    else:
        try:
            from pymysql.connections import Connection
        except ImportError:
            raise DependencyError("缺少 MySQL 协议驱动，请安装对应数据库扩展") from None
        # 让驱动解析 CA、客户端证书与验证选项，此处仅创建 TLS 上下文。
        try:
            context = Connection(defer_connect=True, **tls).ctx
        except (AttributeError, OSError, TypeError, ValueError):
            raise ValueError("无法加载数据库 TLS 配置，请核对 CA 和客户端证书参数") from None
    for name in tls:
        options.pop(name)
    options["ssl"] = _DatabaseTLSContext(context, host)


async def forward_database(ssh, db_type, options):
    if db_type == "maxcompute":
        # 工作台的 MaxCompute 导入通过 SOCKS 保留远端 DNS、下载地址和 TLS 主机名。
        listener = await ssh.forward_socks(LOCAL_HOST, 0)
        return listener, options, f"socks5h://{LOCAL_HOST}:{listener.get_port()}"

    options = dict(options)
    if db_type == "oracle":
        dsn = options.get("dsn")
        dsn = dsn.removeprefix("//") if isinstance(dsn, str) else ""
        target = urlsplit("//" + dsn)
        if not target.hostname or not target.path.strip("/") or not re.fullmatch(r"[\w.\-:\[\]]+/[\w.$\-]+", dsn) or options.get("params") is not None:
            raise ValueError("Oracle 经 SSH 连接时请使用 主机[:端口]/服务名 格式的 dsn")
        host, port = target.hostname, target.port or 1521
    else:
        host = options.get("host", "localhost")
        clickhouse_https = db_type == "clickhouse" and (options.get("secure") or options.get("interface") == "https")
        default_port = 8443 if clickhouse_https else _TCP_PORTS[db_type]
        port = options.get("port") or default_port
        if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("请填写跳板机可访问的数据库主机和有效 TCP 端口")
        host = host.strip()

    if db_type in {"mysql", "starrocks"}:
        if any(options.get(name) for name in ("unix_socket", "read_default_file", "read_default_group")):
            raise ValueError("经 SSH 连接 MySQL 协议时请直接提供 TCP 连接参数")
        _mysql_tls(options, host)
    elif db_type == "clickhouse":
        if any(options.get(name) for name in ("dsn", "http_proxy", "https_proxy")):
            raise ValueError("ClickHouse 经 SSH 连接时请直接提供主机、端口和 TLS 参数")
        options.setdefault("server_host_name", host)
    elif db_type in {"hive", "impala"} and options.get("auth_mechanism") == "GSSAPI":
        options.setdefault("krb_host", host)

    listener = await ssh.forward_local_port(LOCAL_HOST, 0, host, port)
    if db_type == "oracle":
        options["dsn"] = f"{LOCAL_HOST}:{listener.get_port()}{target.path}"
    else:
        options.update(host=LOCAL_HOST, port=listener.get_port())
    return listener, options, None


class DatabaseTunnel:
    """在独立事件循环中维持转发，供同步驱动和 Notebook 共用。"""

    def __init__(self, db_type, connect_kwargs, config):
        self._loop = None
        self._closing = Event()
        self._ready = Future()
        self._thread = Thread(target=self._run, args=(db_type, connect_kwargs, config), name="hscredit-db-ssh", daemon=True)
        self._thread.start()
        try:
            self.connect_kwargs = self._ready.result()
        except BaseException:
            self.close()
            raise

    def _run(self, db_type, connect_kwargs, config):
        try:
            asyncio.run(self._serve(db_type, connect_kwargs, config))
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)

    async def _serve(self, db_type, connect_kwargs, config):
        self._stop = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        try:
            async with open_ssh(config) as ssh:
                listener, forwarded, _ = await forward_database(ssh, db_type, connect_kwargs)
                async with listener:
                    self._ready.set_result(forwarded)
                    if not self._closing.is_set():
                        await self._stop.wait()
        except ValueError as exc:
            raise DatabaseConnectionError(str(exc)) from None
        except DependencyError as exc:
            raise DatabaseConnectionError(str(exc)) from exc

    def close(self):
        self._closing.set()
        if self._loop is not None:
            with suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()
