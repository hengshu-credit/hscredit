"""在单次数据库操作期间建立 SSH 转发并校验跳板机身份。"""

import asyncio
import hmac
import re
import ssl
from contextlib import asynccontextmanager
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr


class SSHTunnel(BaseModel):
    host: str = Field(min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    username: str = Field(min_length=1)
    auth: Literal["password", "key", "key_password"] = "password"
    password: SecretStr = SecretStr("")
    private_key: SecretStr = SecretStr("")
    passphrase: SecretStr = SecretStr("")
    host_fingerprint: str = ""

    class Config:
        extra = "forbid"

    def ssh_options(self, ssh):
        password = self.password.get_secret_value() if self.auth != "key" else None
        if not self.host.strip() or not self.username.strip() or (self.auth != "key" and not password):
            raise ValueError("请补全 SSH 主机、用户名和所选方式的登录凭据")
        key = None
        if self.auth != "password":
            try:
                key = ssh.import_private_key(self.private_key.get_secret_value().strip(), self.passphrase.get_secret_value() or None)
            except (ssh.KeyImportError, ssh.KeyEncryptionError, ValueError):
                raise ValueError("无法读取 SSH 私钥，请检查私钥文件及其解密口令") from None
        methods = {"password": ["password"], "key": ["publickey"], "key_password": ["publickey", "password"]}
        options = dict(
            host=self.host.strip(),
            port=self.port,
            username=self.username.strip(),
            password=password,
            client_keys=[key] if key else None,
            preferred_auth=methods[self.auth],
            agent_path=None,
            config=None,
            connect_timeout=15,
            disable_trivial_auth=True,
        )
        fingerprint = self.host_fingerprint.strip()
        if fingerprint:
            if not re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", fingerprint):
                raise ValueError("SSH 主机指纹应为 OpenSSH 的 SHA256 指纹")

            class PinnedHost(ssh.SSHClient):
                def validate_host_public_key(self, host, addr, port, key):
                    return hmac.compare_digest(key.get_fingerprint("sha256"), fingerprint)

            options.update(client_factory=PinnedHost, known_hosts=ssh.import_known_hosts(""))
        return options


class _DatabaseTLSContext(ssl.SSLContext):
    """PyMySQL 连接本地转发端口时，按真实数据库主机名校验证书。"""

    def wrap_socket(self, sock, **kwargs):
        kwargs["server_hostname"] = self.database_host
        return super().wrap_socket(sock, **kwargs)


async def _forward_database(ssh, db_type, options):
    if db_type == "maxcompute":
        # API 和数据下载可能使用不同域名，保留原始 URL、TLS 名称和远端 DNS。
        listener = await ssh.forward_socks("127.0.0.1", 0)
        return listener, options, f"socks5h://127.0.0.1:{listener.get_port()}"

    if db_type == "oracle":
        dsn = options["dsn"].removeprefix("//")
        target = urlsplit("//" + dsn)
        if not target.hostname or not target.path.strip("/") or target.query or target.fragment or target.username or not re.fullmatch(r"[\w.\-:\[\]]+/[\w.$\-]+", dsn):
            raise ValueError("Oracle 经 SSH 连接时请填写 主机[:端口]/服务名")
        host, port = target.hostname, target.port or 1521
    else:
        host = options["host"]
        defaults = {"mysql": 3306, "starrocks": 9030, "clickhouse": 8443 if options.get("secure") else 8123, "hive": 10000, "impala": 21050}
        port = options.get("port", defaults[db_type])
    if db_type in {"mysql", "starrocks"} and options.pop("ssl_verify_identity", False):
        options.pop("ssl_verify_cert", None)
        context = _DatabaseTLSContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_default_certs()
        context.database_host = host
        options["ssl"] = context
    elif db_type == "clickhouse":
        options["server_host_name"] = host
    elif options.get("auth_mechanism") == "GSSAPI":
        options["krb_host"] = host

    listener = await ssh.forward_local_port("127.0.0.1", 0, host, port)
    if db_type == "oracle":
        options["dsn"] = f"127.0.0.1:{listener.get_port()}{target.path}"
    else:
        options.update(host="127.0.0.1", port=listener.get_port())
    return listener, options, None


@asynccontextmanager
async def database_route(request):
    options = request.connection_options()
    if request.tunnel is None:
        yield options, None
        return
    try:
        import asyncssh
    except ImportError:
        raise ValueError("缺少 SSH 依赖，请在平台环境安装 hscredit[web]") from None
    try:
        async with asyncssh.connect(**request.tunnel.ssh_options(asyncssh)) as ssh:
            listener, options, proxy = await _forward_database(ssh, request.db_type, options)
            async with listener:
                yield options, proxy
    except asyncssh.HostKeyNotVerifiable:
        raise ValueError("SSH 主机身份校验失败，请核对主机指纹或平台用户的 known_hosts") from None
    except asyncssh.PermissionDenied:
        raise ValueError("SSH 登录失败，请核对认证方式和跳板机凭据") from None
    except (asyncssh.Error, asyncio.TimeoutError, OSError):
        raise ValueError("SSH 隧道连接失败，请检查跳板机地址、端口及转发权限") from None
