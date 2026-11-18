"""在单次数据库操作期间建立 SSH 转发并校验跳板机身份。"""

import ssl
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from hscredit.database.tunnel import SSHTunnelConfig, forward_database, open_ssh
from hscredit.exceptions import DependencyError


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
        return SSHTunnelConfig(
            host=self.host,
            port=self.port,
            username=self.username,
            auth=self.auth,
            password=self.password.get_secret_value(),
            private_key=self.private_key.get_secret_value(),
            passphrase=self.passphrase.get_secret_value(),
            host_fingerprint=self.host_fingerprint,
        ).ssh_options(ssh)


@asynccontextmanager
async def database_route(request):
    options = request.connection_options()
    if request.tunnel is None:
        yield options, None
        return
    if request.db_type in {"mysql", "starrocks"} and options.pop("ssl_verify_identity", False):
        options.pop("ssl_verify_cert", None)
        options["ssl"] = ssl.create_default_context()
    try:
        async with open_ssh(request.tunnel) as ssh:
            listener, options, proxy = await forward_database(ssh, request.db_type, options)
            async with listener:
                yield options, proxy
    except DependencyError as exc:
        raise ValueError(str(exc)) from None
