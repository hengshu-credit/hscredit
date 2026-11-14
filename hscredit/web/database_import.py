"""通过 hscredit 数据库接口读取查询结果，生成平台数据集快照。"""

from contextlib import ExitStack, contextmanager
from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr
from starlette.concurrency import run_in_threadpool

from hscredit.database import Database, DatabaseConnectionError, DatabaseError, StreamState
from hscredit.exceptions import DependencyError
from .database_tunnel import SSHTunnel, database_route
from .storage import MAX_DATASET_COLUMNS, MAX_DATASET_ROWS


class DatabaseConnectionRequest(BaseModel):
    db_type: Literal["mysql", "starrocks", "clickhouse", "hive", "impala", "oracle", "maxcompute"]
    address: str = Field(min_length=1)
    port: Optional[int] = Field(default=None, ge=1, le=65535)
    database: str = ""
    username: str = ""
    password: SecretStr = SecretStr("")
    secure: bool = False
    auth_mechanism: Optional[Literal["PLAIN", "LDAP", "NOSASL", "GSSAPI"]] = None
    tunnel: Optional[SSHTunnel] = None

    class Config:
        extra = "forbid"

    def connection_options(self):
        address = self.address.strip()
        if not address:
            raise ValueError("请填写数据库连接地址")
        password = self.password.get_secret_value()
        if self.db_type == "oracle":
            return {"dsn": address, "user": self.username, "password": password}
        if self.db_type == "maxcompute":
            if not self.database.strip():
                raise ValueError("请填写 MaxCompute 项目名")
            return {"endpoint": address, "project": self.database.strip(), "access_id": self.username, "secret_access_key": password}
        options = {"host": address}
        if self.port is not None:
            options["port"] = self.port
        if self.database.strip():
            options["database"] = self.database.strip()
        if self.username:
            options["username" if self.db_type == "clickhouse" else "user"] = self.username
        if password:
            options["password"] = password
        if self.db_type in {"hive", "impala"}:
            if self.auth_mechanism:
                options["auth_mechanism"] = self.auth_mechanism
            options["use_ssl"] = self.secure
        elif self.db_type == "clickhouse":
            options["secure"] = self.secure
        elif self.secure:
            options.update(ssl_verify_cert=True, ssl_verify_identity=True)
        return options


class DatabaseImportRequest(DatabaseConnectionRequest):
    name: str = Field(min_length=1, max_length=150)
    sql: str = Field(min_length=1)


def _select_query(sql, db_type):
    try:
        from sqlglot import exp, parse
        from sqlglot.errors import SqlglotError
    except ImportError:
        raise ValueError("缺少 SQL 解析依赖，请在平台环境安装 hscredit[web]") from None

    dialect = {"impala": "hive", "maxcompute": "hive"}.get(db_type, db_type)
    sql = sql.strip().removesuffix(";").rstrip()
    try:
        statements = parse(sql, read=dialect)
    except SqlglotError:
        raise ValueError("无法解析查询，请检查数据库类型和 SQL 语法") from None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("请填写一条用于取数的 SELECT 查询，可包含 WITH 子句")
    query = statements[0]
    if query.find(exp.DML, exp.DDL, exp.Into, exp.Lock, exp.Command):
        raise ValueError("导入查询不能包含写入操作或锁定子句")
    if any(comment.upper().startswith(("!", "M!")) for node in query.walk() for comment in node.comments or ()):
        raise ValueError("导入查询不支持可执行注释")
    return sql


@contextmanager
def _database(db_type, options, proxy):
    try:
        with ExitStack() as stack:
            if proxy:
                try:
                    from odps.config import option_context
                except ImportError:
                    raise DependencyError("缺少 MaxCompute 驱动") from None
                stack.enter_context(option_context({"api_proxy": proxy, "data_proxy": proxy}))
            yield stack.enter_context(Database(db_type, **options))
    except DependencyError:
        raise ValueError(f'缺少数据库驱动，请在平台环境运行 pip install "hscredit[db-{db_type}]"') from None
    except DatabaseConnectionError as exc:
        if isinstance(exc.__cause__, DependencyError):
            raise ValueError(f'缺少数据库驱动，请在平台环境运行 pip install "hscredit[db-{db_type}]"') from None
        raise ValueError("数据库连接失败，请检查地址、账号和连接参数") from None
    except DatabaseError:
        raise ValueError("数据库读取失败，请检查连接、查询语句及账号权限") from None


async def check_database_connection(request):
    def probe(options, proxy):
        with _database(request.db_type, options, proxy) as database:
            sql = "SELECT 1 FROM DUAL" if request.db_type == "oracle" else "SELECT 1"
            rows = database.query(sql, result="rows")
            if not rows or rows[0][0] != 1:
                raise ValueError("数据库探测查询未返回预期结果")

    async with database_route(request) as (options, proxy):
        await run_in_threadpool(probe, options, proxy)
    return {"message": "已通过 SSH 隧道连接数据库" if request.tunnel else "数据库连接成功"}


def _read_snapshot(db_type, sql, options, proxy):
    with _database(db_type, options, proxy) as database, database.stream_query(sql, chunksize=10_000) as stream:
        for chunk in stream:
            if stream.rows_read > MAX_DATASET_ROWS or len(chunk.columns) > MAX_DATASET_COLUMNS:
                raise ValueError("查询结果超过 200 万行或 2000 个字段，请缩小查询范围")
        if stream.state != StreamState.COMPLETED:
            raise ValueError("查询读取中断，请重新导入")
        return stream.to_dataframe()


async def import_database_dataset(workspace, request):
    if not request.name.strip():
        raise ValueError("请填写数据集名称")
    sql = _select_query(request.sql, request.db_type)
    async with database_route(request) as (options, proxy):
        frame = await run_in_threadpool(_read_snapshot, request.db_type, sql, options, proxy)
    if frame.empty:
        raise ValueError("查询结果为空，请调整查询条件")
    try:
        return await run_in_threadpool(workspace.add_dataset, frame, request.name)
    except (TypeError, OverflowError):
        raise ValueError("查询结果包含无法保存的字段类型，请在 SQL 中转换为数值、日期或文本后导入") from None
