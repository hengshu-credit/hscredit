"""持久化推理调用，在管理端按服务检索并按保留期限清理。"""

import asyncio
import base64
import json
import logging
import re
import sqlite3
import time
from contextlib import closing, contextmanager
from urllib.parse import unquote_plus
from uuid import uuid4

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders

logger = logging.getLogger(__name__)
DAY = 86400
_PREDICTION = re.compile(r"/([a-f0-9]{32})/predict$")
_CREDENTIAL = re.compile(r"authorization|cookie|token|api[-_]?key|password|secret|credential|signature", re.I)


def _safe_headers(headers):
    # 使用键值对列表保留重复头，凭据只留名称。
    pairs = [(key.decode("latin-1"), value.decode("latin-1")) for key, value in headers]
    return [(key, "〔已隐藏〕" if _CREDENTIAL.search(key) else value) for key, value in pairs]


def _safe_query(raw):
    parts = raw.decode("latin-1").split("&")
    for index, item in enumerate(parts):
        key, separator, _ = item.partition("=")
        if separator and _CREDENTIAL.search(unquote_plus(key)):
            parts[index] = key + "=〔已隐藏〕"
    return "&".join(parts)


def _body_view(body):
    try:
        return {"encoding": "utf-8", "text": body.decode("utf-8")}
    except UnicodeDecodeError:
        return {"encoding": "base64", "text": base64.b64encode(body).decode("ascii")}


class ServiceCalls:
    """保存请求和响应原文，列表查询只读取摘要字段。"""

    def __init__(self, workspace):
        self.path = workspace.root / "services" / "calls.sqlite3"

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            with db:
                yield db

    def initialize(self):
        with self.connection() as db:
            db.execute("PRAGMA auto_vacuum=FULL")
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS retention (
                    service TEXT PRIMARY KEY, days INTEGER NOT NULL CHECK(days > 0)
                );
                CREATE TABLE IF NOT EXISTS calls (
                    id TEXT PRIMARY KEY, service TEXT NOT NULL, started_at REAL NOT NULL,
                    duration_ms REAL NOT NULL, status INTEGER NOT NULL, target TEXT NOT NULL,
                    request_headers TEXT NOT NULL, response_headers TEXT NOT NULL,
                    request_body BLOB NOT NULL, response_body BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS calls_by_service ON calls(service, started_at DESC, id DESC);
            """)
        self.purge()

    @staticmethod
    def _days(db, service):
        policy = db.execute("SELECT days FROM retention WHERE service=?", (service,)).fetchone()
        return policy["days"] if policy else None

    def set_retention(self, service, days):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = self._days(db, service)
            limits = [value for value in (previous, days) if value is not None]
            if limits:
                # 延长期限时也先清理旧策略下已过期的记录。
                db.execute("DELETE FROM calls WHERE service=? AND started_at<=?", (service, time.time() - min(limits) * DAY))
            if days is None:
                db.execute("DELETE FROM retention WHERE service=?", (service,))
            else:
                db.execute("INSERT OR REPLACE INTO retention VALUES (?, ?)", (service, days))

    def purge(self):
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            policies = db.execute("SELECT service, days FROM retention").fetchall()
            db.executemany("DELETE FROM calls WHERE service=? AND started_at<=?", [(row["service"], now - row["days"] * DAY) for row in policies])

    async def maintain(self):
        while True:
            await asyncio.sleep(3600)
            try:
                await run_in_threadpool(self.purge)
            except sqlite3.Error:
                logger.exception("清理过期推理调用记录失败")

    def append(self, identifier, service, started, duration, status, target, request_headers, response_headers, request_body, response_body):
        with self.connection() as db:
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    service,
                    started,
                    duration,
                    status,
                    target,
                    json.dumps(request_headers, ensure_ascii=False),
                    json.dumps(response_headers, ensure_ascii=False),
                    request_body,
                    response_body,
                ),
            )

    def page(self, service, page, page_size, failed_only, call_id):
        with self.connection() as db:
            db.execute("BEGIN")
            days = self._days(db, service)
            where = "service=?"
            args = [service]
            if days is not None:
                where += " AND started_at>?"
                args.append(time.time() - days * DAY)
            if failed_only:
                where += " AND status>=400"
            if call_id:
                where += " AND id=?"
                args.append(call_id)
            total = db.execute("SELECT COUNT(*) FROM calls WHERE " + where, args).fetchone()[0]
            rows = db.execute("SELECT id, started_at, duration_ms, status FROM calls WHERE " + where + " ORDER BY started_at DESC, id DESC LIMIT ? OFFSET ?", [*args, page_size, (page - 1) * page_size])
            return {"total": total, "items": [dict(r) for r in rows], "retention_days": days}

    def detail(self, service, call_id):
        with self.connection() as db:
            row = db.execute(
                """
                SELECT calls.* FROM calls LEFT JOIN retention USING(service)
                WHERE service=? AND id=? AND (days IS NULL OR started_at>?-days*86400)
            """,
                (service, call_id, time.time()),
            ).fetchone()
        if row is None:
            raise FileNotFoundError("调用记录不存在或已过期")
        return {
            "id": call_id,
            "method": "POST",
            "target": row["target"],
            "started_at": row["started_at"],
            "status": row["status"],
            "duration_ms": row["duration_ms"],
            "request_headers": json.loads(row["request_headers"]),
            "response_headers": json.loads(row["response_headers"]),
            "request_body": _body_view(row["request_body"]),
            "response_body": _body_view(row["response_body"]),
        }


class RecordPrediction:
    """记录已发布预测接口的 HTTP 消息，包括参数校验及鉴权失败的响应。"""

    def __init__(self, app, workspace, calls):
        self.app, self.workspace, self.calls = app, workspace, calls

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        root = scope.get("root_path", "").rstrip("/")
        path = scope["path"].removeprefix(root)
        match = _PREDICTION.fullmatch(path)
        if not match or not (self.workspace.path("services", match[1]) / "metadata.json").is_file():
            return await self.app(scope, receive, send)
        identifier = uuid4().hex
        started, clock = time.time(), time.perf_counter()
        request_body, response_body = bytearray(), bytearray()
        status, response_headers = None, []
        target = root + path
        if scope["query_string"]:
            target += "?" + _safe_query(scope["query_string"])

        async def capture_request():
            message = await receive()
            if message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
            return message

        async def capture_response(message):
            nonlocal status, response_headers
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["X-Request-ID"] = identifier
                response_headers = _safe_headers(message["headers"])
                status = message["status"]
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    try:
                        await run_in_threadpool(self.calls.append, identifier, match[1], started, (time.perf_counter() - clock) * 1000, status, target, _safe_headers(scope["headers"]), response_headers, request_body, response_body)
                    except sqlite3.Error:
                        logger.exception("保存推理调用 %s 失败", identifier)
            await send(message)

        await self.app(scope, capture_request, capture_response)
