"""数据库测试使用的可观察假实现。"""

from dataclasses import dataclass, field

from hscredit.database import DatabaseCapabilities


class FakeAdapter:
    """不访问外部服务的适配器，用于验证公共门面。"""

    capabilities = DatabaseCapabilities(write_modes={"a", "o", "d"})
    instances = []

    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        self.connect_kwargs = connect_kwargs
        self.pool_options = pool_options
        self.adapter_options = adapter_options
        self.closed = False
        self.calls = []
        type(self).instances.append(self)

    def query(self, sql, params=None, result="dataframe"):
        self.calls.append(("query", sql, params, result))
        return {"sql": sql, "params": params, "result": result}

    def close(self):
        self.calls.append(("close",))
        self.closed = True


@dataclass
class FakeDBAPIState:
    """记录 DB-API 资源的真实状态变化。"""

    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    fail_execute: Exception = None
    fail_executemany: Exception = None
    connections: list = field(default_factory=list)
    cursors: list = field(default_factory=list)
    pool_kwargs: dict = field(default_factory=dict)
    pool_closed: bool = False


class FakeDBAPIDriver:
    """携带共享状态的 DB-API 模块替身。"""

    threadsafety = 1
    paramstyle = "format"

    def __init__(self, state):
        self.state = state


class FakeCursor:
    def __init__(self, state):
        self.state = state
        self.description = [
            (name, None, None, None, None, None, None) for name in state.columns
        ]
        self.executed = None
        self.executemany_call = None
        self.closed = False
        self.rowcount = -1
        self._position = 0
        state.cursors.append(self)

    def execute(self, sql, params=None):
        self.executed = (sql, params)
        if self.state.fail_execute is not None:
            raise self.state.fail_execute
        self.rowcount = len(self.state.rows)
        return self.rowcount

    def executemany(self, sql, values):
        materialized = list(values)
        self.executemany_call = (sql, materialized)
        if self.state.fail_executemany is not None:
            raise self.state.fail_executemany
        self.rowcount = len(materialized)
        return self.rowcount

    def fetchall(self):
        return list(self.state.rows)

    def fetchmany(self, size=None):
        size = 1 if size is None else size
        start = self._position
        self._position += size
        return list(self.state.rows[start : start + size])

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, state):
        self.state = state
        self.commit_calls = 0
        self.rollback_calls = 0
        self.close_calls = 0
        state.connections.append(self)

    def cursor(self, *args, **kwargs):
        del args, kwargs
        return FakeCursor(self.state)

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1

    def close(self):
        self.close_calls += 1


class FakePooledDB:
    def __init__(self, creator, **kwargs):
        self.creator = creator
        self.state = creator.state
        self.state.pool_kwargs = dict(kwargs)

    def connection(self, *args, **kwargs):
        del args, kwargs
        return FakeConnection(self.state)

    def close(self):
        self.state.pool_closed = True
