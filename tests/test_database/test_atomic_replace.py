"""MySQL真实方言/专用会话的假驱动故障注入；不连接外部数据库。"""

from copy import deepcopy
from dataclasses import dataclass, field
import re
from types import SimpleNamespace

import pandas as pd
import pytest

from hscredit.database import Database, DatabaseCapabilityError, DatabaseWriteError, register_adapter
from hscredit.database.adapters.base import BaseDatabaseAdapter
from hscredit.database.adapters.mysql import MySQLAdapter
from hscredit.database.adapters.starrocks import StarRocksAdapter
from hscredit.exceptions import InputValidationError, ValidationError


@dataclass
class State:
    tables: dict = field(default_factory=lambda: {"risk.target": [(99, "old")]})
    calls: list = field(default_factory=list)
    version: str = "8.0.36"
    vendor: str = "MySQL Community Server - GPL"
    grants: list = field(default_factory=lambda: ["GRANT SELECT, TRIGGER ON *.* TO 'maint'@'localhost'"])
    engine: str = "InnoDB"
    stage_engine: str = "InnoDB"
    table_type: str = "BASE TABLE"
    foreign_key: bool = False
    trigger: bool = False
    fail_create: bool = False
    create_collision: bool = False
    fail_batch: int = 0
    fail_rename: bool = False
    lost_rename_response: bool = False
    fail_cleanup: bool = False
    fail_close: bool = False
    schema_drift: bool = False
    fake_count: bool = False
    lost_commit_response: bool = False
    sql_mode: str = "ALLOW_INVALID_DATES"
    ignore_safety_settings: bool = False
    warning_batch: int = 0
    ddl_options: str = ""
    commits: int = 0
    batch_calls: int = 0
    rename_calls: int = 0
    connects: int = 0
    locked: bool = False


def names(sql):
    identifiers = [part.replace("``", "`") for part in re.findall(r"`((?:``|[^`])*)`", sql)]
    return [".".join(identifiers[i:i + 2]) for i in range(0, len(identifiers), 2)]


class Cursor:
    def __init__(self, connection):
        self.connection, self.state = connection, connection.state
        self.rows = []

    def execute(self, sql, params=None):
        state = self.state
        state.calls.append(("execute", sql, params))
        self.rows = []
        if sql.startswith("SELECT VERSION"):
            self.rows = [(state.version, state.vendor)]
        elif sql == "SELECT @@SESSION.sql_mode":
            self.rows = [(self.connection.sql_mode,)]
        elif sql == "SELECT @@SESSION.sql_mode, @@SESSION.lock_wait_timeout":
            self.rows = [(self.connection.sql_mode, self.connection.lock_wait_timeout)]
        elif sql == "SET SESSION sql_mode=%s":
            if not state.ignore_safety_settings:
                self.connection.sql_mode = params[0]
        elif sql == "SET SESSION lock_wait_timeout=30":
            if not state.ignore_safety_settings:
                self.connection.lock_wait_timeout = 30
        elif sql.startswith("SHOW GRANTS"):
            self.rows = [(value,) for value in state.grants]
        elif sql.startswith("SELECT ENGINE"):
            engine = state.stage_engine if params[1].startswith("__hsc_stage_") else state.engine
            self.rows = [(engine, state.table_type)] if ".".join(params) in state.tables else []
        elif "information_schema.key_column_usage" in sql:
            self.rows = [("fk",)] if state.foreign_key else []
        elif "information_schema.triggers" in sql:
            self.rows = [("trigger",)] if state.trigger else []
        elif sql.startswith("SHOW CREATE TABLE"):
            self.rows = [("target", "CREATE target (id BIGINT, name TEXT) ENGINE=InnoDB" + state.ddl_options + (" CHANGED" if state.schema_drift and state.commits else ""))]
        elif sql.startswith("CREATE TABLE"):
            stage, target = names(sql)
            if state.create_collision:
                state.tables[stage] = [(777, "someone else")]
                raise RuntimeError("staging exists")
            if state.fail_create:
                raise RuntimeError("create failed")
            assert stage.split('.')[0] == target.split('.')[0]
            state.tables[stage] = []
        elif sql.startswith("LOCK TABLES"):
            state.locked = True
        elif sql == "UNLOCK TABLES":
            state.locked = False
        elif sql.startswith("INSERT INTO"):
            state.batch_calls += 1
            if state.batch_calls == state.fail_batch:
                raise RuntimeError("batch write failed")
            prefix = sql.rsplit(" VALUES ", 1)[0]
            column_count = len(re.findall(r"`((?:``|[^`])*)`", prefix)) - 2
            values = [tuple(params[i:i + column_count]) for i in range(0, len(params), column_count)]
            self.connection.pending = (names(sql)[0], values)
            self.connection.warning_count = int(state.batch_calls == state.warning_batch)
        elif sql == "SHOW COUNT(*) WARNINGS":
            self.rows = [(self.connection.warning_count,)]
        elif sql.startswith("SELECT COUNT(*)"):
            self.rows = [(len(state.tables[names(sql)[0]]) + int(state.fake_count),)]
        elif sql.startswith("RENAME TABLE"):
            state.rename_calls += 1
            assert state.locked
            if state.fail_rename:
                raise RuntimeError("rename rejected")
            old, backup, stage, target = names(sql)
            assert target == old and backup not in state.tables
            updated = dict(state.tables)
            updated[backup] = updated.pop(old)
            updated[target] = updated.pop(stage)
            state.tables = updated
            if state.lost_rename_response:
                raise ConnectionError("rename response lost")
        elif sql.startswith("DROP TABLE"):
            if state.fail_cleanup:
                raise RuntimeError("cleanup failed")
            target = names(sql)[0]
            assert target != "risk.target"
            state.tables.pop(target)
        else:
            raise AssertionError(sql)
        return 0

    def fetchall(self):
        return self.rows

    def executemany(self, sql, values):
        raise AssertionError("原子替换不能使用会自动拆语句并丢失早期警告的executemany")

    def close(self):
        self.state.calls.append(("cursor_close",))
        if self.state.fail_close:
            raise RuntimeError("cursor close failed")


class Connection:
    def __init__(self, state):
        self.state, self.pending = state, None
        self.sql_mode = state.sql_mode
        self.lock_wait_timeout = 31536000
        self.warning_count = 0

    def autocommit(self, value):
        assert value is False

    def cursor(self):
        return Cursor(self)

    def commit(self):
        self.state.calls.append(("commit",))
        if self.pending:
            table, values = self.pending
            self.state.tables[table].extend(values)
            self.pending = None
        self.state.commits += 1
        if self.state.lost_commit_response:
            raise ConnectionError("commit response lost")

    def rollback(self):
        self.pending = None

    def close(self):
        self.state.calls.append(("connection_close",))
        self.state.locked = False
        if self.state.fail_close:
            raise RuntimeError("connection close failed")


class FakeMySQL(MySQLAdapter):
    def __init__(self, *, connect_kwargs, pool_options, adapter_options):
        BaseDatabaseAdapter.__init__(self, connect_kwargs=connect_kwargs, pool_options=pool_options, adapter_options=adapter_options)
        state = connect_kwargs["state"]
        def connect(**kwargs):
            state.connects += 1
            return Connection(state)
        self.driver = SimpleNamespace(connect=connect)


def database(state=None):
    register_adapter("atomic_mysql_fake", FakeMySQL, replace=True)
    state = state or State()
    return Database("atomic_mysql_fake", state=state, database="risk"), state


def data():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["new_a", "new_b", "new_c"]})


def test_atomic_replace_only_publishes_once_and_keeps_old_backup():
    db, state = database()
    old = deepcopy(state.tables["risk.target"])
    result = db.atomic_replace_table(data(), "target", batch_size=2)
    assert result.completed and result.target_published is True
    assert result.rows_staged == result.rows_published == result.rows_inserted == 3
    assert result.batches_staged == 2 and result.batches_committed == 1
    assert state.tables[result.backup_table] == old
    assert state.tables["risk.target"] == list(data().itertuples(index=False, name=None))
    assert result.staging_table not in state.tables
    assert state.rename_calls == 1 and state.connects == 1
    rename_position = next(i for i, call in enumerate(state.calls) if call[0] == "execute" and call[1].startswith("RENAME"))
    assert all(i < rename_position for i, call in enumerate(state.calls) if call[0] == "execute" and call[1].startswith("INSERT"))
    assert not any("TRUNCATE" in str(call) or "DROP TABLE `risk`.`target`" in str(call) for call in state.calls)


@pytest.mark.parametrize("failure", ["prepare", "batch", "iterator"])
def test_prepublication_failures_keep_old_target_and_report_staged_counts(failure):
    db, state = database(State(fail_create=failure == "prepare", fail_batch=2 if failure == "batch" else 0))
    old = deepcopy(state.tables)
    def chunks():
        yield data().iloc[:2]
        raise RuntimeError("input failed")
    with pytest.raises(DatabaseWriteError) as error:
        db.atomic_replace_table(chunks() if failure == "iterator" else data(), "risk.target", batch_size=2)
    result = error.value.result
    assert result.target_published is False and result.rows_published == 0
    assert result.rows_staged == (0 if failure == "prepare" else 2)
    assert result.batches_staged == (0 if failure == "prepare" else 1)
    assert state.tables == old and state.rename_calls == 0


@pytest.mark.parametrize("lost_response", [False, True])
def test_publication_unknown_does_not_retry_or_remove_either_possible_version(lost_response):
    db, state = database(State(fail_rename=not lost_response, lost_rename_response=lost_response))
    with pytest.raises(DatabaseWriteError) as error:
        db.atomic_replace_table(data(), "risk.target", batch_size=2)
    result = error.value.result
    assert result.target_published is None and result.rows_published is None
    assert result.phase == "publication_unknown" and result.rows_staged == 3
    assert state.rename_calls == 1 and state.connects == 1
    assert not any(call[0] == "execute" and call[1].startswith("DROP") for call in state.calls)
    if lost_response:
        assert state.tables[result.backup_table] == [(99, "old")]
        assert len(state.tables["risk.target"]) == 3
    else:
        assert state.tables["risk.target"] == [(99, "old")]
        assert result.staging_table in state.tables


def test_cleanup_failure_does_not_mask_primary_error():
    db, state = database(State(fail_batch=2, fail_cleanup=True, fail_close=True))
    with pytest.raises(DatabaseWriteError) as error:
        db.atomic_replace_table(data(), "risk.target", batch_size=2)
    assert "batch write failed" in str(error.value.driver_error)
    assert error.value.cleanup_error is not None
    assert len(error.value.result.details["cleanup_error_types"]) == 3
    assert state.tables["risk.target"] == [(99, "old")]


def test_cleanup_failure_after_publication_is_not_reported_as_unpublished():
    db, state = database(State(fail_cleanup=True))
    result = db.atomic_replace_table(data(), "risk.target", keep_backup=False)
    assert result.completed and result.target_published is True
    assert result.phase == "published_cleanup_warning"
    assert result.backup_table in state.tables


def test_unknown_staging_commit_never_counts_as_target_publication():
    db, state = database(State(lost_commit_response=True))
    with pytest.raises(DatabaseWriteError) as error:
        db.atomic_replace_table(data(), "risk.target", batch_size=2)
    result = error.value.result
    assert result.target_published is False and result.rows_published == 0
    assert result.rows_staged == result.batches_staged == 0
    assert result.details["staging_commit_unknown"] is True
    assert state.rename_calls == 0 and state.batch_calls == 1
    assert state.tables == {"risk.target": [(99, "old")]}


def test_successful_opt_in_backup_cleanup_only_removes_owned_backup():
    db, state = database()
    result = db.atomic_replace_table(data(), "risk.target", keep_backup=False)
    assert result.completed and result.target_published is True
    assert set(state.tables) == {"risk.target"}
    drop_statements = [call[1] for call in state.calls if call[0] == "execute" and call[1].startswith("DROP")]
    assert drop_statements == [f"DROP TABLE `risk`.`{result.backup_table.split('.')[1]}`"]


@pytest.mark.parametrize("options", [
    {"version": "5.7.44"}, {"version": "8.0.12"}, {"version": "8.0.36-MariaDB"},
    {"vendor": "TiDB"}, {"engine": "MyISAM"}, {"table_type": "VIEW"},
    {"foreign_key": True}, {"trigger": True}, {"grants": ["GRANT SELECT, TRIGGER ON risk.* TO 'user'@'%' "]},
    {"grants": ["GRANT ALL PRIVILEGES ON *.* TO 'user'@'%'", "REVOKE SELECT ON secret.* FROM 'user'@'%'"]},
])
def test_unsafe_capabilities_are_rejected_before_any_table_change(options):
    db, state = database(State(**options))
    with pytest.raises(DatabaseCapabilityError):
        db.atomic_replace_table(data(), "risk.target")
    assert state.tables == {"risk.target": [(99, "old")]}
    assert not any(call[0] == "execute" and call[1].startswith(("CREATE", "DROP", "RENAME")) for call in state.calls)


@pytest.mark.parametrize("flag", ["schema_drift", "fake_count", "stage_engine"])
def test_locked_revalidation_stops_stale_or_incomplete_staging(flag):
    db, state = database(State(**{flag: "MyISAM" if flag == "stage_engine" else True}))
    with pytest.raises(DatabaseCapabilityError) as error:
        db.atomic_replace_table(data(), "risk.target")
    assert error.value.result.target_published is False
    assert state.tables == {"risk.target": [(99, "old")]}
    assert state.rename_calls == 0 and not state.locked


def test_staging_collision_is_never_cleaned_up_as_owned():
    db, state = database(State(create_collision=True))
    with pytest.raises(DatabaseWriteError) as error:
        db.atomic_replace_table(data(), "risk.target")
    assert state.tables[error.value.result.staging_table] == [(777, "someone else")]
    assert not any(call[0] == "execute" and call[1].startswith("DROP") for call in state.calls)


def test_generated_names_are_same_schema_and_tampering_is_rejected():
    db, _ = database()
    session = db.adapter.atomic_replace_session("risk.target")
    assert re.fullmatch(r"risk\.__hsc_stage_[0-9a-f]{32}", session.staging)
    with pytest.raises(ValidationError):
        session._managed_name("other.__hsc_stage_" + session.token, "stage")
    with pytest.raises(ValidationError):
        db.atomic_replace_table(data(), "catalog.risk.target")
    with pytest.raises(InputValidationError):
        db.atomic_replace_table(data().iloc[:0], "risk.target")


def test_other_backends_do_not_consume_input_or_fall_back_to_destructive_write():
    assert not StarRocksAdapter.capabilities.atomic_replace
    db, state = database()
    db.adapter.database_type = "starrocks"
    def forbidden():
        raise AssertionError("不支持后端不应消费输入")
        yield data()
    with pytest.raises(DatabaseCapabilityError):
        db.atomic_replace_table(forbidden(), "risk.target")
    assert state.connects == 0


def test_session_strict_mode_and_finite_lock_timeout_apply_only_to_owned_connection():
    db, state = database()
    result = db.atomic_replace_table(data(), "risk.target", batch_size=2)
    assert result.completed
    assert state.sql_mode == "ALLOW_INVALID_DATES"  # 假服务器初始/全局配置未变
    setting = next(call for call in state.calls if call[0] == "execute" and call[1] == "SET SESSION sql_mode=%s")
    modes = set(setting[2][0].split(','))
    assert {'STRICT_TRANS_TABLES', 'STRICT_ALL_TABLES', 'NO_ZERO_DATE', 'NO_ZERO_IN_DATE'}.issubset(modes)
    assert 'ALLOW_INVALID_DATES' not in modes
    assert any(call[:2] == ("execute", "SET SESSION lock_wait_timeout=30") for call in state.calls)
    sql_calls = [call[1] for call in state.calls if call[0] == "execute"]
    assert sql_calls.index("SET SESSION lock_wait_timeout=30") < next(i for i, sql in enumerate(sql_calls) if sql.startswith("LOCK TABLES"))
    for index, call in enumerate(state.calls):
        if call[0] == "execute" and call[1].startswith("INSERT"):
            assert state.calls[index + 1][:2] == ("execute", "SHOW COUNT(*) WARNINGS")
            assert state.calls[index + 2][0] == "commit"
    statements = [call[1] for call in state.calls if call[0] == "execute" and call[1].startswith("INSERT")]
    assert len(statements) == 2
    assert statements[0].count("(%s, %s)") == 2


@pytest.mark.parametrize("batch", [1, 2])
def test_any_batch_warning_blocks_commit_and_publication(batch):
    db, state = database(State(warning_batch=batch))
    with pytest.raises(DatabaseWriteError) as error:
        db.atomic_replace_table(data(), "risk.target", batch_size=2)
    result = error.value.result
    assert result.target_published is False and result.rows_published == 0
    assert result.batches_staged == batch - 1
    assert state.commits == batch - 1
    assert state.rename_calls == 0 and state.tables == {"risk.target": [(99, "old")]}


@pytest.mark.parametrize("options", [
    {"ignore_safety_settings": True},
    {"ddl_options": " DATA DIRECTORY='/external'"},
    {"ddl_options": " INDEX DIRECTORY='/external'"},
    {"sql_mode": "NO_BACKSLASH_ESCAPES", "ddl_options": " COMMENT='path\\' DATA DIRECTORY='/external'"},
])
def test_unverified_safety_or_unpreserved_options_fail_before_create(options):
    db, state = database(State(**options))
    with pytest.raises(DatabaseCapabilityError):
        db.atomic_replace_table(data(), "risk.target")
    assert not any(call[0] == "execute" and call[1].startswith("CREATE") for call in state.calls)
    assert state.tables == {"risk.target": [(99, "old")]}


def test_directory_words_inside_comment_are_not_table_options():
    db, _ = database(State(ddl_options=" COMMENT='DATA DIRECTORY is not an option here'"))
    assert db.atomic_replace_table(data(), "risk.target").completed
