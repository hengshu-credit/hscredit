"""受限MySQL整表替换：专用无重试连接、暂存写入、一次多表RENAME。

仅接受Oracle MySQL 8.0.13+/8.4、InnoDB基础表和可证实的元数据可见性。
原子DDL并不等于把分批DML包在一个可回滚事务；发布前的提交仅发生在暂存表。
方言依据：MySQL官方 rename-table.html、atomic-ddl.html、lock-tables.html。
"""

import re
from uuid import uuid4

from ...exceptions import ValidationError
from ..exceptions import DatabaseCapabilityError, DatabaseQueryError, database_error_from
from ..writing import BatchWriteResult, split_qualified_name


class MySQLAtomicReplaceSession:
    """单次操作拥有的连接、随机暂存名及发布收据；不重试任何SQL。"""

    def __init__(self, adapter, table_name):
        if len(split_qualified_name(table_name)) > 2:
            raise ValidationError("MySQL原子替换仅支持同schema的 schema.table 名称")
        self.adapter = adapter
        self.schema, self.table = adapter._schema_and_table(table_name)
        if any(not part or len(part) > 64 or '\x00' in part for part in (self.schema, self.table)):
            raise ValidationError("MySQL schema/表名必须为1到64个字符且不含空字符")
        if self.schema.lower() in {'mysql', 'information_schema', 'performance_schema', 'sys'}:
            raise DatabaseCapabilityError("不允许原子替换MySQL系统schema中的表")
        self.token = uuid4().hex
        self.target = f"{self.schema}.{self.table}"
        self.staging = f"{self.schema}.__hsc_stage_{self.token}"
        self.backup = f"{self.schema}.__hsc_backup_{self.token}"
        self.connection = None
        self.cursor = None
        self.staging_created = False
        self.staging_creation_unknown = False
        self.publish_attempted = False
        self.target_published = False
        self.staging_commit_unknown = False
        self.cleanup_errors = []
        self.locked = False
        self.keep_backup = True
        self.target_ddl = None
        self.session_sql_modes = set()

    def _managed_name(self, name, kind):
        parts = split_qualified_name(name)
        expected = f"__hsc_{kind}_{self.token}"
        if len(parts) != 2 or parts[0] != self.schema or parts[1] != expected or not re.fullmatch(r"[0-9a-f]{32}", self.token):
            raise ValidationError("拒绝操作不属于本次同schema暂存/备份命名空间的表")
        return self.adapter.quote_qualified_name(name)

    def _execute(self, sql, params=None):
        try:
            return self.adapter.execute_cursor(self.cursor, sql, params)
        except Exception as exc:
            raise database_error_from(DatabaseQueryError, "MySQL原子替换SQL执行失败", cause=exc, sql=sql, params=params)

    def _rows(self, sql, params=None):
        self._execute(sql, params)
        return list(self.cursor.fetchall())

    def __enter__(self):
        try:
            # 不经DBUtils/SteadyDB，避免连接恢复时自动重放已发送的写入/DDL。
            self.connection = self.adapter.driver.connect(**self.adapter.connect_kwargs)
            self.connection.autocommit(False)
            self.cursor = self.connection.cursor()
            return self
        except BaseException:
            self._close_resources()
            raise

    def _check_server_and_visibility(self):
        rows = self._rows("SELECT VERSION(), @@version_comment")
        version, comment = rows[0] if rows else ('', '')
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
        numbers = tuple(int(value) for value in match.groups()) if match else ()
        allowed = numbers and (numbers[:2] == (8, 4) or (numbers[:2] == (8, 0) and numbers[2] >= 13))
        vendor = (str(version) + ' ' + str(comment)).lower()
        if not allowed or 'mysql' not in vendor or any(name in vendor for name in ('mariadb', 'tidb', 'starrocks', 'oceanbase', 'percona')):
            raise DatabaseCapabilityError("当前只验证Oracle MySQL 8.0.13+/8.4的原子替换，不接受其它版本或协议兼容后端")
        grants = [str(row[0]).upper() for row in self._rows("SHOW GRANTS FOR CURRENT_USER()")]
        privileges = set()
        for statement in grants:
            match = re.match(r"^GRANT (.+?) ON \*\.\* TO ", statement)
            if match:
                privileges.update(part.strip() for part in match.group(1).split(','))
        # 不展开角色/部分授权；没有完整可见性时空结果不能证明没有入向FK。
        if 'ALL PRIVILEGES' not in privileges and not {'SELECT', 'TRIGGER'}.issubset(privileges):
            raise DatabaseCapabilityError("无法证明完整外键/触发器元数据可见性；此受控维护入口要求可验证的全局SELECT和TRIGGER可见性，不会自动改权限")
        if any(statement.startswith('REVOKE ') for statement in grants):
            raise DatabaseCapabilityError("检测到部分撤销授权，无法证明完整元数据可见性")

    def _check_target(self, table=None):
        table = self.table if table is None else table
        rows = self._rows("SELECT ENGINE, TABLE_TYPE FROM information_schema.tables WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s", (self.schema, table))
        if len(rows) != 1 or str(rows[0][0]).lower() != 'innodb' or str(rows[0][1]).upper() != 'BASE TABLE':
            raise DatabaseCapabilityError("原子替换要求目标已存在且为InnoDB基础表，不支持视图/其它引擎")
        foreign = self._rows(
            "SELECT CONSTRAINT_NAME FROM information_schema.key_column_usage WHERE "
            "(TABLE_SCHEMA=%s AND TABLE_NAME=%s AND REFERENCED_TABLE_NAME IS NOT NULL) OR "
            "(REFERENCED_TABLE_SCHEMA=%s AND REFERENCED_TABLE_NAME=%s) LIMIT 1",
            (self.schema, table, self.schema, table),
        )
        triggers = self._rows("SELECT TRIGGER_NAME FROM information_schema.triggers WHERE EVENT_OBJECT_SCHEMA=%s AND EVENT_OBJECT_TABLE=%s LIMIT 1", (self.schema, table))
        if foreign or triggers:
            raise DatabaseCapabilityError("目标存在入向/出向外键或触发器，不能用暂存表替换其依赖语义")
        ddl = self._rows(f"SHOW CREATE TABLE {self.adapter.quote_qualified_name(self.schema + '.' + table)}")
        if len(ddl) != 1 or len(ddl[0]) < 2:
            raise DatabaseCapabilityError("无法读取完整目标表结构，拒绝替换")
        definition = str(ddl[0][1])
        # 去掉引用的标识符/字符串后检查选项；避免将列名或注释正文当成目录选项。
        no_backslash = 'NO_BACKSLASH_ESCAPES' in self.session_sql_modes
        single_quoted = r"'(?:''|[^'])*'" if no_backslash else r"'(?:''|\\.|[^'\\])*'"
        double_quoted = r'"(?:""|[^"])*"' if no_backslash or 'ANSI_QUOTES' in self.session_sql_modes else r'"(?:""|\\.|[^"\\])*"'
        structural = re.sub(r"`(?:``|[^`])*`|" + single_quoted + "|" + double_quoted, " ", definition)
        if re.search(r"\b(?:DATA|INDEX)\s+DIRECTORY\b", structural, flags=re.IGNORECASE):
            raise DatabaseCapabilityError("CREATE TABLE LIKE不保留DATA DIRECTORY/INDEX DIRECTORY，当前入口拒绝此类表")
        return definition

    def _configure_session_safety(self):
        """只调整本操作专用连接，不修改全局配置或业务连接。"""
        rows = self._rows("SELECT @@SESSION.sql_mode")
        if len(rows) != 1:
            raise DatabaseCapabilityError("无法读取会话sql_mode，拒绝暂存写入")
        modes = {mode.strip().upper() for mode in str(rows[0][0]).split(',') if mode.strip()}
        required = {'STRICT_ALL_TABLES', 'STRICT_TRANS_TABLES', 'NO_ZERO_DATE', 'NO_ZERO_IN_DATE',
                    'ERROR_FOR_DIVISION_BY_ZERO', 'NO_ENGINE_SUBSTITUTION'}
        modes.difference_update({'ALLOW_INVALID_DATES'})
        modes.update(required)
        self._execute("SET SESSION sql_mode=%s", (','.join(sorted(modes)),))
        self._execute("SET SESSION lock_wait_timeout=30")
        verified = self._rows("SELECT @@SESSION.sql_mode, @@SESSION.lock_wait_timeout")
        if len(verified) != 1:
            raise DatabaseCapabilityError("无法验证会话数据/锁等待策略")
        actual = {mode.strip().upper() for mode in str(verified[0][0]).split(',')}
        if not required.issubset(actual) or 'ALLOW_INVALID_DATES' in actual or int(verified[0][1]) != 30:
            raise DatabaseCapabilityError("严格sql_mode或30秒元数据锁等待策略未生效，拒绝写入")
        self.session_sql_modes = actual

    def prepare(self):
        self._check_server_and_visibility()
        self._configure_session_safety()
        self.target_ddl = self._check_target()
        staged = self._managed_name(self.staging, 'stage')
        self._managed_name(self.backup, 'backup')
        self.staging_creation_unknown = True
        self._execute(f"CREATE TABLE {staged} LIKE {self.adapter.quote_qualified_name(self.target)}")
        # 只有明确成功的CREATE才拥有清理权，名字碰撞不能删除他人的表。
        self.staging_created = True
        self.staging_creation_unknown = False

    def write_batch(self, batch):
        self._managed_name(self.staging, 'stage')
        sql = self.adapter.build_insert_sql(self.staging, [str(column) for column in batch.columns], mode='o')
        values = [tuple(self.adapter._dbapi_value(value) for value in row) for row in batch.itertuples(index=False, name=None)]
        prefix, placeholders = sql.rsplit(' VALUES ', 1)
        sql = prefix + ' VALUES ' + ', '.join([placeholders] * len(values))
        parameters = tuple(value for row in values for value in row)
        try:
            # PyMySQL.executemany可能拆成多条INSERT，随后只能看见最后一条的警告。
            # 每个已界定批次发送一条多VALUES语句，不自动拆分或重试；过大时明确失败。
            self._execute(sql, parameters)
            warning_rows = self._rows("SHOW COUNT(*) WARNINGS")
            if len(warning_rows) != 1 or int(warning_rows[0][0]) != 0:
                raise DatabaseCapabilityError("暂存批量INSERT产生警告或无法确认警告数量，未提交该批并拒绝发布；请检查数据类型/截断/日期")
            self.staging_commit_unknown = True
            self.connection.commit()
            self.staging_commit_unknown = False
        except Exception as exc:
            raise database_error_from(DatabaseQueryError, "写入暂存表失败，未向目标发布", cause=exc, sql=sql, params=parameters)
        return BatchWriteResult(inserted=len(batch), updated=0, skipped=0)

    def publish(self, expected_rows):
        target = self.adapter.quote_qualified_name(self.target)
        staging = self._managed_name(self.staging, 'stage')
        backup = self._managed_name(self.backup, 'backup')
        self._execute(f"LOCK TABLES {target} WRITE, {staging} WRITE")
        self.locked = True
        # 写锁下再次验证，防止构建暂存期间结构或依赖发生变化。
        self._check_server_and_visibility()
        if self._check_target() != self.target_ddl:
            raise DatabaseCapabilityError("目标表结构在暂存写入期间发生变化，拒绝发布")
        self._check_target(split_qualified_name(self.staging)[1])
        rows = self._rows(f"SELECT COUNT(*) FROM {staging}")
        if not rows or int(rows[0][0]) != expected_rows:
            raise DatabaseCapabilityError("暂存表行数与已接收输入不一致，拒绝发布")
        self.publish_attempted = True
        self.target_published = None  # 发出请求至收到成功响应之间，结果不可推断。
        self._execute(f"RENAME TABLE {target} TO {backup}, {staging} TO {target}")
        self.target_published = True

    def _cleanup(self):
        if self.connection is not None and self.target_published is not True:
            try:
                self.connection.rollback()
            except Exception as exc:
                self.cleanup_errors.append(exc)
        if self.locked:
            try:
                self._execute("UNLOCK TABLES")
                self.locked = False
            except Exception as exc:
                self.cleanup_errors.append(exc)
        cleanup = None
        if self.target_published is False and self.staging_created:
            cleanup = self._managed_name(self.staging, 'stage')
        elif self.target_published is True and not self.keep_backup and not self.locked:
            cleanup = self._managed_name(self.backup, 'backup')
        # 发布结果未知时，不删除暂存或备份，也不尝试再次rename。
        if cleanup is not None:
            try:
                self._execute(f"DROP TABLE {cleanup}")
            except Exception as exc:
                self.cleanup_errors.append(exc)

    def _close_resources(self):
        for resource in (self.cursor, self.connection):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    self.cleanup_errors.append(exc)

    def __exit__(self, exc_type, exc, traceback):
        try:
            try:
                self._cleanup()
            except Exception as cleanup_error:
                self.cleanup_errors.append(cleanup_error)
        finally:
            self._close_resources()
        return False
