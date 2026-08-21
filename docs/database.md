# 数据库连接、流式读写与表结构导出

`hscredit.database` 为风控建模和数据分析流程提供统一的数据库入口，覆盖连接池、参数化 SQL、可中断流式读取、DataFrame 建表、分批写入和全库字段清单导出。数据库驱动均为可选依赖；普通 `import hscredit` 不会加载任何数据库驱动。

完整类与方法签名见 {doc}`api/database`。

## 安装

按实际数据库安装对应扩展：

| 数据库 | 安装命令 | 驱动与连接策略 |
|:---|:---|:---|
| MySQL / MariaDB | `pip install hscredit[db-mysql]` | PyMySQL + DBUtils |
| Hive | `pip install hscredit[db-hive]` | Impyla + DBUtils |
| Impala | `pip install hscredit[db-impala]` | Impyla + DBUtils |
| Oracle | `pip install hscredit[db-oracle]` | python-oracledb 原生连接池 |
| StarRocks | `pip install hscredit[db-starrocks]` | MySQL 协议 + 可选 Stream Load |
| ClickHouse | `pip install hscredit[db-clickhouse]` | clickhouse-connect 原生 DataFrame 流 |
| MaxCompute | `pip install hscredit[db-maxcompute]` | PyODPS DB-API + 原生表读写 |
| 全部数据库 | `pip install hscredit[database-all]` | 安装以上全部驱动 |

缺少驱动时只会在创建对应 `Database` 时抛出中文 `DependencyError`，不会阻止其他 hscredit 模块导入。

## 建立连接池

连接参数直接传给数据库驱动；连接池参数单独放在 `pool_options` 中，避免与驱动参数混淆：

```python
import os

from hscredit import Database

db = Database(
    "mysql",
    host="127.0.0.1",
    port=3306,
    user="risk_user",
    password=os.environ["RISK_DB_PASSWORD"],
    database="risk_db",
    pool_options={
        "mincached": 1,
        "maxcached": 5,
        "maxconnections": 10,
        "blocking": True,
    },
)
```

`Database` 支持上下文管理器。退出上下文或调用 `close()` 后，连接会归还并关闭池资源：

```python
with Database("mysql", **connect_params) as db:
    frame = db.query("SELECT 1 AS value")
```

密码、AccessKey、Token 和带凭据 DSN 不会进入 `repr`、用户错误信息或日志。

## 参数化查询与 SQL 执行

普通查询默认返回 DataFrame，也可以返回记录字典或原始行：

```python
frame = db.query(
    "SELECT user_id, score FROM model_score WHERE score >= %s",
    params=(600,),
)

records = db.query(
    "SELECT user_id, score FROM model_score WHERE score >= %s",
    params=(600,),
    result="records",
)

rows = db.query("SELECT 1", result="rows")
```

执行 DDL/DML 时使用 `execute()` 或 `executemany()`：

```python
db.execute("DELETE FROM model_score WHERE batch_id = %s", params=(batch_id,))
db.executemany(
    "INSERT INTO audit_log(event_id, event_name) VALUES (%s, %s)",
    [(1, "开始"), (2, "完成")],
)
```

数据值始终通过驱动参数绑定传递，不应使用字符串格式化拼接用户数据。表名、字段名等数据库对象由适配器分段引用。

## 流式读取、进度条与主动中断

`stream_query()` 返回 `QueryStream`，每次产生一个 DataFrame 分块：

```python
stream = db.stream_query(
    "SELECT * FROM feature_db.user_profile WHERE created_at >= %s",
    params=("2026-01-01",),
    chunksize=50_000,
    progress=True,
)

for chunk in stream:
    consume(chunk)
    if should_stop():
        stream.stop("达到抽样上限")

partial = stream.to_dataframe()
print(partial.attrs["completed"])
print(partial.attrs["rows_read"])
```

进度行为有明确约束：

- `progress=False` 时不执行额外统计 SQL。
- `progress=True` 且未提供 `total_rows` 时，适配器先生成 `SELECT COUNT(1) FROM (...)`。
- 复杂查询可以传入 `count_sql`；已知总数可以传入 `total_rows`，两者都能避免自动包装。
- `retain=True` 默认保留已读分块，因此 `stop()` 或读取期间按 `Ctrl+C` 后可直接合并部分结果。
- 只需要恒定内存消费时设置 `retain=False`；此模式不会保留已经消费的数据，也不能调用 `to_dataframe()` 合并历史分块。

需要直接获得一个完整或部分 DataFrame 时，使用自动消费流的便捷接口：

```python
frame = db.read_query(
    sql,
    params=params,
    chunksize=50_000,
    progress=True,
)
```

若读取被主动中断，返回值的 `DataFrame.attrs` 会记录 `completed=False`、`rows_read`、`total_rows`、`state`、`interrupted_at` 和 `interrupt_reason`。

## 自动建表

`create_table()` 根据 DataFrame 字段类型生成后端 DDL，各数据库可以通过 `dialect_options` 指定物理表参数：

```python
db.create_table(
    first_chunk,
    "feature_db.user_profile",
    dialect_options={
        "key_columns": ["user_id"],
        "engine": "InnoDB",
        "table_comment": "用户特征宽表",
        "column_comments": {
            "user_id": "用户编号",
            "risk_score": "风险分",
        },
    },
)
```

常用方言参数包括 `key_columns`、`column_types`、`column_comments`、`table_comment`、`storage`、`engine`、`partition_columns`、`order_by`、`buckets` 和 `lifecycle`。具体可用项由目标适配器决定。

`column_types` 默认只接受由字母、数字、空格及平衡的 `()` / `<>` 组成的安全类型表达式，例如 `DECIMAL(18, 2)`、`ARRAY<STRING>`、`Nullable(String)`。不接受引号、注释、分号或原样 SQL 片段。

## 流式写入与 mode

`stream_write()` 接受单个 DataFrame、DataFrame 分块迭代器、映射记录迭代器或位置记录迭代器。位置记录必须同时提供 `columns`：

```python
result = db.stream_write(
    dataframe_chunks,
    "feature_db.user_profile",
    mode="r",
    batch_size=10_000,
    key_columns=["user_id"],
)
```

四种 mode 的含义固定：

| mode | 行为 |
|:---:|:---|
| `a` | 追加；主键冲突时保留已有记录，不写入冲突新行 |
| `r` | 追加；主键冲突时以新记录覆盖已有记录 |
| `o` | 保留目标表结构，清空数据后重新写入 |
| `d` | 先校验新 DDL，再删除目标表、重建结构并写入 |

`a/r` 只有在数据库及目标表模型能原生保证对应语义时才开放；不支持时抛出 `DatabaseCapabilityError`，不会使用并发不安全的客户端“先查再写”。

### 后端写入能力

| 后端 / 表模型 | `a` | `r` | `o` | `d` | 说明 |
|:---|:---:|:---:|:---:|:---:|:---|
| MySQL / MariaDB | ✓ | ✓ | ✓ | ✓ | `a` 仅捕获重复键错误；批次遇到其他错误时整体回滚 |
| Oracle 主键表 | ✓ | ✓ | ✓ | ✓ | `a/r` 使用不同 MERGE 分支 |
| Hive 普通表 | ✓ | — | ✓ | ✓ | 传入 `key_columns` 时不能保证冲突忽略 |
| Hive 事务表 | ✓ | 条件支持 | ✓ | ✓ | `r` 需要事务表、MERGE 与 `key_columns` |
| Impala Parquet 表 | ✓ | — | ✓ | ✓ | 不提供唯一键语义 |
| Impala Kudu 表 | ✓ | ✓ | ✓ | ✓ | `INSERT` 丢弃重复主键行，`UPSERT` 覆盖 |
| StarRocks Duplicate Key | — | — | ✓ | ✓ | 重复键模型不保证冲突忽略或覆盖 |
| StarRocks Primary / Unique Key | — | ✓ | ✓ | ✓ | 原生 upsert；无可靠冲突忽略模式 |
| ClickHouse MergeTree | ✓ | — | ✓ | ✓ | `a` 不接受主键冲突保证 |
| ClickHouse ReplacingMergeTree | — | ✓ | ✓ | ✓ | `r` 为最终一致性，读取可按场景使用 `FINAL` |
| MaxCompute 普通表 | ✓ | — | ✓ | ✓ | 无主键唯一约束 |
| MaxCompute 事务表 | ✓ | 条件支持 | ✓ | ✓ | `r` 使用临时表 MERGE，必须提供 `key_columns` |

`WriteResult` 记录接收行数、驱动可确认的插入/更新/跳过数、已提交批次数、失败批次和一致性。后端未返回权威行数时，相应字段保持 `None`，不会按批次长度伪造。

`o/d` 是破坏性操作。`d` 会在 DROP 前完成字段、类型、注释、表模型和 DDL 参数校验；但 DROP 与 CREATE 不是所有后端都能组成原子事务，生产使用前仍应依赖数据库备份、权限控制和变更流程。

## 导出数据库表结构

`export_schema()` 按“每个字段一行”返回 DataFrame。只把输出列名映射为中文，数据库返回的表类型、引擎、数据类型、可空标记、默认值和注释保持原始值：

```python
schema = db.export_schema(
    targets=[
        "risk_db",                       # 整个数据库
        "feature_db.user_profile",       # 指定表
        "catalog_name.risk_db.orders",   # catalog.database.table
    ],
)
```

不传 `targets` 时扫描当前账号可见对象。精确指定的多级表名不存在或不可访问时抛出 `DatabaseMetadataError`；全库扫描中个别对象无权限时，其余结果仍会返回，错误明细保存在 `DataFrame.attrs["错误"]`。

输出列包括数据库类型、目录、数据库名、模式名、表名、完整表名、表类型、表注释、表引擎、字段名、字段序号、数据类型、完整数据类型、Pandas 类型、是否可空、默认值、主键/唯一键/分区键/排序键/分桶键和字段注释。

### Excel 导出

表结构文件只支持 `.xlsx`，并始终通过 `hscredit.excel.dataframe2excel` 生成：

```python
schema = db.export_schema(
    targets=["risk_db", "feature_db.user_profile"],
    output="数据库表结构.xlsx",
    excel_params={
        "sheet_name": "字段清单",
        "title": "数据库字段信息",
        "theme_color": "2639E9",
        "auto_width": True,
    },
)
```

默认参数为 `sheet_name="表结构"`、`title="数据库表结构"`、`index=False`、`decimal=None`、`auto_filter=True` 和 `auto_width=True`。`excel_params` 中的显式值优先。不提供 CSV、TSV 或旧版 `.xls` 导出。

## 扩展其他数据库

第三方适配器通过注册表接入，注册动作不会加载其他数据库驱动：

```python
from hscredit import Database, register_adapter
from hscredit.database.adapters.base import BaseDatabaseAdapter


class CustomDatabaseAdapter(BaseDatabaseAdapter):
    database_type = "custom_db"

    # 实现 query、open_stream、create_table、write_batch、inspect_schema 等契约。


register_adapter("custom_db", CustomDatabaseAdapter, aliases=("custom",))
custom = Database("custom", endpoint="https://database.example")
```

适配器应声明 `DatabaseCapabilities`，按需覆盖连接池、计数 SQL、标识符引用、类型映射、元数据扫描和写入模式。无法原生保证的能力应明确拒绝，不能静默降级。

## 异常与集成验证

数据库模块使用以下异常：

- `DatabaseConnectionError`：连接、连接池或原生客户端初始化失败。
- `DatabaseQueryError`：查询、DDL 或 DML 执行失败。
- `DatabaseWriteError`：流式写入失败，可从 `result` 获取部分提交状态。
- `DatabaseMetadataError`：元数据读取或 Excel 导出失败。
- `DatabaseCapabilityError`：数据库或表模型不能保证所请求语义。

仓库为七类后端提供环境变量门控的真实集成测试。未配置服务时测试会明确 skip，不代表远程数据库已经验证。对应入口位于 `tests/test_database/integration/`。
