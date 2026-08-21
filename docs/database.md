# 数据库与 NoSQL 连接池、读写及表结构导出

`hscredit.database` 为风控建模和数据分析流程提供统一的数据存储入口，覆盖 SQL 数据库以及 Redis、MongoDB 的连接池。SQL 后端支持参数化查询、可中断流式读取、DataFrame 建表、分批写入和全库字段清单导出；NoSQL 后端提供同名的单条、批量和自适应 CRUD 方法。数据库驱动均为可选依赖；普通 `import hscredit` 不会加载任何数据库驱动。

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
| Redis | `pip install hscredit[db-redis]` | redis-py 原生 ConnectionPool / BlockingConnectionPool |
| MongoDB | `pip install hscredit[db-mongodb]` | PyMongo MongoClient 内建连接池 |
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

## Redis 与 MongoDB 的统一 NoSQL 方法

Redis 和 MongoDB 对外使用相同的方法名；参数中的 `resource` 在 Redis 表示 key 或 keys，在 MongoDB 表示集合名：

| 方法 | Redis | MongoDB |
|:---|:---|:---|
| `read_one` / `read_many` | `GET` / `MGET` | `find_one` / `find` |
| `write_one` / `write_many` | `SET` / `MSET` | 单条或批量 insert/update/replace |
| `delete_one` / `delete_many` | 单 key / 多 key `DELETE` | `delete_one` / `delete_many` |
| `exists` | key 是否存在 | 是否存在匹配文档 |
| `read` / `write` / `delete` | 按单 key、keys 或映射自动分派 | 按 `limit`、文档形态和 `many` 自动分派 |

Redis 连接示例：

```python
redis_db = Database(
    "redis",
    url="redis://127.0.0.1:6379/0",
    decode_responses=True,
    pool_options={
        "max_connections": 20,
        "blocking": True,
        "timeout": 2,
    },
)

redis_db.write("score:1001", "720", ttl=3600)
redis_db.write({"score:1002": "680", "score:1003": "700"})
assert redis_db.read("score:1001") == "720"
scores = redis_db.read(["score:1002", "score:1003"])
redis_db.delete(["score:1001", "score:1002", "score:1003"])
```

MongoDB 的 `MongoClient` 自身管理连接池，`pool_options` 会转换为 PyMongo 的 `minPoolSize`、`maxPoolSize`、`maxConnecting`、`waitQueueTimeoutMS` 和 `maxIdleTimeMS`：

```python
mongo_db = Database(
    "mongodb",
    uri="mongodb://127.0.0.1:27017/risk",
    database="risk",
    pool_options={"min_pool_size": 1, "max_pool_size": 20},
)

mongo_db.write("model_score", {"user_id": 1001, "score": 720})
mongo_db.write(
    "model_score",
    [{"user_id": 1002, "score": 680}, {"user_id": 1003, "score": 700}],
)
high_scores = mongo_db.read(
    "model_score",
    {"score": {"$gte": 700}},
    sort=[("score", -1)],
)
mongo_db.write_one(
    "model_score",
    {"$set": {"score": 730}},
    selector={"user_id": 1001},
    mode="update",
)
mongo_db.delete("model_score", {"user_id": 1001})
```

自适应规则保持可预测且优先保护数据：

- Redis 的字符串或 bytes key 使用单条方法，key 序列和 key-value 映射使用批量方法。
- MongoDB `read()` 默认返回匹配文档列表；`limit=1` 或 `many=False` 返回单个文档。
- MongoDB `write()` 对单个映射使用 `write_one()`，对文档序列使用 `write_many()`。
- MongoDB `delete()` 默认只删除首个匹配文档；批量删除必须显式传入 `many=True`。空 selector 的批量删除还必须设置 `allow_all=True`。

所有写入、更新和删除返回 `NoSQLWriteResult`，统一暴露 `acknowledged`、`affected_count`、`matched_count`、`modified_count` 和 `identifiers`。需要发布订阅、pipeline、聚合等高级能力时，可使用 `db.native_client`；Redis 和 MongoDB 不接受 SQL `query()`。

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

### 大 JSON 字段按路径读取

当查询包含非常大的 JSON 字段、但实际只需要少量子字段时，可以通过 `columns` 和 `json_fields` 把路径提取下推到数据库。原始 JSON 不会传输到 Python：

```python
json_fields = {
    "huge_json": {
        "customer_id": "$.customer.id",
        "city": ("$.address.city", "未知"),
        "risk_tags": ("$.risk.tags", []),
    }
}

stream = db.stream_query(
    """
    SELECT id, created_at, huge_json
    FROM feature_db.user_profile
    WHERE created_at >= %s
    """,
    params=("2026-01-01",),
    columns=["id", "created_at"],
    json_fields=json_fields,
    result="records",
    chunksize=50_000,
    progress=True,
)

for records in stream:
    consume(records)
```

`json_fields` 使用“JSON 源字段 → 输出字段 → 字段定义”的顺序：

- 字符串定义只包含 JSONPath，路径缺失或结果为 `null` 时返回 `None`；
- `(JSONPath, 默认值)` 二元组可以指定缺失默认值；列表、字典等可变默认值会为每行独立复制；
- 不指定返回类型，也不在 Python 中强制转换；值保持目标数据库 JSON 函数的原始返回形式；
- `columns` 是需要原样保留的普通字段，不能包含任何 JSON 源字段；输出顺序为 `columns` 后接 `json_fields` 的定义顺序；
- 原始 SQL 必须输出 `columns` 和 JSON 源字段，公共层再包装为只选择所需结果的外层查询；
- JSONPath 必须以 `$` 开头，并拒绝引号、反斜线、分号、注释和控制字符，避免路径被当作 SQL 片段。

`result` 沿用整个 Database 模块已有取值：

| result | `stream_query()` 每批结果 | `read_query()` 合并结果 |
|:---|:---|:---|
| `dataframe` | DataFrame | DataFrame |
| `records` | `list[dict]` | `list[dict]` |
| `rows` | 原始行元组列表 | 原始行元组列表 |

主动停止和键盘中断仍会保留当前已读数据。DataFrame 的读取状态位于 `attrs`；使用列表结果时，可从 `QueryStream.state`、`rows_read` 和 `interrupt_reason` 查看流状态。启用进度时，`COUNT(1)` 针对原始查询执行，不会重复计算 JSON 投影表达式；未启用进度时不会查询总数。

各适配器分别使用 MySQL `JSON_EXTRACT`、Oracle `JSON_VALUE/JSON_QUERY`、StarRocks `GET_JSON_STRING`、ClickHouse `JSON_VALUE/JSON_QUERY`、Hive/Impala `GET_JSON_OBJECT` 和 MaxCompute `JSON_EXTRACT`。第三方适配器可实现 `json_extract_expression()` 获得相同公共接口。

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

### 字符串长度与 JSON 内容推断

没有显式指定 `column_types` 时，适配器会分析当前用于建表的 DataFrame。流式写入只分析首个有效分块，不会预执行、抽样或重复消费后续用户迭代器。

字符串画像包含最大字符数、最大 UTF-8 字节数以及 JSON 标记。JSON 使用严格规则：所有非空值都必须是字符串、都能被标准 JSON 解析，并且顶层都是对象或数组。JSON 标量（如 `123`、`true`、`"text"`）、无效 JSON、混合普通文本或混入 Python `dict/list` 时均不会自动推断为 JSON。显式 `column_types` 始终优先。

| 后端 | 普通字符推断 | JSON 字符串推断 |
|:---|:---|:---|
| MySQL / MariaDB | 观察长度增加 20% 余量并落到稳定 VARCHAR 档位；超过 `varchar_max_length=255` 后按 UTF-8 字节数使用 `TEXT`、`MEDIUMTEXT`、`LONGTEXT` | `JSON` |
| Oracle | 长度不超过 `varchar_max_length=4000` 时使用自适应 `VARCHAR2(n CHAR)`，否则使用 `CLOB` | 默认 `CLOB` 兼容旧版本；`native_json=True` 时使用原生 `JSON` |
| StarRocks | 使用自适应 `VARCHAR(n)`；超过单字段 65533 字节时明确报错 | `JSON`，SQL 协议写入自动使用 `parse_json(%s)` |
| ClickHouse | `String`，不按长度截断 | `json_type="auto"` 在服务器 25.3+ 使用 `JSON`，旧版本回退 `String`；也可显式指定 `JSON` 或 `String` |
| Hive / Impala | `STRING`，保留后端无固定长度语义 | 继续使用 `STRING`，不伪造后端原生 JSON 类型 |
| MaxCompute | `STRING` | `JSON` |

可以通过 `infer_json=False` 关闭内容识别。Oracle 可调整 `varchar_max_length`；ClickHouse 可用 `json_type` 固定兼容策略。MySQL 还支持以下容量参数：

- `varchar_max_length`：VARCHAR 的字符数上限，默认 255；
- `varchar_max_bytes`：VARCHAR 的行内字节预算，默认 65533；
- `string_length_headroom`：观察长度的扩容系数，默认 1.2，不能小于 1；
- `charset_max_bytes_per_character`：未知或自定义字符集的单字符最大字节数。对 `utf8mb4`、`utf8`、`latin1` 等已知字符集，不能设置为低于其安全宽度的值。

由于推断只依据首个建表分块，生产任务应让首批数据具有代表性；字段容量或后端版本要求明确时，优先使用 `column_types` 覆盖。

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

仓库为九类后端提供环境变量门控的真实集成测试。Redis 使用 `HSCREDIT_TEST_REDIS_URL`，MongoDB 使用 `HSCREDIT_TEST_MONGODB_URI` 和可选的 `HSCREDIT_TEST_MONGODB_DATABASE`。未配置服务时测试会明确 skip，不代表远程数据库已经验证。对应入口位于 `tests/test_database/integration/`。
