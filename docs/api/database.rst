数据库 ``hscredit.database``
==============================

统一数据库连接池、参数化查询、流式读取、自动建表、分批写入、元数据导出与适配器扩展 API。
使用流程和后端能力矩阵见 :doc:`../database`。

数据库门面
----------

.. autoclass:: hscredit.database.client.Database
   :members:
   :show-inheritance:

流式查询
--------

.. autoclass:: hscredit.database.stream.QueryStream
   :members:
   :show-inheritance:

适配器注册
----------

.. autofunction:: hscredit.database.registry.register_adapter

.. autofunction:: hscredit.database.registry.get_adapter_class

.. autofunction:: hscredit.database.registry.available_adapters

公共类型
--------

.. autoclass:: hscredit.database.types.PoolOptions
   :members:

.. autoclass:: hscredit.database.types.DatabaseCapabilities
   :members:

.. autoclass:: hscredit.database.types.WriteResult
   :members:

.. autoclass:: hscredit.database.types.StreamState
   :members:

.. autoclass:: hscredit.database.metadata.QualifiedTarget
   :members:

.. autoclass:: hscredit.database.metadata.MetadataInspection
   :members:

异常
----

.. automodule:: hscredit.database.exceptions
   :members:
   :show-inheritance:
