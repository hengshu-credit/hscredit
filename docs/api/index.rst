API 参考
========

本节按 ``hscredit`` 的包目录列出公开 API。所有文档均由各组件的 docstring 自动生成，
参数说明、属性与样例与源码保持同步。

``core/models`` 归入核心模块，``report/mining`` 归入报告模块；顶层包和模块在下方并列。
参数写法与输入约定见 :doc:`conventions`，业务教程见“建模实践”。

顶层包 ``hscredit`` 聚合了各子模块的公开符号，并额外转出 sklearn 的 ``Pipeline``、
``make_pipeline``、``VotingClassifier`` / ``StackingClassifier`` 等集成组件，便于直接
``from hscredit import ...`` 使用。

.. toctree::
   :maxdepth: 3

   core
   report
   database
   excel
   utils
   skills_runtime
   exceptions
