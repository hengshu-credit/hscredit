API 参考
========

本节按模块列出 ``hscredit`` 的公开 API。所有文档均由各组件的 docstring 自动生成，
参数说明、属性与样例与源码保持同步。

顶层包 ``hscredit`` 聚合了各子模块的公开符号，并额外转出 sklearn 的 ``Pipeline``、
``make_pipeline``、``VotingClassifier`` / ``StackingClassifier`` 等集成组件，便于直接
``from hscredit import ...`` 使用。

.. toctree::
   :maxdepth: 3

   core
   modeling
   tooling
