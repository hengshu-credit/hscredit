表格神经网络
============

``hscredit.core.models.net`` 提供 ``ResNet``、``FTTransformer``、``TabM``、``TabPFN``，
也可直接从 ``hscredit`` 或 ``hscredit.core.models`` 导入。模型类和后端均按需加载。

.. rubric:: 安装与运行

.. code-block:: bash

   pip install "hscredit[net]"

``net`` 统一安装 PyTorch、``rtdl-revisiting-models``、``tabm``、官方 TabPFN 及既有 TabNet 依赖，
覆盖四种模型；``all`` 通过包含 ``net`` 安装这些依赖。调参额外安装 ``hscredit[net,tune]``。
TabPFN 普通推理兼容 2.0.9；微调和 X-only 预训练需要 Python 3.10+、TabPFN 9.2+。
``net`` 在 Python 3.10+ 安装该新版后端，Python 3.9 保留普通推理能力。
首次使用 TabPFN 时，官方库可能下载预训练权重，也可通过 ``model_path`` 指定本地文件。
默认权重由安装的 TabPFN 版本决定，使用范围及许可请查阅对应官方权重说明。

.. rubric:: 预训练能力核对（2026-10-29）

本节按当前正式导出的四个模型核对，分类沿用信贷的 0/1 二分类。
“官方预训练权重”指发布方已经训练好的权重；``save/load`` 是完整已拟合模型的保存与复用，
不表示再次调用 ``fit`` 会继续训练。现已提供 ``pretrain``、``from_pretrained`` 及 TabPFN 的
``finetune`` 开关，完整说明见 :doc:`training_lifecycle`。

.. list-table::
   :header-rows: 1
   :widths: 19 35 14 14 18

   * - 模型
     - 官方预训练入口
     - 二分类
     - 回归
     - embedding
   * - ResNet
     - 官方提供从头训练架构，未提供通用表格预训练权重入口
     - 支持
     - 支持
     - 预测头前的表示
   * - FTTransformer
     - 官方提供从头训练架构，未提供通用表格预训练权重入口
     - 支持
     - 支持
     - CLS 经归一化及激活后的表示
   * - TabM
     - 官方提供从头训练架构，未提供通用表格预训练权重入口
     - 支持
     - 支持
     - 成员表示或其均值
   * - TabPFN
     - 支持官方预训练权重，使用 model_path 选择
     - 支持
     - 支持
     - 使用后端真实表示；后端明确不支持时可单独跳过

ResNet、FTTransformer、TabM 没有预训练来源时，``fit`` 新建网络和预处理状态；
已有 ``pretrain`` 快照时默认复制快照进行监督训练。加载已监督拟合模型后可直接
调用 ``predict``、``predict_proba``（仅二分类）和 ``get_embeddings``。
这里的 ResNet 是表格架构，不加载图像 ResNet 的 ImageNet 权重。

TabPFN 默认 ``fit`` 建立训练上下文；``finetune=True`` 调用官方梯度微调器更新预训练权重。
可显式传入本地权重，分类和回归应选择官方对应的权重；较新多任务权重是否能共用由官方后端决定。

.. code-block:: python

   classifier = TabPFN(objective="binary", model_path="weights/classifier.ckpt")
   regressor = TabPFN(objective="regression", model_path="weights/regressor.ckpt")
   classifier.fit(X_train, y_class)
   regressor.fit(X_train, y_amount)
   probability = classifier.predict_proba(X_test)
   prediction = regressor.predict(X_test)

权重并不是带有训练上下文的完整 HSCredit 制品。后者使用 ``model.save("model.pkl")`` 和
``type(model).load("model.pkl")``；恢复后可直接预测，无需再次拟合。
``model_path="auto"`` 使用所安装版本的官方默认权重，可能下载权重；版本不同默认权重也可能不同，
不能把“已验证 v2 权重”解读为所有未来权重版本均已验证。

依赖中的 ``pytorch-tabnet`` 原生支持 ``TabNetPretrainer`` 无监督预训练与
``from_unsupervised`` 初始化分类器/回归器，但当前 HSCredit 没有导出 TabNet 模型封装，
因此不把安装该依赖算作已经接入的 net 模型。

核对来源：`rtdl 官方架构与训练示例 <https://github.com/yandex-research/rtdl-revisiting-models/tree/main/package>`_、
`TabM 官方训练说明 <https://github.com/yandex-research/tabm#training>`_、
`TabPFN 权重与模型保存说明 <https://github.com/PriorLabs/TabPFN#-faq>`_、
`TabNet 半监督预训练 <https://github.com/dreamquark-ai/tabnet#semi-supervised-pre-training>`_。

初始能力核对使用仓库验收解释器 ``.audit_tmp/defect-repairs/runtime/Scripts/python.exe``
（Python 3.11.5、PyTorch 2.5.1+cu124、TabPFN 2.0.9、TabM 0.0.3、rtdl 0.0.2）。
2026-10-29 的 net 专项与 Notebook 结构检查共 88 项通过；真实数据 970 行按 776/194 划分，
四种模型分别完成二分类与回归、可用表示提取和八个制品恢复，实际均能输出 embedding。
TabPFN 使用明确指定的本地 v2 分类/回归预训练权重，未把其他版本视为已运行验证。

全库首次执行为 7,058 通过、9 个未配置数据库服务的跳过，以及 6 个 Windows 子进程输出解码失败。
这 6 个失败已复现为 GBK 读取 UTF-8 输出；采用仓库 CI 的 ``PYTHONUTF8=1`` 后，
包含全部失败项的相关 23 项复核通过。格式、关键静态错误、基础设施类型检查、包构建和 Twine 校验通过，
文档构建无警告。上述为初始能力核对的本地结果；随后新增的预训练、微调和续训接口验收
见 :doc:`training_lifecycle`，远程结果须以最终提交对应的 CI 检查为准。

.. rubric:: 共同接口

.. list-table::
   :header-rows: 1
   :widths: 25 38 37

   * - 调用
     - 二分类：objective="binary"
     - 回归：objective="regression"
   * - fit(X, y) / fit(df)
     - 标签必须同时包含 0、1，1 表示坏样本
     - 单个连续数值目标，必须有限且非空
   * - predict(X)
     - 一维 0/1 类别
     - 一维连续预测，保留目标原始单位
   * - predict_proba(X)
     - 两列概率，与 ``classes_`` 顺序一致
     - 明确抛出中文错误
   * - get_embeddings(X) / transform(X)
     - 每行一个二维表示
     - 每行一个二维表示
   * - evaluate(X, y)
     - 沿用 AUC、KS、Gini、LIFT 等统一风控指标
     - 均方根误差、平均绝对误差、决定系数
   * - score(X, y)
     - 准确率
     - R²

四个模型都支持 ``target``、显式 y 优先、DataFrame 字段对齐、sklearn clone/Pipeline、
训练历史和 ``save/load``。保存为 ``.pkl`` 或 ``.joblib`` 会保留预处理及网络；
TabPFN 制品还包含推理所需的训练上下文，加载时需要相应后端和设备环境。
现有风控完整报告、概率评分卡仅适用于二分类；回归请使用 ``evaluate``。
这些网络没有树增益重要性，可通过 ``sklearn.inspection.permutation_importance`` 分析。

.. code-block:: python

   from hscredit import ResNet, FTTransformer, TabM, TabPFN

   model = ResNet(target="FPD", random_state=42)
   model.fit(train_df, eval_set={"验证": valid_df})
   probability = model.predict_proba(test_df)[:, 1]
   prediction = model.predict(test_df)
   embedding = model.get_embeddings(test_df)

   regressor = FTTransformer(objective="regression", random_state=42)
   regressor.fit(X_train, y_train)
   values = regressor.predict(X_test)
   representations = regressor.transform(X_test)

.. rubric:: 训练和表示语义

ResNet/FTTransformer/TabM 仅在实际训练子集拟合中位数填补、标准化及类别字典，
全空数值列保留，类别缺失及未知值编码为 0。类别字段可自动识别或通过
``categorical_features`` 指定字段名/列位置。回归目标在训练子集标准化，预测时还原。
训练支持 ``sample_weight``，零权重行不参与预处理及梯度更新。

默认在未提供 ``eval_set`` 时留出 20% 验证行，用配置的损失早停并恢复最佳参数。
默认损失为交叉熵（分类）或标准化均方误差（回归）。多个评估集仅第一个控制早停；其余集只记录损失和最终指标。
``early_stopping_rounds=None`` 关闭早停；``eval_set=[]`` 或 ``validation_fraction=0``
关闭自动留出。``eval_metric`` 只控制训练后的评估表，不改变早停指标。
若验证标签只有一类，请显式选择适用指标（如 ``eval_metric="logloss"``），AUC 无定义。
测试集应独立于训练及早停验证集。

ResNet 的 embedding 来自残差网络末尾归一化和激活后的表示；FTTransformer 的 embedding
来自 CLS 表示的末尾归一化和激活。两者均为最终预测线性层的输入，维度等于 ``d_block``。

TabM 使用官方 ``tabm.TabM``，每行得到 ``k`` 个成员输出。训练时对各成员独立计算损失后求平均，
避免把平均预测的损失误用于训练；分类推理在 sigmoid 后平均成员概率，回归平均成员预测。
早停监控集成预测的验证损失。``get_embeddings(X)`` 返回成员表示的均值，
``aggregate=None`` 返回 ``(k, 样本数, d_block)``；各成员有各自的预测头，不能用一个线性头
从平均表示重建最终分类概率。数值表示支持 None、``linear_relu`` 和 ``periodic``，
参数放在 ``num_embedding_params``，类别输入由官方网络独热编码。

TabPFN 复用官方数据处理和推理；不支持训练权重的后端会明确拒绝 ``sample_weight``。
``get_embeddings(X)`` 始终请求传入行的 test-token 表示，默认沿集成成员轴求均值，
得到 ``(样本数, 嵌入维度)``。使用 ``aggregate=None`` 返回
``(成员数, 样本数, 嵌入维度)``，单行输入仍保留样本轴。
缺少 embedding 接口或后端明确抛出 ``NotImplementedError`` 时，训练与预测继续可用，
仅调用 ``get_embeddings/transform`` 时提示不支持。可以捕获该异常单独略过表示输出；
字段、维度、非有限数等真实错误仍会抛出，不生成空向量或用预测值冒充 embedding。
后端返回 bfloat16 张量时转换为 float32 NumPy 数组。
监督训练行的表示不是折外特征；将 embedding 用于下游模型时，应在每一折内训练网络并提取验证行表示。

.. rubric:: 损失、优化器和常规参数

ResNet、FTTransformer、TabM 使用相同训练配置：

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 参数
     - 语义
   * - ``loss`` / ``loss_params``
     - 二分类默认 bce，回归默认 mse；还支持 mae、huber、HSCredit 损失名称/对象、PyTorch 损失类/实例及张量函数
   * - ``optimizer``
     - adamw、adam、sgd、rmsprop、adagrad 或 PyTorch 优化器类；不接收已绑定其他网络的实例，不支持闭包式 LBFGS
   * - ``optimizer_params``
     - momentum、betas、eps 等额外参数；学习率和衰减只通过 learning_rate、weight_decay 配置，避免调参被字典覆盖
   * - ``d_block`` / ``n_blocks`` / ``dropout``
     - 表示宽度、网络深度、丢弃概率；FTTransformer 的 d_block 必须被 n_heads 整除
   * - ``k`` / ``arch_type``
     - TabM 成员数与架构；默认 32 / tabm，还可选 tabm-mini、tabm-packed
   * - ``max_epochs`` / ``batch_size``
     - 最大轮数和批量大小；ResNet 的 batch_size 至少为 2
   * - ``early_stopping_rounds`` / ``validation_fraction``
     - 早停耐心与内部留出比例；关闭早停时可使用全部训练行
   * - ``device`` / ``random_state``
     - auto/cpu/cuda 设备与局部随机种子；训练不覆盖调用方的 torch 随机状态

训练后可查看 ``optimizer_``、``loss_fn_``、``get_params()``、``get_training_history()``、
``get_learning_curve()`` 和 ``get_eval_results()``。默认不配置调度器或混合精度。

HSCredit 损失用法：

.. code-block:: python

   from hscredit import TabM, FocalLoss

   model = TabM(
       loss=FocalLoss(alpha=0.75, gamma=2.0),
       optimizer="adamw", optimizer_params={"betas": (0.9, 0.99)},
       learning_rate=0.002, weight_decay=0.0003,
       d_block=128, n_blocks=3, k=16,
       max_epochs=100, batch_size=256, random_state=42,
   )
   model.fit(X_train, y_train, sample_weight=train_weights)

``BaseLoss`` 必须提供逐行 ``loss_values``，并声明 ``is_additive=True``。
全量排序、动态总体权重和绑定逐行金额数组的损失不能直接用于小批次训练。
使用 ``FocalLoss().to_torch()`` 或 ``TorchLossAdapter`` 可单独获得 PyTorch 模块；
该适配将概率梯度经 sigmoid 链式法则传回 logits，涉及 CPU/设备数据交换。
纯 PyTorch 损失函数可直接在设备上计算：

.. code-block:: python

   import torch
   from torch.nn import functional as F

   def weighted_bce(logits, target, positive_weight=2.0):
       values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
       return values * torch.where(target == 1, positive_weight, 1.0)

   model = ResNet(
       loss=weighted_bce, loss_params={"positive_weight": 2.0},
       optimizer="sgd", optimizer_params={"momentum": 0.9, "nesterov": True},
       learning_rate=0.01,
   )

函数接收一维原始 logits（分类）或标准化目标空间中的预测（回归），必须返回同形状的逐行张量，
并保留自动微分计算图。TabM 会先展平成 ``批量 × 成员数``，逐成员计算后归约。
统一训练层再应用 ``sample_weight`` 并求均值；不能自行返回已归约的标量。
PyTorch 损失实例会复制并设置 ``reduction="none"``，不会修改调用方对象。
回归 ``HuberLoss.delta`` 等参数作用于标准化目标空间，公开预测和回归指标仍保持原始单位。

.. rubric:: net 调参与最终重训

``model.tune``、``ModelTuner`` 和 ``AutoTuner.create`` 均支持四种 net 模型。
分类默认搜索指标为 KS，回归默认 RMSE；回归还可选择 mse、mae、r2，或传
``(y_true, prediction) -> float`` 函数并显式指定 direction。
``loss`` 是梯度训练目标，``metric`` 是独立验证/交叉验证的选优指标，``eval_metric`` 是模型最终评估表配置。

.. code-block:: python

   best = TabM(k=16, d_block=128, max_epochs=60, loss="focal", random_state=42).tune(
       X_train, y_train, n_trials=10, cv=3, metric="auc", n_jobs=1,
       search_space={
           "d_block": [64, 128, 256],
           "learning_rate": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
           "loss_params__gamma": [1.0, 2.0],
       },
   )
   history = best.tuner.get_optimization_history()
   oof = best.tuner.get_oof_predictions()

``optimizer_params__momentum``、``loss_params__gamma``、``num_embedding_params__d_embedding``
支持嵌套参数搜索。优化器的参数存在条件关系，例如 momentum 只属于 SGD；
同时搜索优化器类型和特有参数时，使用搜索空间函数为每种优化器返回适用的参数字典。
完整示例见 :download:`net_training_tuning.ipynb <../../examples/net_training_tuning.ipynb>`。

默认网络搜索空间覆盖宽度、深度、dropout、学习率、权重衰减及批量大小；
损失、优化器类型、TabM 的 k/arch_type 保持固定，可通过显式空间另行研究。
FTTransformer 默认候选宽度按配置的 n_heads 对齐。TabM 官方建议分开比较固定 k 的搜索，
而不是同时搜索成员数和所有结构参数；当前默认空间遵循这一建议。

多折验证的早停只使用各训练折内部划分；外层验证折用于评分。
单次外部验证会直接用于早停与选优，仍需独立测试集评价最终效果。
``get_best_model(full_data=True)`` 按最佳试验各折最优轮数的中位数重训，
关闭内部早停并使用完整训练输入；不把外部验证或测试集并入训练。
回归的折外/验证预测表使用中文列名“预测值”，不输出分类 LIFT。

TabPFN 默认搜索 ``n_estimators``、``softmax_temperature`` 等推理参数；启用 ``finetune`` 后，
默认空间改为微调学习率与权重衰减。官方微调配置放在 ``finetune_params``，
不接收普通网络的任意 ``loss`` 或优化器实例。详见 :doc:`training_lifecycle`。

完整可运行演示见 :download:`net_models.ipynb <../../examples/net_models.ipynb>`，
使用项目真实数据中的三项约定特征、商品类别、FPD 和放款金额。

参考实现：`rtdl 论文作者实现 <https://github.com/yandex-research/rtdl-revisiting-models>`_、
`TabM 官方实现和训练协议 <https://github.com/yandex-research/tabm>`_、
`TabPFN 官方实现 <https://github.com/PriorLabs/TabPFN>`_、
`TabPFN embedding 文档 <https://docs.priorlabs.ai/capabilities/embeddings>`_。

.. rubric:: 模型参数

.. autoclass:: hscredit.core.models.net.ResNet
   :members:
   :inherited-members:

.. autoclass:: hscredit.core.models.net.FTTransformer
   :members:
   :inherited-members:

.. autoclass:: hscredit.core.models.net.TabPFN
   :members:
   :inherited-members:

.. autoclass:: hscredit.core.models.net.TabM
   :members:
   :inherited-members:

.. autoclass:: hscredit.core.models.losses.TorchLossAdapter
   :members:

.. rubric:: 验收记录（2026-10-27）

解释器为仓库既有的 ``E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe``，
Python 3.11.5、PyTorch 2.5.1+cu124、rtdl-revisiting-models 0.0.2、TabPFN 2.0.9、
TabM 0.0.3、rtdl-num-embeddings 0.0.12、Optuna 4.9.0、NumPy 2.4.6、Pandas 2.3.3、
scikit-learn 1.6.1。只向该验收环境补装缺失依赖，未升级共享 Anaconda。

* net 最终专项：``test_net_models.py`` 与 ``test_net_training_and_tuning.py`` 共 81 项通过。
  覆盖三种 TabM 架构及数值表示、分类/回归、成员损失独立公式、损失梯度有限差分、
  五种优化器、嵌套配置、早停与全量重训、回归折外预测、评估权重、保存续跑、旧模型恢复和依赖懒加载。
  TabPFN 单测使用边界替身，真实权重由 Notebook 验证。
* 相关回归：``tests/test_models/``、``tests/test_report/test_metric_audit_real_data.py``、
  ``tests/test_utils/test_documentation_examples.py`` 回归阶段共 1,769 项通过；此后补充的架构参数化及
  NumPy 整数配置边界包含在上述 81 项最终专项中。两组结果有重叠，不能相加作为覆盖率。
  大范围回归有 469 条既有运行/弃用等警告。
* 两份 Notebook 的 6 + 7 个代码单元格全部执行。真实数据 970 行，基础示例为 776 行训练
  （神经网络内部再留出验证）、194 行独立测试；四种模型分别执行 FPD 二分类及放款金额回归，
  核对预测、embedding、独立计算的误差及八个制品恢复。训练/调参示例还验证条件优化器搜索、
  损失参数搜索、二折回归与外部验证、原生 TabPFN 参数搜索；后者使用真实 v2 权重及 float32 推理。
* 19 个相关 Python 文件及两份 Notebook 代码通过 Python 3.9 AST 语法检查；实际运行只验证 Python 3.11.5。
  Black 与 Flake8 检查通过；构建元数据确认 ``net`` 包含 tabm、tabpfn 等全部依赖。
  Sphinx 构建成功，本次 net/训练/调参文档无警告或错误；其他页面仍有分箱 docstring 格式、缺失图像及未入目录等诊断。

可复现的测试入口：

.. code-block:: powershell

   $netPython = 'E:\workspace\hscredit\.audit_tmp\defect-repairs\runtime\Scripts\python.exe'
   & $netPython -m pytest tests/test_models/test_net_models.py tests/test_models/test_net_training_and_tuning.py -q --tb=short
   & $netPython -m pytest tests/test_models/ tests/test_report/test_metric_audit_real_data.py tests/test_utils/test_documentation_examples.py -q --tb=short

真实权重验收请使用上方 Notebook，并将内核设为具备这些依赖的解释器。验证数据、
所选权重和少量演示训练轮数不构成模型效果基准或效果保证。
