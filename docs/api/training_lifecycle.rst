迁移续训
========

net 的无标签预训练使用 ``pretrain``；业务分类/回归仍使用 ``fit``、``predict``、
``predict_proba`` 和 ``get_embeddings``。boost 使用 ``fit`` 的续训参数追加提升轮数。

仅用 X 的 net 预训练
--------------------

ResNet、FTTransformer、TabM、TabPFN 都提供 ``pretrain(X, ...)``，不会读取外部业务标签。
如果 X 中有已配置的 target 列，会直接删除该列。预训练返回当前模型，并允许
``get_embeddings``、``transform``、``get_native_model`` 和 ``save``；
业务 ``predict/predict_proba`` 只有在监督 fit 完成后才可调用，避免输出未经训练的业务预测头。

.. code-block:: python

   from hscredit import ResNet

   encoder = ResNet(d_block=64, max_epochs=20, random_state=42)
   encoder.pretrain(X_unlabeled, eval_set=[X_pretrain_valid], pretraining_ratio=0.3)
   embedding = encoder.get_embeddings(X_new)
   encoder.save("pretrained.pkl")
   encoder.fit(X_train, y_train)  # 自动复制本实例的预训练快照，进行监督训练

   regressor = ResNet(objective="regression", d_block=64, random_state=42)
   regressor.fit(X_train, y_amount, from_pretrained=encoder)

``from_pretrained`` 的语义：None 自动使用本实例已有快照；True 要求本实例有快照；
False 忽略快照重新训练；也可传入已 pretrain 的同类模型。复制权重不会修改来源模型。
接续时冻结预训练的填补、标准化和类别字典，字段及网络结构必须兼容。
ResNet/FTTransformer/TabM 可跨二分类和回归迁移编码器，TabPFN 则要求基础网络任务类型相同。
重新调用 pretrain 会产生新快照；重新 fit 从选定快照开始，不表示接续上一次监督训练。

普通网络的掩码重建
~~~~~~~~~~~~~~~~~~

ResNet、FTTransformer、TabM 随机遮盖特征，用标准化数值的均方误差和类别交叉熵重建。
损失仅覆盖被遮盖且原本有观测值的单元格，不把填补值当真值。每个有观测的样本至少遮盖一个字段，
因此在少特征或低遮盖概率下，实际比例可能高于 pretraining_ratio。
TabM 分别计算每个成员的重建损失后平均。优化器及其参数沿用构造器；
构造器 loss 仅用于监督 fit，pretrain 使用上述专用重建目标。

``max_epochs``、``batch_size``、``early_stopping_rounds``、``validation_fraction``
可在 pretrain 中覆盖，None 沿用构造器配置；validation_fraction=0 关闭自动留出。
也可提供无标签矩阵、列表或具名字典作为 eval_set。预处理仅在实际训练子集拟合，
验证掩码固定，按重建损失早停并恢复最佳编码器。可用 get_learning_curve 查询逐轮重建损失。

TabPFN 的字段预测式预训练
~~~~~~~~~~~~~~~~~~~~~~~~~~

TabPFN 使用官方微调器做字段预测式自监督训练，而不是用无标签 X 建立推理缓存。
随机遮盖 X 后，为被遮盖字段构造预测任务，目标字段不会留在输入中泄漏答案；
各任务顺序传递更新后的权重，不会为每个任务重置到官方初始权重。

回归基础网络重建连续数值；分类基础网络将数值经 hscredit 等频分箱后按二进制位重建。
类别字段使用类别编码位作为目标，避免把类别编号的距离当作回归误差。
常量、观测或类别样本不足的任务跳过；没有可用任务时明确报错。
此实现至少需要两个特征，max_epochs 是每个字段任务的轮数，batch_size 是上下文加查询块的样本上限。
预训练后的 embedding 使用最后一个字段预测任务的上下文，监督 fit 后改为业务上下文。

.. code-block:: python

   from hscredit import TabPFN

   model = TabPFN(
       n_estimators=1,
       finetune_params={"model_version": "v2", "max_epochs": 1,
                        "batch_size": 128, "validation_fraction": 0},
   )
   model.pretrain(X_unlabeled, pretraining_ratio=0.5)
   model.fit(X_train, y_train)  # 保留预训练权重，建立业务目标的推理上下文

TabPFN 微调及其 X-only 预训练要求 Python 3.10+、tabpfn>=9.2。
Python 3.9 或较早后端仍可使用普通推理；请求高级训练时会明确报告依赖要求。
权重版本及许可沿用官方规则，显式 model_path 可使用本地权重。

TabPFN 可选监督微调
-------------------

.. code-block:: python

   model = TabPFN(
       finetune=True,
       finetune_params={"max_epochs": 10, "learning_rate": 1e-5,
                        "validation_fraction": 0.1, "early_stopping_rounds": 5},
   )
   model.fit(X_train, y_train)
   model.fit(X_train, y_train, finetune=False)  # 本次仅建立推理上下文
   model.fit(X_train, y_train, finetune=True,
             finetune_params={"max_epochs": 5})

fit 的开关覆盖构造器，fit 的 finetune_params 同名配置覆盖构造器字典。
参数别名如下，同一含义的两种名称不能重复提供：

.. list-table::
   :header-rows: 1

   * - HSCredit 参数
     - 官方参数
   * - max_epochs
     - epochs
   * - batch_size
     - n_finetune_ctx_plus_query_samples
   * - validation_fraction
     - validation_split_ratio
   * - early_stopping_rounds
     - early_stopping / early_stopping_patience；None 关闭早停

其余官方构造参数也放在 finetune_params，例如 model_version、use_lr_scheduler、
n_estimators_finetune。model_path="auto" 时尊重所选版本的默认权重，显式路径优先。
官方微调不支持 sample_weight，传入会明确拒绝。预测和 embedding 使用实际微调后的官方推理估计器。

``finetune_params`` 还提供 ``mixed_precision``，默认 False 使用全精度前向及损失计算，
避免大金额回归在短轮数训练中因 FP16 梯度缩放而跳过参数更新。True 使用官方混合精度流程，
可降低显存开销。全精度模式要求 inference_precision 为 auto 或 torch.float32，
会将底层推理精度也设为 torch.float32。没有有效训练批次或出现非有限训练损失时明确报错。

与 tune 的关系
--------------

参数字典支持双下划线搜索，如 finetune_params__learning_rate。
TabPFN 启用 finetune 后默认搜索微调学习率和权重衰减；未启用时仍搜索推理参数。
最终 full_data 重训关闭内部留出和早停，使用完整训练输入及配置轮数；
官方不提供最优轮数时不会伪造最佳迭代。

tune 会克隆未拟合模型，不隐式携带本实例的预训练状态。使用历史无标签数据的预训练快照时，
可显式传 ``fit_params={"from_pretrained": pretrained_model}``。
独立测试集不得参与预处理、预训练或监督训练；跨验证折复用的预训练快照必须来自允许共享的历史数据。

boost 继续训练
--------------

.. code-block:: python

   from hscredit import XGBoost

   model = XGBoost(n_estimators=100, random_state=42).fit(X_old, y_old)
   model.set_params(n_estimators=20)
   model.fit(X_new, y_new, continue_training=True)  # 追加最多 20 轮

   updated = XGBoost(n_estimators=20, random_state=42)
   updated.fit(X_new, y_new, init_model=model)  # 来源 model 不被修改

.. list-table::
   :header-rows: 1

   * - 模型
     - 原生续训入口
     - 轮数语义
   * - XGBoost
     - xgb_model（同时保留原生参数名）
     - n_estimators 为本次新增轮数
   * - LightGBM
     - init_model
     - n_estimators 为本次新增轮数
   * - CatBoost
     - init_model
     - iterations 为本次新增轮数
   * - NGBoost
     - partial_fit
     - n_estimators 为本次新增轮数

continue_training=True 使用本实例旧模型；init_model 可接收同类已拟合 HSCredit 模型或后端支持的原生对象/路径。
两者不能并用。传 HSCredit 模型时按来源字段对齐，缺失字段报错；原生文件的校验由对应后端完成。
默认普通 fit 仍重新训练，不隐式追加树。来源模型会复制；保存并加载完整模型后也可继续训练。
CatBoost 自动沿用来源模型的类别字段，原生 init_model 续训仅支持 CPU。
XGBoost 新阶段会清除旧的最佳迭代标记，再由当前阶段的早停配置选优，避免新增树被旧标记排除。
NGBoost 原生路径需先 load 为对象；其续训固定旧初始分布，并使用本次早停配置，
并要求学习率、基学习器、目标/分布及相关算法配置兼容，以免改变已有树的贡献。
概率评分刻度仍按本次训练数据或显式业务先验重新拟合。

原生文档：`XGBoost 续训 <https://xgboost.readthedocs.io/en/stable/python/examples/continuation.html>`_、
`LightGBM fit <https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html>`_、
`CatBoost init_model <https://catboost.ai/docs/en/concepts/python-reference_catboost_fit>`_、
`TabPFN 微调 <https://docs.priorlabs.ai/capabilities/fine-tuning>`_、
`TabNet 预训练参考 <https://github.com/dreamquark-ai/tabnet#semi-supervised-pre-training>`_。

完整真实数据演示见 :download:`pretraining_and_continuation.ipynb <../../examples/pretraining_and_continuation.ipynb>`。

本地验收记录（2026-10-29）
--------------------------

解释器为 ``.audit_tmp/defect-repairs/runtime/Scripts/python.exe``（Python 3.11.5），
设置 ``PYTHONUTF8=1`` 和 ``PYTHONIOENCODING=utf-8``。训练扩展、net/boost 参数契约、
net 调参、公开方法文档及 Notebook 结构共 254 项定向检查通过。
Black、关键 Flake8、基础设施 Mypy、Python 3.9 语法检查、环境检查和无警告 Sphinx 构建通过。

真实数据 Notebook 读取 ``examples/hscredit_yyp.xlsx`` 的 970 行，按 620/156/194
划分训练、验证、独立测试集。验证四种 net 的 X-only 参数更新、二分类/回归、embedding 和制品恢复，
TabPFN 普通 fit 不改权重、微调实际改权重，以及四种 boost 从 5 轮追加至 10 轮。
参数更新按可训练参数内容核对，不把运行统计量变化当作权重学习；此小轮数验收不作为效果基准。

实际权重运行使用独立目录中的 TabPFN 9.2.0、官方本地 v2 分类/回归权重，
PyTorch 2.5.1+cu124、TabM 0.0.3、XGBoost 2.0.2、LightGBM 4.1.0、CatBoost 1.2.10、NGBoost 0.5.11。
未升级共享 Anaconda，也未将 v2 权重验证推广到所有后续权重版本。
远程完整测试及版本矩阵结果以本次最终提交 SHA 的 CI 为准，不能用上述本地定向结果替代。
