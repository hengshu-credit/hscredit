"""表格 ResNet 模型。"""

from ._torch_model import TorchTabularModel


class ResNet(TorchTabularModel):
    """带残差连接和批归一化的表格网络，支持二分类与回归。

    **参数**

    :param objective: binary 或 regression；二分类标签为 0/1，1 表示坏样本。
    :param d_block: 残差块宽度，也是输出 embedding 的维度，默认 128。
    :param n_blocks: 残差块数量，默认 3；块内隐藏宽度为 d_block 的两倍。
    :param dropout: 块内和残差分支的丢弃概率，默认 0.1。
    :param cat_embedding_dim: 每个类别字段的表示维度，默认 16。
    :param categorical_features: 类别字段名或零起始列位置列表；None 自动识别非数值/布尔字段。
    :param learning_rate: 优化器学习率，默认 0.001。
    :param weight_decay: 优化器权重衰减，默认 0.00001。
    :param max_epochs: 最大训练轮数，默认 100。
    :param batch_size: 训练及推理批量，至少 2，默认 256。
    :param early_stopping_rounds: 验证损失连续未改善轮数，默认 10；None 关闭早停。
    :param validation_fraction: 无 eval_set 时的自动验证集比例，默认 0.2；0 关闭留出。
    :param device: auto、cpu 或 cuda[:设备编号]，auto 优先使用 CUDA。
    :param target: 可选目标字段名，支持 fit(df)；显式 y 优先且目标列始终移除。
    :param eval_metric: 最终评估集指标；逐轮早停按配置的 loss，不随此参数改变。
    :param random_state: 随机种子，默认 None。
    :param verbose: 是否输出中文逐轮日志。
    :param scorecard_params: 二分类概率评分卡配置，回归不使用。
    :param history_policy: 训练历史策略，默认 summary。
    :param max_history: 最大训练历史数量，默认 20。
    :param weight_type: cost 或 frequency，沿用统一模型的训练权重与评分刻度语义。
    :param loss: None 使用默认损失；支持名称、BaseLoss、PyTorch 损失类/实例和逐行张量函数。
    :param loss_params: 损失名称/类/函数的参数字典，训练层统一使用逐行损失。
    :param optimizer: adamw（默认）、adam、sgd、rmsprop、adagrad 或 PyTorch 优化器类。
    :param optimizer_params: 优化器额外参数，如 momentum、betas；lr/weight_decay 使用模型同名配置。

    **属性**

    :ivar embedding_dim_: 表示维度；get_embeddings/transform 返回二维数组。
    :ivar n_iter_: 实际训练轮数，best_iteration_ 为早停最优轮（从 1 开始）。
    :ivar preprocessor_: 仅使用实际训练行拟合的填补、标准化和类别编码。

    **参考样例**

    >>> model = ResNet(target='FPD', random_state=42)
    >>> model.fit(train_df)
    >>> probability = model.predict_proba(test_df)[:, 1]
    >>> embedding = model.get_embeddings(test_df)

    架构来源：https://github.com/yandex-research/rtdl-revisiting-models
    """

    _network_kind = "ResNet"

    def __init__(
        self,
        objective="binary",
        d_block=128,
        n_blocks=3,
        dropout=0.1,
        cat_embedding_dim=16,
        categorical_features=None,
        learning_rate=0.001,
        weight_decay=0.00001,
        max_epochs=100,
        batch_size=256,
        early_stopping_rounds=10,
        validation_fraction=0.2,
        device="auto",
        target=None,
        eval_metric=None,
        random_state=None,
        verbose=False,
        scorecard_params=None,
        history_policy="summary",
        max_history=20,
        weight_type="cost",
        loss=None,
        loss_params=None,
        optimizer="adamw",
        optimizer_params=None,
    ):
        super().__init__(
            objective=objective,
            target=target,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=1,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            verbose=verbose,
            scorecard_params=scorecard_params,
            history_policy=history_policy,
            max_history=max_history,
            weight_type=weight_type,
        )
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.cat_embedding_dim = cat_embedding_dim
        self.categorical_features = categorical_features
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.loss_params = loss_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
