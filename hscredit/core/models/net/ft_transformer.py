"""特征分词器与 Transformer 组成的表格预测模型。"""

from ._torch_model import TorchTabularModel


class FTTransformer(TorchTabularModel):
    """使用数值/类别特征分词器与 CLS 表示的 FT-Transformer。

    **参数**

    :param objective: binary（二分类，标签 0/1）或 regression（单目标回归）。
    :param d_block: token 与输出 embedding 的维度，默认 128，必须能被 n_heads 整除。
    :param n_blocks: Transformer 块数，默认 3。
    :param n_heads: 注意力头数，默认 8。
    :param dropout: 注意力和前馈层丢弃概率，默认 0.1。
    :param categorical_features: 类别字段名/列位置列表；None 自动识别非数值及布尔字段。
    :param learning_rate: 优化器学习率，默认 0.0001。
    :param weight_decay: 优化器权重衰减，默认 0.00001。
    :param max_epochs: 最大轮数，默认 100。
    :param batch_size: 训练和推理批量，默认 256。
    :param early_stopping_rounds: 首个验证集损失未改善的容忍轮数；None 关闭早停。
    :param validation_fraction: 自动留出比例，默认 0.2；预处理不会使用留出行。
    :param device: auto、cpu 或 cuda[:设备编号]。
    :param target: 可选目标列；支持 fit(df)，显式 y 优先。
    :param eval_metric: 最终评估指标；早停按配置的 loss，不随此参数改变。
    :param random_state: 随机种子。
    :param verbose: 是否输出中文训练日志。
    :param scorecard_params: 仅用于二分类的概率评分卡配置。
    :param history_policy: 历史保留策略，默认 summary。
    :param max_history: 历史记录上限，默认 20。
    :param weight_type: cost 或 frequency，沿用统一模型的权重语义。
    :param loss: None 使用默认损失；支持名称、BaseLoss、PyTorch 损失类/实例和逐行张量函数。
    :param loss_params: 损失名称/类/函数的参数；可通过 loss_params__参数名搜索。
    :param optimizer: adamw（默认）、adam、sgd、rmsprop、adagrad 或 PyTorch 优化器类。
    :param optimizer_params: 优化器额外参数字典；可通过 optimizer_params__参数名搜索。

    **属性**

    :ivar embedding_dim_: CLS 经末尾归一化及激活后的表示维度。
    :ivar n_iter_: 实际轮数；早停后恢复最优网络参数。
    :ivar preprocessor_: 训练阶段拟合的预处理器，未知类别映射为 0。

    **参考样例**

    >>> model = FTTransformer(objective='regression', random_state=42)
    >>> model.fit(X_train, y_train)
    >>> prediction = model.predict(X_test)
    >>> embedding = model.transform(X_test)

    架构来源：https://github.com/yandex-research/rtdl-revisiting-models
    """

    _network_kind = "FTTransformer"

    def __init__(
        self,
        objective="binary",
        d_block=128,
        n_blocks=3,
        n_heads=8,
        dropout=0.1,
        categorical_features=None,
        learning_rate=0.0001,
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
        self.n_heads = n_heads
        self.dropout = dropout
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
