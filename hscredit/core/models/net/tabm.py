"""TabM 参数共享集成网络，支持二分类、回归和成员表示。"""

from ._torch_model import TorchTabularModel


class TabM(TorchTabularModel):
    """封装论文作者的 TabM，训练平均成员损失，分类预测平均成员概率。

    **参数**

    :param objective: binary（标签 0/1）或 regression（单目标回归）。
    :param d_block: 成员隐藏层及 embedding 宽度，默认 512。
    :param n_blocks: 网络深度，默认 3。
    :param k: 集成成员数，默认 32；默认搜索固定此参数，建议分开比较不同 k。
    :param arch_type: tabm、tabm-mini 或 tabm-packed，默认 tabm。
    :param dropout: 网络丢弃概率，默认 0.1。
    :param num_embeddings: None 使用标准化数值；linear_relu 或 periodic 使用可学习数值表示。
    :param num_embedding_params: 数值表示参数，如 d_embedding；periodic 默认 lite=True。
    :param categorical_features: 类别列名/列位置列表，None 自动识别；官方 TabM 对其使用独热输入。
    :param learning_rate: 优化器学习率，默认 0.002。
    :param weight_decay: 权重衰减，默认 0.0003。
    :param max_epochs: 最大训练轮数，默认 100。
    :param batch_size: 训练与推理批量，默认 256。
    :param early_stopping_rounds: 首个验证集的集成预测损失连续未改善轮数；None 关闭早停。
    :param validation_fraction: 无显式 eval_set 时的内部留出比例，默认 0.2。
    :param device: auto、cpu 或 cuda[:设备编号]。
    :param target: 可选目标列名；支持 fit(df)，显式 y 优先。
    :param eval_metric: 最终验证指标；早停按配置的损失计算。
    :param random_state: 局部随机种子。
    :param verbose: 是否输出中文训练日志。
    :param scorecard_params: 二分类的概率评分卡配置。
    :param history_policy: 训练历史策略，默认 summary。
    :param max_history: 最大训练历史数量，默认 20。
    :param weight_type: cost 或 frequency，沿用统一模型的样本权重语义。
    :param loss: 名称、BaseLoss、PyTorch 损失类/实例或函数；自定义函数接收展平的原始预测和标签，返回逐行损失。
    :param loss_params: 损失名称/类/函数参数。训练时对每个成员分别求损失后平均。
    :param optimizer: adamw（默认）、adam、sgd、rmsprop、adagrad 或 PyTorch 优化器类。
    :param optimizer_params: 优化器额外参数；使用 learning_rate 和 weight_decay 配置学习率/衰减。

    **属性**

    :ivar embedding_dim_: 每个成员的表示维度。
    :ivar optimizer_: 当前网络实际使用的优化器。
    :ivar loss_fn_: 返回逐样本损失的实际训练函数。
    :ivar n_iter_: 实际训练轮数；早停后恢复最佳网络参数。

    **参考样例**

    >>> model = TabM(target='FPD', k=16, d_block=128, random_state=42)
    >>> model.fit(train_df)
    >>> probability = model.predict_proba(test_df)
    >>> embedding = model.get_embeddings(test_df)
    >>> members = model.get_embeddings(test_df, aggregate=None)

    官方架构及训练协议：https://github.com/yandex-research/tabm
    """

    _network_kind = "TabM"

    def __init__(
        self,
        objective="binary",
        d_block=512,
        n_blocks=3,
        k=32,
        arch_type="tabm",
        dropout=0.1,
        num_embeddings=None,
        num_embedding_params=None,
        categorical_features=None,
        learning_rate=0.002,
        weight_decay=0.0003,
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
        self.k = k
        self.arch_type = arch_type
        self.dropout = dropout
        self.num_embeddings = num_embeddings
        self.num_embedding_params = num_embedding_params
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
