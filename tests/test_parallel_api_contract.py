"""估计器统一并行 API 契约测试。"""

import inspect

import pandas as pd
import pytest
from sklearn.base import clone

from hscredit.core.binning.base import BaseBinning
from hscredit.core.encoders.base import BaseEncoder
from hscredit.core.selectors.base import BaseFeatureSelector
from hscredit.exceptions import ValidationError
from hscredit.report.mining.base import BaseRuleMiner
from hscredit.utils.parallel import ParallelizableMixin


class _CloneableEncoder(BaseEncoder):
    """仅用于验证基类 sklearn 生命周期的最小编码器。"""

    def _fit(self, X, y=None):
        self.mapping_ = {}

    def _transform(self, X, y=None):
        return X


class _ExecutableBinning(BaseBinning):
    """仅用于验证基类特征任务委托的最小分箱器。"""

    def _fit_feature(self, feature, x, y):
        self.seen_.append(feature)

    def fit(self, X, y=None):
        self.seen_ = []
        self._fit_features(X, pd.Series(y, index=X.index), "_fit_feature")
        return self

    def transform(self, X, metric="indices"):
        return X


@pytest.mark.parametrize(
    "cls",
    [BaseBinning, BaseEncoder, BaseFeatureSelector, BaseRuleMiner],
)
def test_parallel_base_signatures_expose_common_parameters(cls):
    """四个估计器基类必须公开同名、同默认值的并行参数。"""
    params = inspect.signature(cls.__init__).parameters

    assert params["n_jobs"].default == -1
    assert params["parallel_backend"].default is None
    assert params["parallel_config"].default is None


@pytest.mark.parametrize(
    "cls",
    [BaseBinning, BaseEncoder, BaseFeatureSelector, BaseRuleMiner],
)
def test_parallel_base_classes_share_parallelizable_mixin(cls):
    """基类不得各自维护独立执行器。"""
    assert issubclass(cls, ParallelizableMixin)


def test_base_constructor_preserves_parallel_parameters_without_resolution():
    """构造器必须原样保存比例预算和调用者配置。"""
    config = {"batch_size": 8}

    encoder = _CloneableEncoder(
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )

    assert encoder.n_jobs == 0.5
    assert encoder.parallel_backend == "threading"
    assert encoder.parallel_config is config


def test_encoder_clone_preserves_parallel_config_identity_contract():
    """sklearn clone 应保留配置值，并复制可变配置对象。"""
    config = {"batch_size": 8}
    encoder = _CloneableEncoder(
        cols=["类别"],
        n_jobs=0.5,
        parallel_backend="threading",
        parallel_config=config,
    )

    cloned = clone(encoder)

    assert cloned.get_params()["parallel_config"] == config
    assert cloned.parallel_config is not config
    assert cloned.get_params()["n_jobs"] == 0.5
    assert cloned.get_params()["parallel_backend"] == "threading"


def test_base_parallel_execute_preserves_order_for_threading_backend():
    """基类委托共享执行器后必须按任务提交顺序返回结果。"""
    encoder = _CloneableEncoder(n_jobs=2, parallel_backend="threading")

    result = encoder._parallel_execute(abs, [-3, -1, -2])

    assert result == [3, 1, 2]


def test_base_binning_feature_loop_uses_shared_parallel_configuration():
    """分箱基类的既有特征循环不得绕过共享配置校验。"""
    binner = _ExecutableBinning(
        n_jobs=1,
        parallel_config={"未知配置": True},
    )

    with pytest.raises(ValidationError, match="parallel_config"):
        binner.fit(pd.DataFrame({"特征A": [1, 2]}), [0, 1])


def test_base_encoder_parallel_parameters_do_not_change_dual_api():
    """新增并行状态不得改变目标列提取和透传语义。"""
    frame = pd.DataFrame({"类别": ["甲", "乙"], "FPD": [0, 1]})
    encoder = _CloneableEncoder(
        cols=["类别"],
        target="FPD",
        n_jobs=None,
        parallel_config={"batch_size": 1},
    )

    transformed = encoder.fit_transform(frame)

    pd.testing.assert_frame_equal(transformed, frame)
