"""真实训练验证各框架的目标、指标及离线计算使用同一数值口径。"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.model_selection import train_test_split

from hscredit.core.models.losses import FocalLoss, WeightedBCELoss, OrdinalRankLoss
from hscredit.core.models.losses.adapters import TabNetLossAdapter


@pytest.fixture(scope="module")
def loans():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples" / "hscredit_yyp.xlsx"
    if not path.exists():
        pytest.skip("缺少 examples/hscredit_yyp.xlsx")
    columns = ["衡枢鉴真分老客版", "近六个月非银多头机构数", "青云24", "FPD"]
    frame = pd.read_excel(path, usecols=columns).apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().iloc[:500]
    return train_test_split(
        frame[columns[:3]].to_numpy(), frame.FPD.to_numpy(), test_size=0.3, stratify=frame.FPD, random_state=24
    )


@pytest.mark.parametrize("loss", [FocalLoss(alpha=0.7), WeightedBCELoss(pos_weight=2)])
def test_native_xgboost_loss_matches_offline(loans, loss):
    xgb = pytest.importorskip("xgboost")
    x, xv, y, yv = loans
    weight = np.linspace(0.5, 2.0, len(yv))
    train = xgb.DMatrix(x, label=y)
    valid = xgb.DMatrix(xv, label=yv, weight=weight)
    metric = loss.metric()
    history = {}
    model = xgb.train(
        {"disable_default_eval_metric": 1, "nthread": 1, "max_depth": 2, "base_score": 0.0},
        train,
        num_boost_round=4,
        obj=loss.to_xgboost(),
        custom_metric=metric.to_xgboost(raw_score=True),
        evals=[(valid, "验证")],
        evals_result=history,
        verbose_eval=False,
        maximize=metric.greater_is_better,
    )
    p = expit(model.predict(valid, output_margin=True))
    assert history["验证"][metric.name][-1] == pytest.approx(metric(yv, p, sample_weight=weight), abs=1e-6)


@pytest.mark.parametrize("loss", [FocalLoss(alpha=0.7), WeightedBCELoss(pos_weight=2)])
def test_native_lightgbm_loss_matches_offline(loans, loss):
    lgb = pytest.importorskip("lightgbm")
    x, xv, y, yv = loans
    weight = np.linspace(0.5, 2.0, len(yv))
    metric = loss.metric()
    history = {}
    model = lgb.train(
        {
            "objective": loss.to_lightgbm(api="native"),
            "metric": "None",
            "num_threads": 1,
            "verbosity": -1,
            "min_data_in_leaf": 5,
        },
        lgb.Dataset(x, label=y),
        num_boost_round=4,
        valid_sets=[lgb.Dataset(xv, label=yv, weight=weight)],
        valid_names=["验证"],
        feval=metric.to_lightgbm(api="native", raw_score=True),
        callbacks=[lgb.record_evaluation(history)],
    )
    p = expit(model.predict(xv, raw_score=True))
    assert history["验证"][metric.name][-1] == pytest.approx(metric(yv, p, sample_weight=weight))


def test_native_catboost_metric_matches_offline(loans):
    cb = pytest.importorskip("catboost")
    x, xv, y, yv = loans
    weight = np.linspace(0.5, 2.0, len(yv))
    loss = FocalLoss(alpha=0.7)
    model = cb.CatBoostClassifier(
        loss_function=loss.to_catboost(),
        eval_metric=loss.metric().to_catboost(),
        iterations=4,
        depth=2,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x, y, eval_set=cb.Pool(xv, label=yv, weight=weight))
    p = model.predict_proba(xv)[:, 1]
    history = model.get_evals_result()["validation"]
    assert history["CatBoostMetric"][model.best_iteration_] == pytest.approx(loss.evaluate(yv, p, sample_weight=weight))


def test_ngboost_monitors_its_actual_custom_loss(loans):
    ngb = pytest.importorskip("ngboost")
    x, xv, y, yv = loans
    loss = FocalLoss(alpha=0.7)
    model = ngb.NGBClassifier(**loss.ngboost_params(), n_estimators=4, verbose=False, random_state=42)
    model.fit(x, y.astype(int))
    p = model.predict_proba(xv)[:, 1]
    assert model.score(xv, yv.astype(int)) == pytest.approx(loss.evaluate(yv, p))


def test_frameworks_reject_batch_unsafe_loss():
    loss = OrdinalRankLoss()
    with pytest.raises(ValueError, match="分批"):
        loss.to_catboost()
    with pytest.raises(ValueError, match="逐样本"):
        loss.to_ngboost()


def test_tabnet_focal_backward_matches_finite_difference():
    # Windows 下 Torch 与多个树框架的原生 DLL 可能产生加载顺序冲突；
    # 在独立进程验证真实 autograd，不将依赖初始化失败当作通过或跳过。
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("torch") is None:
        pytest.skip("未安装PyTorch")
    code = """
import hscredit
import torch
import numpy as np
from hscredit.core.models.losses import FocalLoss
from hscredit.core.models.losses.adapters import TabNetLossAdapter
loss = FocalLoss(alpha=.7)
target = torch.tensor([0., 1., 0.], dtype=torch.float64)
margin = torch.tensor([-.5, .9, .4], dtype=torch.float64, requires_grad=True)
value = TabNetLossAdapter(loss).loss_fn()(margin, target)
value.backward()
numerical = []
for i in range(3):
    upper = margin.detach().numpy().copy()
    lower = upper.copy()
    upper[i] += 1e-5
    lower[i] -= 1e-5
    numerical.append((loss.evaluate(target.numpy(), upper, raw_score=True) - loss.evaluate(target.numpy(), lower, raw_score=True)) / 2e-5)
np.testing.assert_allclose(margin.grad.numpy(), numerical, rtol=1e-5)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
