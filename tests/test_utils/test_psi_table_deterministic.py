import numpy as np

from hscredit.core.metrics import psi_table


def test_psi_table_supervised_methods_are_deterministic():
    rng = np.random.RandomState(42)
    expected = np.r_[rng.normal(0, 1, 300), rng.normal(2, 0.5, 80)]
    actual = np.r_[rng.normal(0.4, 1.1, 300), rng.normal(1.6, 0.7, 80)]

    first = psi_table(expected, actual, method="best_iv", max_n_bins=5)
    second = psi_table(expected, actual, method="best_iv", max_n_bins=5)

    assert first["分箱"].astype(str).tolist() == second["分箱"].astype(str).tolist()
    np.testing.assert_allclose(first["PSI贡献"].to_numpy(), second["PSI贡献"].to_numpy())

