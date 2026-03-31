import numpy as np
import pandas as pd
import pytest

from hscredit.core.encoders import OrdinalEncoder, WOEEncoder


def test_encoder_getitem_returns_feature_mapping_ordinal():
    df = pd.DataFrame({
        'city': ['A', 'B', 'A', 'C', np.nan],
        'target': [0, 1, 0, 1, 0]
    })

    enc = OrdinalEncoder(cols=['city'])
    enc.fit(df[['city']])

    mapping = enc['city']
    assert isinstance(mapping, dict)
    assert 'A' in mapping


def test_encoder_getitem_returns_feature_mapping_woe():
    df = pd.DataFrame({
        'city': ['A', 'B', 'A', 'C', 'B', np.nan],
        'target': [0, 1, 0, 1, 0, 1]
    })

    enc = WOEEncoder(cols=['city'])
    enc.fit(df[['city']], df['target'])

    mapping = enc['city']
    assert isinstance(mapping, dict)
    assert 'A' in mapping


def test_encoder_getitem_raises_before_fit():
    enc = OrdinalEncoder(cols=['city'])
    with pytest.raises(ValueError):
        _ = enc['city']


def test_encoder_getitem_raises_key_error_for_unknown_feature():
    df = pd.DataFrame({'city': ['A', 'B', 'A']})
    enc = OrdinalEncoder(cols=['city'])
    enc.fit(df)

    with pytest.raises(KeyError):
        _ = enc['unknown_feature']
