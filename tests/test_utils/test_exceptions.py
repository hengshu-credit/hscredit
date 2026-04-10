import pandas as pd
import pytest

from hscredit import (
    FeatureNotFoundError,
    InputTypeError,
    InputValidationError,
    NotFittedError,
    ValidationError,
)
from hscredit.core.encoders import WOEEncoder
from hscredit.core.rules import Rule
from hscredit.core.selectors import VarianceSelector
from hscredit.core.binning import UniformBinning
from hscredit.utils import check_xy_inputs, convert_to_dataframe, load_pickle


def test_input_utils_raise_unified_exceptions():
    with pytest.raises(InputValidationError):
        convert_to_dataframe(None)

    with pytest.raises(FeatureNotFoundError):
        check_xy_inputs(pd.DataFrame({"a": [1, 2, 3]}), y=None, target="target")


def test_io_raises_validation_error_for_unknown_engine(tmp_path):
    dummy_file = tmp_path / "dummy.pkl"
    dummy_file.write_bytes(b"not-used")

    with pytest.raises(ValidationError):
        load_pickle(dummy_file, engine="unknown")


def test_common_estimators_raise_not_fitted_error():
    with pytest.raises(NotFittedError):
        UniformBinning().get_bin_table("age")

    with pytest.raises(NotFittedError):
        VarianceSelector().transform(pd.DataFrame({"a": [1, 2, 3]}))

    with pytest.raises(NotFittedError):
        WOEEncoder()["feature_a"]


def test_rule_raises_unified_type_and_feature_errors():
    rule = Rule("age > 18")

    with pytest.raises(InputTypeError):
        rule.predict([1, 2, 3])

    with pytest.raises(FeatureNotFoundError):
        rule.predict(pd.DataFrame({"income": [100, 200, 300]}))