"""字符串内容画像与 JSON 识别测试。"""

import pandas as pd

from hscredit.database.type_inference import (
    profile_string_series,
    resolve_bounded_string_length,
)


def test_json_requires_every_non_null_value_to_be_object_or_array():
    profile = profile_string_series(pd.Series(['{"id": 1}', "[1, 2]", None], dtype="object"))

    assert profile.all_strings is True
    assert profile.all_json_documents is True


def test_json_scalar_and_mixed_plain_text_are_not_json_columns():
    scalar_profile = profile_string_series(pd.Series(["123", "true", '"text"'], dtype="object"))
    mixed_profile = profile_string_series(pd.Series(['{"id": 1}', "普通文本"], dtype="object"))

    assert scalar_profile.all_json_documents is False
    assert mixed_profile.all_json_documents is False


def test_string_profile_tracks_characters_and_utf8_bytes_independently():
    profile = profile_string_series(pd.Series(["衡枢", "abc"], dtype="object"))

    assert profile.max_characters == 3
    assert profile.max_utf8_bytes == 6


def test_non_string_object_values_disable_json_inference():
    profile = profile_string_series(pd.Series(['{"id": 1}', {"id": 2}], dtype="object"))

    assert profile.all_strings is False
    assert profile.all_json_documents is False


def test_bounded_length_adds_headroom_and_uses_stable_buckets():
    assert resolve_bounded_string_length(3, maximum=255) == 16
    assert resolve_bounded_string_length(50, maximum=255) == 64
    assert resolve_bounded_string_length(200, maximum=255) == 255
