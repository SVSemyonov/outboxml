import pytest

from outboxml.core import enums


@pytest.mark.parametrize("enum_cls", [obj for obj in vars(enums).values()
                                      if isinstance(obj, type) and issubclass(obj, enums.Enum) and obj is not enums.Enum])
def test_enum_values_are_ascii(enum_cls):
    for member in enum_cls:
        if isinstance(member.value, str):
            assert member.value.isascii(), f"{enum_cls.__name__}.{member.name} = {member.value!r}"


def test_feature_types_for_selection_categorical():
    assert enums.FeatureTypesForSelection.categorical == "CATEGORICAL"
