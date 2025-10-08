from braian import BrainHemisphere, BrainSlice, MissingQuantificationError
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def slice_df() -> pd.DataFrame:
    return pd.DataFrame(dict(
        acronym=["root", "Isocortex", "RE"],
        hemisphere=[BrainHemisphere.LEFT.value]*3,
        area=[10, 5, 1],
        cFos=[10, 5, 1]))

@pytest.fixture
def cfos_units() -> dict:
    return dict(area="mm2", cFos="cFos", unknown_quantification="invalid units")

@pytest.fixture
def slice1(slice_df: pd.DataFrame, request) -> BrainSlice:
    units: pd.DataFrame = request.getfixturevalue("cfos_units")
    return BrainSlice(slice_df, units=units, animal="animal1", name="slice1", ontology="atlas", check=False)

@pytest.mark.parametrize(
    "hems, is_split",
    [
        ([BrainHemisphere.LEFT, BrainHemisphere.RIGHT, BrainHemisphere.LEFT], True),
        ([BrainHemisphere.LEFT]*3, True),
        ([BrainHemisphere.RIGHT]*3, True),
        ([BrainHemisphere.BOTH.value]*3, False)
    ]
)
def test_constructor_is_split(hems: list[BrainHemisphere], is_split, request):
    slice_df: pd.DataFrame = request.getfixturevalue("slice_df")
    slice_df["hemisphere"] = hems
    s = BrainSlice(slice_df, units={"area": "mm2", "cFos": "cFos"}, animal="animal1", name="slice1", ontology="atlas", check=False)
    assert s.is_split == is_split
    assert s.markers == ["cFos"]

@pytest.mark.parametrize(
    "region, hem, expected",
    [
        ("root", BrainHemisphere.BOTH, [10]),
        ("root", BrainHemisphere.LEFT, [10]),
        ("root", BrainHemisphere.RIGHT, []),
        ("NOT_FOUND", BrainHemisphere.BOTH, []),
        ("NOT_FOUND", BrainHemisphere.LEFT, []),
        ("NOT_FOUND", BrainHemisphere.RIGHT, []),
    ]
)
def test_regions(region, hem, expected, request):
    slice1: BrainSlice = request.getfixturevalue("slice1")
    assert list(slice1.region(region, metric="cFos", hemisphere=hem)) == expected

@pytest.mark.parametrize(
    "df, units, drop",
    [
        ("slice_df", "cfos_units", "acronym"),
        ("slice_df", "cfos_units", "hemisphere"),
        ("slice_df", "cfos_units", "area"),
        ("slice_df", "cfos_units", "cFos"),
    ]
)
def test_constructor_missing_column(df, units: dict, drop: str, request):
    df: pd.DataFrame = request.getfixturevalue(df)
    units: dict = request.getfixturevalue(units)
    with pytest.raises(MissingQuantificationError):
        BrainSlice(df.drop(columns=drop), units=units, animal="animal", name="slice", ontology="atlas", check=False)

@pytest.mark.parametrize(
    "df, units, drop",
    [
        ("slice_df", "cfos_units", "area"),
        ("slice_df", "cfos_units", "cFos"),
    ]
)
def test_constructor_missing_units(df: pd.DataFrame, units: dict, drop: str, request):
    df: pd.DataFrame = request.getfixturevalue(df)
    units: dict = request.getfixturevalue(units)
    with pytest.raises(ValueError):
        del units[drop]
        BrainSlice(df, units=units, animal="animal", name="slice", ontology="atlas", check=False)