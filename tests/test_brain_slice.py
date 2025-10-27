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
def slice1_split(slice_df: pd.DataFrame, request) -> BrainSlice:
    units: pd.DataFrame = request.getfixturevalue("cfos_units")
    return BrainSlice(slice_df, units=units, animal="animal1", name="slice1_split", ontology="atlas", check=False)

@pytest.fixture
def slice1_merged(slice1_split: BrainSlice) -> BrainSlice:
    return slice1_split.merge_hemispheres()

@pytest.mark.parametrize(
    "hems, is_split",
    [
        ([BrainHemisphere.LEFT, BrainHemisphere.RIGHT, BrainHemisphere.LEFT], True),
        ([BrainHemisphere.LEFT]*3, True),
        ([BrainHemisphere.RIGHT]*3, True),
        ([BrainHemisphere.MERGED.value]*3, False)
    ]
)
def test_constructor_is_split(hems: list[BrainHemisphere], is_split, request):
    slice_df: pd.DataFrame = request.getfixturevalue("slice_df")
    slice_df["hemisphere"] = hems
    s = BrainSlice(slice_df, units={"area": "mm2", "cFos": "cFos"}, animal="animal1", name="slice1_split", ontology="atlas", check=False)
    assert s.is_split == is_split
    assert s.markers == ["cFos"]

empty_regions_merged = pd.DataFrame(columns=["cFos"], dtype=int)
empty_regions_split = pd.DataFrame(columns=["hemisphere", "cFos"], dtype=int)

@pytest.mark.parametrize(
    "slice, region, hem, expected",
    [
        ("slice1_split", "root", BrainHemisphere.MERGED, pd.DataFrame({"hemisphere": [1], "cFos": 10})),
        ("slice1_merged", "root", BrainHemisphere.MERGED, 10),
        ("slice1_split", "root", BrainHemisphere.LEFT, 10),
        ("slice1_merged", "root", BrainHemisphere.LEFT, empty_regions_merged),
        ("slice1_split", "root", BrainHemisphere.RIGHT, empty_regions_split),
        ("slice1_merged", "root", BrainHemisphere.RIGHT, empty_regions_merged),
        ("slice1_split", "NOT_FOUND", BrainHemisphere.MERGED, empty_regions_split),
        ("slice1_merged", "NOT_FOUND", BrainHemisphere.MERGED, empty_regions_merged),
        ("slice1_split", "NOT_FOUND", BrainHemisphere.LEFT, empty_regions_split),
        ("slice1_merged", "NOT_FOUND", BrainHemisphere.LEFT, empty_regions_merged),
        ("slice1_split", "NOT_FOUND", BrainHemisphere.RIGHT, empty_regions_split),
        ("slice1_merged", "NOT_FOUND", BrainHemisphere.RIGHT, empty_regions_merged),
    ]
)
def test_regions(slice, region, hem, expected, request):
    slice: BrainSlice = request.getfixturevalue(slice)
    result = slice.region(region, metric="cFos", hemisphere=hem)
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(expected, result)
    else:
        assert expected == result

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