from braian import BrainHemisphere, BrainData
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def data1() -> pd.Series:
    return pd.Series(data=[10, 5, 1],
                     index=["root", "Isocortex", "RE"])

@pytest.fixture
def data1() -> pd.Series:
    return pd.Series(data=[1, 2, 2],
                     index=["a", "b", "c"])

@pytest.fixture
def bd1_b(request) -> BrainData:
    data: pd.DataFrame = request.getfixturevalue("data1")
    return BrainData(data, name="bd1_b", metric="m", units="cFos",
                     hemisphere=BrainHemisphere.MERGED, ontology="atlas", check=False)

@pytest.fixture
def bd1_r(request) -> BrainData:
    data: pd.DataFrame = request.getfixturevalue("data1")
    return BrainData(data, name="bd1_r", metric="m", units="cFos",
                     hemisphere=BrainHemisphere.RIGHT, ontology="atlas", check=False)

@pytest.fixture
def bd1_l(request) -> BrainData:
    data: pd.DataFrame = request.getfixturevalue("data1")
    return BrainData(data, name="bd1_l", metric="m", units="cFos",
                     hemisphere=BrainHemisphere.LEFT, ontology="atlas", check=False)

@pytest.mark.parametrize(
    "string1, string2, same_units, same_hemisphere, should_raise",
    [
        ("left", "right", False, False, False),
        ("left", "right", False, True, True),
        ("left", "right", True, False, True),
        ("left", "right", True, True, True),
        ("left", "left", False, False, False),
        ("left", "left", False, True, False),
        ("left", "left", True, False, False),
        ("left", "left", True, True, False),
    ]
)
def test_reduce_raises(string1: str, string2: str,
                       same_units: bool, same_hemisphere: bool,
                       should_raise: bool, request):
    data1: pd.Series = request.getfixturevalue("data1")
    if string1 != string2:
        bd1 = BrainData(data1, name="bd1", metric="metric1", units=string1, hemisphere=BrainHemisphere(string1), ontology="atlas", check=False)
        bd2 = BrainData(data1, name="bd1", metric="metric2", units=string2, hemisphere=BrainHemisphere(string2), ontology="atlas", check=False)
        with pytest.raises(ValueError):
            BrainData.reduce(bd1, bd2, op=pd.DataFrame.mean, same_units=False, same_hemisphere=False)
    # same metric
    bd3 = BrainData(data1, name="bd1", metric="metric", units=string1, hemisphere=BrainHemisphere(string1), ontology="atlas", check=False)
    bd4 = BrainData(data1, name="bd1", metric="metric", units=string2, hemisphere=BrainHemisphere(string2), ontology="atlas", check=False)
    if should_raise:
        with pytest.raises(ValueError):
            BrainData.reduce(bd3, bd4, op=pd.DataFrame.mean, same_units=same_units, same_hemisphere=same_hemisphere)
    else:
        redux = BrainData.reduce(bd3, bd4, op=pd.DataFrame.mean, same_units=same_units, same_hemisphere=same_hemisphere)
        assert redux.data.values == data1.values