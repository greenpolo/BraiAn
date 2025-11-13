import braian
import braian.stats as bas
import pandas as pd
import pytest

@pytest.skip(reason="incomplete", allow_module_level=True)
def test_fold_change():
    density0 = braian.AnimalGroup(...)
    density1 = braian.AnimalGroup(...)
    region = "XII"
    xii_g1 = density1.mean["cFos","l"][region]
    pd.Series([b["cFos","l"][region]/xii_g1 for b in density0]).mean() == density0.apply(lambda b: bas.fold_change(b,density1)).mean()["cFos","l"][region]