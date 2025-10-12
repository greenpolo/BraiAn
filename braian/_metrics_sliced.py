import numpy as np
import pandas as pd

from enum import Enum, auto
from pandas.core.groupby import DataFrameGroupBy

__all__ = ["SlicedMetric", "SliceMetrics"]

# https://en.wikipedia.org/wiki/Coefficient_of_variation
def coefficient_variation(x: np.ndarray) -> np.float64:
    if x.ndim == 1:
        avg = x.mean()
        if len(x) > 1 and avg != 0:
            return x.std(ddof=1) / avg
        else:
            return 0
    else: # compute it for each column of the DataFrame and return a Series
        return x.apply(coefficient_variation, axis=0)

class SlicedMetric(Enum):
    r"""
    Enum of the metrics used to reduce region data from [`SlicedBrain`][braian.SlicedBrain]
    into a [`AnimalBrain`][braian.AnimalBrain].

    Attributes
    ----------
    SUM
        Computes the sum of all the sections data from the same region into a single value
    MEAN
        Computes the average $\mu$ of all the sections data from the same region into a single value
    STD
        Computes the standard deviation $\sigma$ between all the sections data from the same region into a single value
    CVAR
        Computes the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation)
        $\frac \mu \sigma$ between all the sections data from the same region into a single value
    """
    SUM = auto()
    MEAN = auto()
    STD = auto()
    CVAR = auto()

    @property
    def raw(self) -> bool:
        """
        Whether the result of the metric's reduction across multiple brain sections can be considered raw or not.\\
        A reduction produces _raw_ data if the result is an indirect quantification within a brain region
        (e.g., sum—or average—of the cell counts found in each [`BrainSlice`][braian.BrainSlice]).
        """
        return self in (SlicedMetric.SUM, SlicedMetric.MEAN)

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f'<{cls_name}.{self.name}>'

    def __str__(self):
        return self.name.lower()

    def __format__(self, format_spec: str):
        return repr(self)

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        match value.lower():
            case "sum":
                return SlicedMetric.SUM
            case "avg" | "mean":
                return SlicedMetric.MEAN
            case "variation" | "cvar" | "coefficient of variation":
                return SlicedMetric.CVAR
            case "std" | "standard deviation":
                return SlicedMetric.STD

    def __call__(self, data: pd.Series|pd.DataFrame|DataFrameGroupBy):
        match self:
            case SlicedMetric.SUM:
                return data.sum()
            case SlicedMetric.MEAN:
                return data.mean()
            case SlicedMetric.STD:
                return data.std(ddof=1)
            case SlicedMetric.CVAR:
                if isinstance(data, DataFrameGroupBy):
                    return data.apply(coefficient_variation)
                else:
                    return coefficient_variation(data)
            case _:
                raise ValueError(f"{self} does not support BrainSlices reductions")

SliceMetrics = SlicedMetric