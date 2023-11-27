import numpy as np
from enum import Enum

# decorates a function [xs: list->float] so that if the list is too short, it returns NaN
def min_count(fun, min, **kwargs):
    def nan_if_less(xs: list) -> float:
        if len(xs) >= min:
            return fun(xs, **kwargs)
        else:
            return np.NaN
    return nan_if_less

# https://en.wikipedia.org/wiki/Coefficient_of_variation
def coefficient_variation(x) -> np.float64:
    if x.ndim == 1:
        avg = x.mean()
        if len(x) > 1 and avg != 0:
            return x.std(ddof=1) / avg
        else:
            return 0
    else: # compute it for each column of the DataFrame and return a Series
        return x.apply(coefficient_variation, axis=0)   

class BrainMetrics(Enum):
    SUM = 1
    MEAN = 2
    CVAR = 3
    STD = 4
    DENSITY = 5
    PERCENTAGE = 6
    RELATIVE_DENSITY = 7
    OVERLAPPING = 8
    SIMILARITY_INDEX = 9
    DENSITY_DIFFERENCE = 10

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
                return BrainMetrics.SUM
            case "avg" | "mean":
                return BrainMetrics.MEAN
            case "variation" | "cvar" | "coefficient of variation":
                return BrainMetrics.CVAR
            case "std" | "standard deviation":
                return BrainMetrics.STD
            case "density" | "dens" | "d":
                return BrainMetrics.DENSITY
            case "percentage" | "perc" | "%" | "p":
                return BrainMetrics.PERCENTAGE
            case "relative_density" | "relativedensity" | "relative density" | "rd":
                return BrainMetrics.RELATIVE_DENSITY
            case "overlaps" | "overlap" | "overlapping":
                return BrainMetrics.OVERLAPPING
            case "similarity" | "similarity index" | "similarity_index" | "similarityindex" | "sim":
                return BrainMetrics.SIMILARITY_INDEX
            case "ddiff" | "density_difference" | "density difference":
                return BrainMetrics.DENSITY_DIFFERENCE
            case _:
                return None

    def fold_slices(self, min_slices: int):
        match self:
            case BrainMetrics.MEAN:
                return min_count(np.mean, min_slices, axis=0)
            case BrainMetrics.STD:
                return min_count(np.std, min_slices, ddof=1, axis=0)
            case BrainMetrics.CVAR:
                return min_count(coefficient_variation, min_slices)
            case _:
                raise ValueError(f"{self} does not support BrainSlices reductions")

    def analyse(self, brain, *args, **kwargs): # AnimalBrain -> AnimalBrain:
        match self:
            case BrainMetrics.DENSITY:
                return brain.density(*args, **kwargs)
            case BrainMetrics.PERCENTAGE:
                return brain.percentage(*args, **kwargs)
            case BrainMetrics.RELATIVE_DENSITY:
                return brain.relative_density(*args, **kwargs)
            case BrainMetrics.OVERLAPPING:
                return brain.markers_overlap(*args, **kwargs)
            case BrainMetrics.SIMILARITY_INDEX:
                return brain.markers_similarity_index(*args, **kwargs)
            case BrainMetrics.DENSITY_DIFFERENCE:
                return brain.markers_difference(*args, **kwargs)
            case _:
                raise ValueError(f"{self} is meant for BrainSlices reduction, not for AnimalBrain analysis")