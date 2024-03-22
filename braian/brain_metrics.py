# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
from enum import Enum, auto

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
    SUM = auto()
    MEAN = auto()
    CVAR = auto()
    STD = auto()
    DENSITY = auto()
    PERCENTAGE = auto()
    RELATIVE_DENSITY = auto()
    OVERLAPPING = auto()
    JACCARD_INDEX = auto()
    SIMILARITY_INDEX = auto()
    OVERLAP_COEFFICIENT = auto()
    CHANCE_LEVEL = auto()
    DENSITY_DIFFERENCE = auto()
    FOLD_CHANGE = auto()
    DIFF_CHANGE = auto()

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
            case "jaccard" | "jaccard_index" | "jaccard index":
                return BrainMetrics.JACCARD_INDEX
            case "similarity" | "similarity index" | "similarity_index" | "similarityindex" | "sim":
                return BrainMetrics.SIMILARITY_INDEX
            case "szymkiewicz_simpson" | "overlap_coeff" | "overlap_coefficient" | "overlap coefficient":
                return BrainMetrics.OVERLAP_COEFFICIENT
            case "chance" | "chance_level" | "chance level":
                return BrainMetrics.CHANCE_LEVEL
            case "ddiff" | "density_difference" | "density difference":
                return BrainMetrics.DENSITY_DIFFERENCE
            case "fold_change" | "fold change":
                return BrainMetrics.FOLD_CHANGE
            case "diff_change" | "diff change":
                return BrainMetrics.DIFF_CHANGE
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
            case BrainMetrics.JACCARD_INDEX:
                return brain.markers_jaccard_index(*args, **kwargs)
            case BrainMetrics.SIMILARITY_INDEX:
                return brain.markers_similarity_index(*args, **kwargs)
            case BrainMetrics.OVERLAP_COEFFICIENT:
                return brain.markers_overlap_coefficient(*args, **kwargs)
            case BrainMetrics.CHANCE_LEVEL:
                return brain.markers_chance_level(*args, **kwargs)
            case BrainMetrics.DENSITY_DIFFERENCE:
                return brain.markers_difference(*args, **kwargs)
            case BrainMetrics.FOLD_CHANGE:
                return brain.fold_change(*args, **kwargs)
            case BrainMetrics.DIFF_CHANGE:
                return brain.diff_change(*args, **kwargs)
            case _:
                raise ValueError(f"{self} is meant for BrainSlices reduction, not for AnimalBrain analysis")