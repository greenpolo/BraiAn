import itertools
import numpy as np
import pandas as pd
import re
from collections.abc import Collection, Iterable
from typing import Self

from braian.deflector import deflect
from braian.brain_hierarchy import AllenBrainHierarchy

__all__ = ["BrainData"]

class UnkownBrainRegionsError(Exception):
    def __init__(self, unknown_regions: Iterable[str]):
        super().__init__(f"The following regions are unknown to the given brain ontology: '"+"', '".join(unknown_regions)+"'")

def extract_acronym(region_class):
    '''
    This function extracts the region acronym from a QuPath's PathClass assigned by ABBA
    Example: "Left: AVA" becomes "AVA".
    '''
    acronym = re.compile("[Left|Right]: (.+)").findall(region_class)
    if len(acronym) == 0:
        # the region's class didn't distinguish between left|right hemispheres 
        return str(region_class)
    return acronym[0]

def get_hemisphere_name(hem: str):
    match hem.lower():
        case "left" | "l":
            return "Left"
        case "right" | "r":
            return "Right"
        case _:
            raise ValueError(f"Unrecognised hemisphere '{hem}'!")

def is_split_left_right(index: pd.Index):
    return (index.str.startswith("Left: ", na=False) | \
            index.str.startswith("Right: ", na=False)).all()

def split_index(regions: list[str]) -> list[str]:
    return [": ".join(t) for t in itertools.product(("Left", "Right"), regions)]

def sort_by_ontology(data: pd.DataFrame|pd.Series, brain_ontology: AllenBrainHierarchy,
                     fill=False, fill_value=np.nan) -> pd.DataFrame|pd.Series:
        all_regions = brain_ontology.list_all_subregions("root", mode="depth")
        if is_split_left_right(data.index):
            all_regions = split_index(all_regions)
        if len(unknown_regions:=data.index[~data.index.isin(all_regions)]) > 0:
            raise UnkownBrainRegionsError(unknown_regions)
        if not fill:
            all_regions = np.array(all_regions)
            all_regions = all_regions[np.isin(all_regions, data.index)]
        # NOTE: if fill_value=np.nan -> converts dtype to float
        return data.reindex(all_regions, copy=False, fill_value=fill_value)

class BrainData(metaclass=deflect(on_attribute="data", arithmetics=True, container=True)):
    @staticmethod
    def merge(first: Self, second: Self, *others: Self, op=pd.DataFrame.mean, name=None, op_name=None,
              same_metrics=True, same_units=True, **kwargs) -> Self:
        assert first.metric == second.metric and all([first.metric == other.metric for other in others]),\
            f"Merging must be done between BrainData of the same metric, instead got {[first.metric, second.metric, *[other.metric for other in others]]}!"
        if same_units:
            assert first.units == second.units and all([first.units == other.units for other in others]),\
                f"Merging must be done between BrainData of the same units, {[first.units, second.units, *[other.units for other in others]]}!"
        if name is None:
            name = ":".join([first.data_name, second.data_name, *[other.data_name for other in others]])
        if op_name is None:
            op_name = op.__name__
        data = op(pd.concat([first.data, second.data, *[other.data for other in others]], axis=1), axis=1, **kwargs)
        return BrainData(data, name, f"{first.metric}:{op_name} (n={len(others)+2})", first.units) 

    @staticmethod
    def mean(*args, **kwargs) -> Self:
        return BrainData.merge(*args, op=pd.DataFrame.mean, same_metrics=True, same_units=True, **kwargs)

    @staticmethod
    def minimum(*args, **kwargs) -> Self:
        return BrainData.merge(*args, op=pd.DataFrame.min, same_metrics=True, same_units=False, **kwargs)

    @staticmethod
    def maximum(*args, **kwargs) -> Self:
        return BrainData.merge(*args, op=pd.DataFrame.max, same_metrics=True, same_units=False, **kwargs)

    RAW_TYPE: str = "raw"

    def __init__(self, data: pd.Series, name: str, metric: str, units: str,
                 brain_ontology:AllenBrainHierarchy|None=None, fill_nan=False) -> None:
        self.data = data.copy()
        self.is_split = is_split_left_right(self.data.index)
        self.data_name = str(name) # data_name
        self.data.name = self.data_name
        self.metric = str(metric)
        if units is not None:
            self.units = str(units)
        else:
            self.units = ""
            print(f"WARNING: {self} has no units")
        if brain_ontology is not None:
            self.sort_by_ontology(brain_ontology, fill_nan, inplace=True)
    
    def __str__(self) -> str:
        return f"BrainData(name={self.data_name}, metric={self.metric})"
    
    def sort_by_ontology(self, brain_ontology: AllenBrainHierarchy,
                          fill_nan=False, inplace=False) -> Self:
        data = sort_by_ontology(self.data, brain_ontology, fill=fill_nan, fill_value=np.nan)
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self
    
    def root(self, hemisphere=None) -> float:
        acronym = "root"
        if self.is_split:
            if hemisphere is None:
                raise ValueError(f"You have to specify the hemisphere of '{acronym}' you want!")
            acronym = f"{get_hemisphere_name(hemisphere)}: {acronym}"
        if acronym not in self.data:
            raise ValueError(f"No data for '{acronym}' in {self}!")
        return self.data[acronym]

    def min(self) -> float:
        return self.data[self.data != np.inf].min()

    def max(self) -> float:
        return self.data[self.data != np.inf].max()

    def remove_region(self, region: str, *regions, inplace=False, fill_nan=False) -> Self:
        data = self.data.copy() if not inplace else self.data
        regions = [region, *regions]
        if fill_nan:
            data[regions] = np.nan
        else:
            data = data[data.index.isin(regions)]
        return self if inplace else BrainData(data, name=self.data_name, metric=self.metric, units=self.units)

    @property
    def regions(self) -> list[str]:
        return list(self.data.index)

    def set_regions(self, brain_regions: list[str], brain_ontology: AllenBrainHierarchy,
                    fill=np.nan, overwrite=False, inplace=False) -> Self:
        if isinstance(fill, Collection):
            brain_regions = np.asarray(brain_regions)
            if len(fill) != len(brain_regions):
                raise ValueError("'fill' argument requires a collection of the same length as 'brain_regions'")
        else:
            assert isinstance(fill, (int, float, np.number)), "'fill' argument must either be a collection or a number"
            fill = itertools.repeat(fill)
        if not all(are_regions := brain_ontology.are_regions(brain_regions, "acronym")):
            unknown_regions = brain_regions[~are_regions]
            raise ValueError("Unrecognised regions in the given ontology: "+unknown_regions)
        data = self.data.copy() if not inplace else self.data
        for region,value in zip(brain_regions, fill):
            if not overwrite and region in data.index:
                continue
            data[region] = value
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self

    def select_from_list(self, brain_regions: list[str], fill_nan=False, inplace=False) -> Self:
        if not (unknown_regions:=np.isin(brain_regions, self.data.index)).all():
            unknown_regions = np.array(brain_regions)[~unknown_regions]
            raise ValueError(f"Can't find some regions in {self}: '"+"', '".join(unknown_regions)+"'!")
        if fill_nan:
            data = self.data.reindex(index=brain_regions, fill_value=np.nan)
        else:
            data = self.data[self.data.index.isin(brain_regions)]
        if not inplace:
            return BrainData(data, self.data_name, self.metric, self.units)
        else:
            self.data = data
            return self
    
    def select_from_ontology(self, brain_ontology: AllenBrainHierarchy,
                              *args, **kwargs) -> Self:
        selected_allen_regions = brain_ontology.get_selected_regions()
        selectable_regions = set(self.data.index).intersection(set(selected_allen_regions))
        return self.select_from_list(list(selectable_regions), *args, **kwargs)
    
    def merge_hemispheres(self) -> Self:
        if self.metric not in ("sum",):
            raise ValueError(f"Cannot properly merge '{self.metric}' BrainData from left/right hemispheres into a single region!")
        corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in self.data.index]
        data = self.data.groupby(corresponding_region).sum(min_count=1)
        return BrainData(data, name=self.data_name, metric=self.metric, units=self.units)