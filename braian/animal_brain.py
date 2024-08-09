import copy
import numpy as np
import os
import pandas as pd
import re

from enum import Enum, auto
from pandas.core.groupby import DataFrameGroupBy
from typing import Generator, Self

from braian.brain_hierarchy import AllenBrainHierarchy
from braian.sliced_brain import SlicedBrain, EmptyBrainError
from braian.brain_data import BrainData

__all__ = ["AnimalBrain", "SliceMetrics"]

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

class SliceMetrics(Enum):
    SUM = auto()
    MEAN = auto()
    CVAR = auto()
    STD = auto()

    @property
    def _raw(self) -> bool:
        return self in (SliceMetrics.SUM, SliceMetrics.MEAN)

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
                return SliceMetrics.SUM
            case "avg" | "mean":
                return SliceMetrics.MEAN
            case "variation" | "cvar" | "coefficient of variation":
                return SliceMetrics.CVAR
            case "std" | "standard deviation":
                return SliceMetrics.STD

    def apply(self, grouped_by_region: DataFrameGroupBy):
        match self:
            case SliceMetrics.SUM:
                return grouped_by_region.sum()
            case SliceMetrics.MEAN:
                return grouped_by_region.mean()
            case SliceMetrics.STD:
                return grouped_by_region.std(ddof=1)
            case SliceMetrics.CVAR:
                return grouped_by_region.apply(coefficient_variation)
            case _:
                raise ValueError(f"{self} does not support BrainSlices reductions")

    def __call__(self, sliced_brain: SlicedBrain, min_slices: int, densities: bool):
        all_slices = sliced_brain.concat_slices(densities=densities)
        all_slices = all_slices.groupby(all_slices.index).filter(lambda g: len(g) >= min_slices)
        raw = not densities and self._raw
        return self.apply(all_slices.groupby(all_slices.index)), raw

class AnimalBrain:
    RAW_DATA = "raw"

    @staticmethod
    def from_slices(sliced_brain: SlicedBrain,
                    mode: SliceMetrics|str=SliceMetrics.SUM, min_slices: int=0,
                    hemisphere_distinction: bool=True, densities: bool=False) -> Self:
        if not hemisphere_distinction:
            sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)

        name = sliced_brain.name
        markers = copy.copy(sliced_brain.markers)
        mode = SliceMetrics(mode)
        if len(sliced_brain.slices) < min_slices:
            raise EmptyBrainError(sliced_brain.name)
        redux, raw = mode(sliced_brain, min_slices, densities=densities)
        if redux.shape[0] == 0:
            raise EmptyBrainError(sliced_brain.name)
        metric = f"{str(mode)}_densities" if densities else str(mode)
        areas = BrainData(redux["area"], name=name, metric=metric, units="mm²")
        markers_data = {
            m: BrainData(redux[m], name=name, metric=metric, units=m)
            for m in markers
        }
        return AnimalBrain(markers_data=markers_data, areas=areas, raw=raw)

    def __init__(self, markers_data: dict[str,BrainData], areas: BrainData, raw: bool=False) -> None:
        assert len(markers_data) > 0 and areas is not None, "You must provide both a dictionary of BrainData (markers) and an additional BrainData for the areas/volumes of each region"
        self.markers = tuple(markers_data.keys())
        self.markers_data = markers_data
        self.areas = areas
        self.raw = raw
        assert all([m.data_name == self.name for m in markers_data.values()]), "All markers' BrainData must be from the same animal!"
        assert all([m.metric == self.mode for m in markers_data.values()]), "All markers' BrainData must have the same metric!"
        assert all([m.is_split == self.is_split for m in markers_data.values()]), "Markers' BrainData must either all have split hemispheres or none!"
        assert self.is_split == areas.is_split, "Markers' and areas' BrainData must either both have split hemispheres or none!"
        return

    @property
    def mode(self) -> str:
        return self.markers_data[self.markers[0]].metric

    @property
    def is_split(self) -> bool:
        return self.markers_data[self.markers[0]].is_split

    @property
    def name(self) -> str:
        return self.markers_data[self.markers[0]].data_name

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AnimalBrain(name='{self.name}', mode={self.mode}, markers={list(self.markers)})"

    def __getitem__(self, marker: str):
        return self.markers_data[marker]

    def remove_region(self, *region: str) -> None:
        for data in self.markers_data.values():
            data.remove_region(*region, inplace=True, fillnan=True)
        self.areas.remove_region(*region, inplace=True, fillnan=True)

    def remove_smaller_subregions(self, area_threshold, brain_ontology: AllenBrainHierarchy) -> None:
        small_regions = {smaller_region for small_region in self.areas.data[self.areas.data <= area_threshold].index
                                        for smaller_region in brain_ontology.list_all_subregions(small_region)}
        self.remove_region(*small_regions)

    @property
    def regions(self) -> list[str]:
        # assumes areas' and all markers' BrainData are synchronized
        return self.areas.regions

    def sort_by_ontology(self, brain_ontology: AllenBrainHierarchy,
                          fill=False, inplace=False):
        markers_data = {marker: m_data.sort_by_ontology(brain_ontology, fill=fill, inplace=inplace) for marker, m_data in self.markers_data.items()}
        areas = self.areas.sort_by_ontology(brain_ontology, fill=fill, inplace=inplace)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, areas=areas, raw=self.raw)
        else:
            return self

    def select_from_list(self, regions: list[str], fill_nan=False, inplace=False) -> Self:
        markers_data = {marker: m_data.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for marker, m_data in self.markers_data.items()}
        areas = self.areas.select_from_list(regions, fill_nan=fill_nan, inplace=inplace)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, areas=areas, raw=self.raw)
        else:
            return self

    def select_from_ontology(self, brain_ontology: AllenBrainHierarchy, fill_nan=False, *args, **kwargs) -> Self:
        selected_allen_regions = brain_ontology.get_selected_regions()
        if not fill_nan:
            selectable_regions = set(self.regions).intersection(set(selected_allen_regions))
        else:
            selectable_regions = selected_allen_regions
        return self.select_from_list(list(selectable_regions), fill_nan=fill_nan, *args, **kwargs)

    def get_units(self, marker=None):
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self.markers_data[marker].units

    def to_pandas(self, units=False) -> pd.DataFrame:
        data = pd.concat({f"area ({self.areas.units})" if units else "area": self.areas.data,
                          **{f"{m} ({m_data.units})" if units else m: m_data.data for m,m_data in self.markers_data.items()}}, axis=1)
        data.columns.name = str(self.mode)
        return data

    def to_csv(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        mode_str = str(self.mode)
        output_path = os.path.join(output_path, f"{self.name}_{mode_str}.csv")
        data = self.to_pandas(units=True)
        data.to_csv(output_path, sep="\t", mode="w", index_label=mode_str)
        print(f"{self} saved to {output_path}")

    @staticmethod
    def is_raw(mode: str) -> bool:
        try:
            return SliceMetrics(mode)._raw            
        except ValueError:
            return mode == AnimalBrain.RAW_DATA

    @staticmethod
    def from_pandas(animal_name, df: pd.DataFrame) -> Self:
        if type(mode:=df.columns.name) != str:
            mode = str(df.columns.name)
        raw = AnimalBrain.is_raw(mode)
        markers_data = dict()
        areas = None
        regex = r'(.+) \((.+)\)$'
        pattern = re.compile(regex)
        for column, data in df.items():
            # extracts name and units from the column's name. E.g. 'area (mm²)' -> ('area', 'mm²')
            matches = re.findall(pattern, column)
            name, units = matches[0] if len(matches) == 1 else (column, None)
            if name == "area":
                areas = BrainData(data, animal_name, mode, units)
            else: # it's a marker
                markers_data[name] = BrainData(data, animal_name, mode, units)
        return AnimalBrain(markers_data=markers_data, areas=areas, raw=raw)
    
    @staticmethod
    def exists_csv(animal_name, root_dir, mode: str=None) -> bool:
        filename = f"{animal_name}.csv" if mode is not None else f"{animal_name}_{str(mode)}.csv"
        return os.path.exists(os.path.join(root_dir, filename))

    @staticmethod
    def from_csv(animal_name, root_dir, mode: str=None) -> Self:
        # read CSV
        filename = f"{animal_name}.csv" if mode is None else f"{animal_name}_{str(mode)}.csv"
        df = pd.read_csv(os.path.join(root_dir, filename), sep="\t", header=0, index_col=0)
        if df.index.name == "Class":
            # is old csv
            raise ValueError("Trying to read an AnimalBrain from an outdated formatted .csv. Please re-run the analysis from the SlicedBrain!")
        df.columns.name = df.index.name
        df.index.name = None
        return AnimalBrain.from_pandas(animal_name, df)

    @staticmethod
    def filter_selected_regions(animal_brain: Self, brain_ontology: AllenBrainHierarchy) -> Self:
        brain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.select_from_ontology(brain_ontology) for m, m_data in brain.markers_data.items()}
        brain.areas = brain.areas.select_from_ontology(brain_ontology)
        return brain

    @staticmethod
    def merge_hemispheres(animal_brain: Self) -> Self:
        brain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.merge_hemispheres() for m, m_data in brain.markers_data.items()}
        brain.areas = brain.areas.merge_hemispheres()
        return brain

def extract_name_and_units(ls) -> Generator[str, None, None]:
    regex = r'(.+) \((.+)\)$'
    pattern = re.compile(regex)
    for s in ls:
        matches = re.findall(pattern, s)
        assert len(matches) == 1, f"Cannot find units in column '{s}'"
        yield matches[0]