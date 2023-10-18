import copy
import os
import numpy as np
import pandas as pd
from itertools import product, chain
from functools import reduce
from typing import Self

from .brain_data import BrainData
from .brain_hierarchy import AllenBrainHierarchy
from .brain_metrics import BrainMetrics
from .animal_brain import AnimalBrain, BrainMetrics
from .utils import save_csv

def common_regions(animals: list[AnimalBrain]) -> list[str]:
    all_regions = [set(brain.get_regions()) for brain in animals]
    return list(reduce(set.__or__, all_regions))

def have_same_regions(animals: list[AnimalBrain]) -> bool:
    regions = animals[0].get_regions()
    all_regions = [set(brain.get_regions()) for brain in animals]
    return len(reduce(set.__and__, all_regions)) ==  len(regions)

class AnimalGroup:
    def __init__(self, name: str, animals: list[AnimalBrain], metric: BrainMetrics, merge_hemispheres=False,
                 brain_onthology: AllenBrainHierarchy=None, fill_nan=True, **kwargs) -> None:
        self.name = name
        # if not animals or not brain_onthology:
        #     raise ValueError("You must specify animals: list[AnimalBrain] and brain_onthology: AllenBrainHierarchy.")
        assert len(animals) > 0, "Inside the group there must be at least one animal."
        self.markers = np.asarray(animals[0].markers)
        assert all([marker in self.markers for brain in animals for marker in brain.markers]), "All AnimalBrain composing the group must use the same markers."
        self.metric = BrainMetrics(metric)
        assert all([brain.mode == animals[0].mode for brain in animals]), "All AnimalBrains of a group must be hava been processed the same way."
        self.n = len(animals)
        self.is_split = animals[0].is_split
        assert all(self.is_split == brain.is_split for brain in animals), "All AnimalBrains of a group must either have spit hemispheres or not."
        if self.is_split and merge_hemispheres:
            merge = AnimalBrain.merge_hemispheres
        else:
            merge = lambda brain: brain
        if animals[0].mode != self.metric:
            analyse = lambda brain: self.metric.analyse(brain, **kwargs)
        else:
            analyse = lambda brain: brain
        if brain_onthology is not None:
            sort = lambda brain: brain.sort_by_onthology(brain_onthology, fill=fill_nan, inplace=False)
        elif fill_nan:
            regions = common_regions(animals)
            sort = lambda brain: brain.select_from_list(regions, fill=True, inplace=False)
        elif have_same_regions(animals):
            sort = lambda brain: brain
        else:
            # now BrainGroup.get_regions(), which returns the regions of the first animal, is correct
            raise ValueError("Cannot set fill_nan=False and brain_onthology=None if all animals of the group don't have the same brain regions.")
        self.animals: list[AnimalBrain] = [sort(analyse(merge(brain))) for brain in animals]
        self.mean = self._update_mean()
    
    def __str__(self) -> str:
        return f"AnimalGroup(metric={self.metric}, n={self.n})"

    def _update_mean(self) -> dict:
        return {marker: BrainData.mean(*[brain[marker] for brain in self.animals]) for marker in self.markers}

    def combine(self, op, **kwargs):
        return {marker: BrainData.merge(*[brain[marker] for brain in self.animals], op=op, **kwargs) for marker in self.markers}

    def to_pandas(self, marker=None, units=False):
        if marker in self.markers:
            df = pd.concat({brain.name: brain.markers_data[marker].data for brain in self.animals}, join="outer", axis=1)
            df.columns.name = str(self.metric)
            if units:
                a = self.animals[0]
                df.rename(columns={marker: f"{marker} ({a[marker].units})"}, inplace=True)
            return df
        df = {"area": pd.concat({brain.name: brain.areas.data for brain in self.animals}, join="outer", axis=0)}
        for marker in self.markers:
            all_animals = pd.concat({brain.name: brain.markers_data[marker].data for brain in self.animals}, join="outer", axis=0)
            df[marker] = all_animals
        df = pd.concat(df, join="outer", axis=1)
        df = df.reorder_levels([1,0], axis=0)
        ordered_indices = product(self.get_regions(), [animal.name for animal in self.animals])
        df = df.reindex(ordered_indices)
        df.columns.name = str(self.metric)
        if units:
            a = self.animals[0]
            df.rename(columns={col: f"{col} ({a[col].units if col != 'area' else a.areas.units})" for col in df.columns}, inplace=True)
        return df
    
    def sort_by_onthology(self, brain_onthology: AllenBrainHierarchy, fill=True, inplace=True) -> None:
        for brain in self.animals:
            brain.sort_by_onthology(brain_onthology, fill=fill, inplace=True)
    
    def get_animals(self):
        return [brain.name for brain in self.animals]

    def get_regions(self) -> list[str]:
        # NOTE: all animals of the group are expected to have the same regions!
        # if not have_same_regions(animals):
        #     # NOTE: if the animals of the AnimalGroup were not sorted by onthology, the order is not guaranteed to be significant
        #     print(f"WARNING: animals of {self} don't have the same brain regions. "+\
        #           "The order of the brain regions is not guaranteed to be significant. It's better to first call sort_by_onthology()")
        #     return list(reduce(set.union, all_regions))
        #     # return set(chain(*all_regions))
        return self.animals[0].get_regions()

    def merge_hemispheres(self, inplace=False):
        animals = [AnimalBrain.merge_hemispheres(brain) for brain in self.animals]
        if not inplace:
            return AnimalGroup(self.name, animals, metric=self.metric, brain_onthology=None, fill_nan=False)
        else:
            self.animals = animals
            self.mean = self._update_mean()
            return self

    def is_comparable(self, other) -> bool:
        if not isinstance(other, AnimalGroup):
            return False
        return set(self.markers) == set(other.markers) and \
                self.is_split == other.is_split and \
                self.metric == other.metric # and \
                # set(self.get_regions()) == set(other.get_regions())
    
    def select(self, regions: list[str], fill_nan=False, inplace=False) -> Self:
        animals = [brain.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for brain in self.animals]
        if not inplace:
            # self.metric == animals.metric -> no self.metric.analyse(brain) is computed
            return AnimalGroup(self.name, animals, metric=self.metric, brain_onthology=None, fill_nan=False)
        else:
            self.animals = animals
            self.mean = self._update_mean()
            return self

    def select_animal(self, animal_name: str):
        return next((brain for brain in self.animals if brain.name == animal_name))

    def remove_smaller_subregions(self, *args, **kwargs) -> None:
        for brain in self.animals:
            brain.remove_smaller_subregions(*args, **kwargs)
        self.mean = self._update_mean()

    def get_units(self, marker=None):
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self.animals[0].get_units(marker)

    def get_plot_title(self, marker=None):
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get the plot title for marker '{marker}'!"
        match self.metric:
            case BrainMetrics.DENSITY:
                return f"[#{marker} / area]"
            case BrainMetrics.PERCENTAGE:
                return f"[#{marker} / brain]"
            case BrainMetrics.RELATIVE_DENSITY:
                return f"[#{marker} / area] / [{marker} (brain) / area (brain)]"
            case _:
                raise ValueError(f"Don't know the appropriate title for {self.metric}")

    def to_csv(self, output_path, file_name, overwrite=False) -> None:
        df = self.to_pandas(units=True)
        save_csv(df, output_path, file_name, overwrite=overwrite, index_label=(df.columns.name, None))

    @staticmethod
    def from_pandas(group_name, df: pd.DataFrame):
        animals = [AnimalBrain.from_pandas(animal_name, df.xs(animal_name, level=1)) for animal_name in df.index.unique(1)]
        return AnimalGroup(group_name, animals, df.columns.name, fill_nan=False)

    @staticmethod
    def from_csv(group_name, root_dir, file_name):
        # # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=0, index_col=[0,1])
        df.columns.name = df.index.names[0]
        df.index.names = (None, None)
        return AnimalGroup.from_pandas(group_name, df)