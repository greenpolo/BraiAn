import copy
import os
import numpy as np
import numpy.typing as npt
import pandas as pd
from collections.abc import Callable, Sequence
from itertools import product
from functools import reduce
from typing import Self

from braian.brain_data import BrainData
from braian.ontology import AllenBrainOntology
from braian.animal_brain import AnimalBrain
from braian.utils import save_csv

__all__ = ["AnimalGroup", "PLS"]

def _common_regions(animals: list[AnimalBrain]) -> list[str]:
    all_regions = [set(brain.regions) for brain in animals]
    return list(reduce(set.__or__, all_regions))

def _have_same_regions(animals: list[AnimalBrain]) -> bool:
    regions = animals[0].regions
    all_regions = [set(brain.regions) for brain in animals]
    return len(reduce(set.__and__, all_regions)) ==  len(regions)

class AnimalGroup:
    def __init__(self, name: str, animals: Sequence[AnimalBrain], merge_hemispheres: bool=False,
                 brain_ontology: AllenBrainOntology=None, fill_nan: bool=True) -> None:
        """
        Creates an experimental cohort from a set of `AnimalBrain`.\
        In order for a cohort to be valid, it must consist of brains with
        the same type of data (i.e. [metric][braian.AnimalBrain.mode]),
        the same [markers][braian.AnimalBrain.markers] and
        the data must all be hemisphere-aware or not (i.e. [`AnimalBrain.is_split`][braian.AnimalBrain.is_split]).

        Parameters
        ----------
        name
            The name of the cohort.
        animals
            The animals part of the group.
        merge_hemispheres
            If True, it merges, for each region, the data from left/right hemispheres into a single value.
        brain_ontology
            The ontology to which the brains' data was registered against.
            If specified, it sorts the data in the depth-first search order with respect to the hierarchy.
        fill_nan
            If True, it sets the value to [`NaN`][numpy.nan] for all the regions missing in the given `animals`.\\
            A region is missing if:

            * it is in `brain_ontology`, when specified;
            * or it is present in some animals but not all.

        Raises
        ------
        ValueError
            When there is no option to make sure that all animals of the cohort work on the same brain regions,
            as `fill_nan=False`, `brain_ontology=None` and some animal misses at least one region compared to the rest.\
            See [`AnimalBrain.select_from_list`][braian.AnimalBrain.select_from_list] or
            [`AnimalBrain.select_from_ontology`][braian.AnimalBrain.select_from_ontology] if you want to prepare
            the brains in advance.

        See also
        --------
        [`AnimalBrain.merge_hemispheres`][braian.AnimalBrain.merge_hemispheres]
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        [`AnimalBrain.sort_by_ontology`][braian.AnimalBrain.sort_by_ontology]
        [`BrainData.sort_by_ontology`][braian.BrainData.sort_by_ontology]
        [`AnimalBrain.select_from_list`][braian.AnimalBrain.select_from_list]
        [`AnimalBrain.select_from_ontology`][braian.AnimalBrain.select_from_ontology]
        """        
        self.name = name
        """The name of the group."""
        # if not animals or not brain_ontology:
        #     raise ValueError("You must specify animals: list[AnimalBrain] and brain_ontology: AllenBrainOntology.")
        assert len(animals) > 0, "A group must be made of at least one animal." # TODO: should we enforce a statistical signficant n? E.g. MIN=4
        _all_markers = {marker for brain in animals for marker in brain.markers}
        assert all(marker in brain.markers for marker in _all_markers for brain in animals), "All AnimalBrain in a group must have the same markers."
        assert all(brain.mode == animals[0].mode for brain in animals[1:]), "All AnimalBrains in a group must be have the same metric."
        is_split = animals[0].is_split
        assert all(is_split == brain.is_split for brain in animals), "All AnimalBrains of a group must either have spit hemispheres or not."
        if is_split and merge_hemispheres:
            merge = AnimalBrain.merge_hemispheres
        else:
            merge = lambda brain: brain
        if brain_ontology is not None:
            sort: Callable[[AnimalBrain], AnimalBrain] = lambda brain: brain.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=False)
        elif fill_nan:
            regions = _common_regions(animals)
            sort: Callable[[AnimalBrain], AnimalBrain] = lambda brain: brain.select_from_list(regions, fill_nan=True, inplace=False)
        elif _have_same_regions(animals):
            sort = lambda brain: brain
        else:
            # now BrainGroup.regions, which returns the regions of the first animal, is correct
            raise ValueError("Cannot set fill_nan=False and brain_ontology=None if all animals of the group don't have the same brain regions.")
        self._animals: list[AnimalBrain] = [sort(merge(brain)) for brain in animals] # brain |> merge |> sort -- OLD: brain |> merge |> analyse |> sort
        self._mean: dict[str, BrainData] = self._update_mean()

    @property
    def n(self) -> int:
        return len(self._animals)

    @property
    def metric(self) -> str:
        return self._animals[0].mode

    @property
    def is_split(self) -> bool:
        return self._animals[0].is_split

    @property
    def markers(self) -> npt.NDArray[np.str_]:
        return np.asarray(self._animals[0].markers)

    @property
    def animals(self) -> list[AnimalBrain]:
        return list(self._animals)

    @property
    def mean(self) -> dict[str, BrainData]:
        return dict(self._mean)

    def markers_corr(self, marker1: str, marker2: str, other: Self=None) -> BrainData:
        if other is None:
            other = self
        else:
            assert self.metric == other.metric
        corr = self.to_pandas(marker1).corrwith(other.to_pandas(marker2), method="pearson", axis=1)
        return BrainData(corr, self.name, str(self.metric)+f"-corr (n={self.n})", f"corr({marker1}, {marker2})")
    
    def __str__(self) -> str:
        return f"AnimalGroup('{self.name}', metric={self.metric}, n={self.n})"

    def _update_mean(self) -> dict[str, BrainData]:
        return {marker: BrainData.mean(*[brain[marker] for brain in self._animals], name=self.name) for marker in self.markers}

    def combine(self, op, **kwargs) -> dict[str, BrainData]:
        """
        _summary_

        Parameters
        ----------
        op
            _description_
        **kwargs
            Other keyword arguments are passed to [`BrainData.reduce`][braian.BrainData.reduce].

        Returns
        -------
        :
            _description_
        """
        return {marker: BrainData.reduce(*[brain[marker] for brain in self._animals], op=op, name=self.name, **kwargs) for marker in self.markers}

    def to_pandas(self, marker=None, units=False) -> pd.DataFrame:
        if marker in self.markers:
            df = pd.concat({brain.name: brain[marker].data for brain in self._animals}, join="outer", axis=1)
            df.columns.name = str(self.metric)
            if units:
                a = self._animals[0]
                df.rename(columns={marker: f"{marker} ({a[marker].units})"}, inplace=True)
            return df
        df = {"area": pd.concat({brain.name: brain.areas.data for brain in self._animals}, join="outer", axis=0)}
        for marker in self.markers:
            all_animals = pd.concat({brain.name: brain[marker].data for brain in self._animals}, join="outer", axis=0)
            df[marker] = all_animals
        df = pd.concat(df, join="outer", axis=1)
        df = df.reorder_levels([1,0], axis=0)
        ordered_indices = product(self.regions, [animal.name for animal in self._animals])
        df = df.reindex(ordered_indices)
        df.columns.name = str(self.metric)
        if units:
            a = self._animals[0]
            df.rename(columns={col: f"{col} ({a[col].units if col != 'area' else a.areas.units})" for col in df.columns}, inplace=True)
        return df
    
    def sort_by_ontology(self, brain_ontology: AllenBrainOntology, fill_nan=True, inplace=True) -> None:
        if not inplace:
            return AnimalGroup(self.name, self._animals, brain_ontology=brain_ontology, fill_nan=fill_nan)
        else:
            for brain in self._animals:
                brain.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=True)
            return self
    
    def get_animals(self) -> list[str]:
        return [brain.name for brain in self._animals]

    @property
    def regions(self) -> list[str]:
        # NOTE: all animals of the group are expected to have the same regions!
        # if not have_same_regions(animals):
        #     # NOTE: if the animals of the AnimalGroup were not sorted by ontology, the order is not guaranteed to be significant
        #     print(f"WARNING: animals of {self} don't have the same brain regions. "+\
        #           "The order of the brain regions is not guaranteed to be significant. It's better to first call sort_by_ontology()")
        #     return list(reduce(set.union, all_regions))
        #     # return set(chain(*all_regions))
        return self._animals[0].regions

    def merge_hemispheres(self, inplace=False) -> Self:
        animals = [AnimalBrain.merge_hemispheres(brain) for brain in self._animals]
        if not inplace:
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self._animals = animals
            self._mean = self._update_mean()
            return self

    def is_comparable(self, other) -> bool:
        if not isinstance(other, AnimalGroup):
            return False
        return set(self.markers) == set(other.markers) and \
                self.is_split == other.is_split and \
                self.metric == other.metric # and \
                # set(self.regions) == set(other.regions)
    
    def select(self, regions: list[str], fill_nan=False, inplace=False) -> Self:
        animals = [brain.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for brain in self._animals]
        if not inplace:
            # self.metric == animals.metric -> no self.metric.analyse(brain) is computed
            return AnimalGroup(self.name, animals, brain_ontology=None, fill_nan=False)
        else:
            self._animals = animals
            self._mean = self._update_mean()
            return self

    def select_animal(self, animal_name: str) -> AnimalBrain:
        return next((brain for brain in self._animals if brain.name == animal_name))

    def get_units(self, marker=None) -> str:
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self._animals[0].get_units(marker)

    def to_csv(self, output_path, file_name, overwrite=False) -> None:
        df = self.to_pandas(units=True)
        save_csv(df, output_path, file_name, overwrite=overwrite, index_label=(df.columns.name, None))

    @staticmethod
    def from_pandas(df: pd.DataFrame, group_name: str) -> Self:
        animals = [AnimalBrain.from_pandas(df.xs(animal_name, level=1), animal_name) for animal_name in df.index.unique(1)]
        return AnimalGroup(group_name, animals, fill_nan=False)

    @staticmethod
    def from_csv(group_name, root_dir, file_name) -> Self:
        # # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=0, index_col=[0,1])
        df.columns.name = df.index.names[0]
        df.index.names = (None, None)
        return AnimalGroup.from_pandas(df, group_name)