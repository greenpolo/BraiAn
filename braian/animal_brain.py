# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import copy
import numpy as np
import os
import pandas as pd
import re

from enum import Enum, auto
from typing import Self

from braian.brain_hierarchy import AllenBrainHierarchy
from braian.brain_metrics import BrainMetrics
from braian.sliced_brain import SlicedBrain
from braian.brain_data import BrainData

class AnimalBrain:
    def __init__(self, markers_data: dict[BrainData]=None, areas: BrainData=None) -> None:
        assert len(markers_data) > 0 and areas is not None, "You must provide both a dictionary of BrainData (markers) and an additional BrainData for the areas/volumes of each region"
        first_data = tuple(markers_data.values())[0]
        self.name = first_data.data_name
        self.mode = BrainMetrics(first_data.metric)
        self.is_split = first_data.is_split
        assert all([m.data_name == self.name for m in markers_data.values()]), "All BrainData must be from the same animal!"
        assert all([BrainMetrics(m.metric) == self.mode for m in markers_data.values()]), "All BrainData must be of the same metric!"
        assert self.is_split == areas.is_split and all([m.is_split == self.is_split for m in markers_data.values()]), "All BrainData must either have split hemispheres or not!"
        self.markers = list(markers_data.keys())
        self.markers_data = markers_data
        self.areas = areas
        return

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

    def remove_smaller_subregions(self, area_threshold, brain_onthology: AllenBrainHierarchy) -> None:
        small_regions = {smaller_region for small_region in self.areas.data[self.areas.data <= area_threshold].index
                                        for smaller_region in brain_onthology.list_all_subregions(small_region)}
        self.remove_region(*small_regions)

    def get_regions(self):
        # assumes areas' and all markers' BrainData are synchronized
        return self.areas.get_regions()

    def sort_by_onthology(self, brain_onthology: AllenBrainHierarchy,
                          fill=False, inplace=False):
        markers_data = {marker: m_data.sort_by_onthology(brain_onthology, fill=fill, inplace=inplace) for marker, m_data in self.markers_data.items()}
        areas = self.areas.sort_by_onthology(brain_onthology, fill=fill, inplace=inplace)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, areas=areas)
        else:
            return self

    def select_from_list(self, regions: list[str], fill_nan=False, inplace=False) -> Self:
        markers_data = {marker: m_data.select_from_list(regions, fill_nan=fill_nan, inplace=inplace) for marker, m_data in self.markers_data.items()}
        areas = self.areas.select_from_list(regions, fill_nan=fill_nan, inplace=inplace)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, areas=areas)
        else:
            return self

    def select_from_onthology(self, brain_onthology: AllenBrainHierarchy, fill_nan=False, *args, **kwargs) -> Self:
        selected_allen_regions = brain_onthology.get_selected_regions()
        if not fill_nan:
            selectable_regions = set(self.get_regions()).intersection(set(selected_allen_regions))
        else:
            selectable_regions = selected_allen_regions
        return self.select_from_list(list(selectable_regions), fill_nan=fill_nan, *args, **kwargs)

    def get_units(self, marker=None):
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self.markers_data[marker].units

    def density(self) -> Self:
        assert self.mode == BrainMetrics.SUM, f"Cannot compute densities for AnimalBrains whose slices' cell count were not summed (mode={self.mode})."
        markers_data = dict()
        for marker in self.markers:
            data = self.markers_data[marker] / self.areas
            data.metric = str(BrainMetrics.DENSITY)
            data.units = f"{marker}/{self.areas.units}"
            markers_data[marker] = data
        return AnimalBrain(markers_data=markers_data, areas=self.areas)

    def percentage(self) -> BrainData:
        assert self.mode == BrainMetrics.SUM, "Cannot compute percentages for AnimalBrains whose slices' cell count were not summed."
        if self.is_split:
            hems = ("L", "R")
        else:
            hems = (None,)
        markers_data = dict()
        for marker in self.markers:
            brainwide_cell_counts = sum((self.markers_data[marker].root(hem) for hem in hems))
            data = self.markers_data[marker] / brainwide_cell_counts
            data.metric = str(BrainMetrics.PERCENTAGE)
            data.units = f"{marker}/{marker} in root"
            markers_data[marker] = data
        return AnimalBrain(markers_data=markers_data, areas=self.areas)

    def relative_density(self) -> BrainData:
        assert self.mode == BrainMetrics.SUM, "Cannot compute relative densities for AnimalBrains whose slices' cell count were not summed."
        if self.is_split:
            hems = ("L", "R")
        else:
            hems = (None,)
        markers_data = dict()
        for marker in self.markers:
            brainwide_area = sum((self.areas.root(hem) for hem in hems))
            brainwide_cell_counts = sum((self.markers_data[marker].root(hem) for hem in hems))
            data = (self.markers_data[marker] / self.areas) / (brainwide_cell_counts / brainwide_area)
            data.metric = str(BrainMetrics.RELATIVE_DENSITY)
            data.units = f"{marker} density/root {marker} density"
            markers_data[marker] = data
        return AnimalBrain(markers_data=markers_data, areas=self.areas)

    def markers_overlap(self, marker1: str, marker2: str) -> Self:
        if self.mode not in (BrainMetrics.SUM, BrainMetrics.MEAN):
            raise ValueError("Cannot compute the overlapping of two markers for AnimalBrains whose slices' cell count were not summed or averaged.")
        for m in (marker1, marker2):
            if m not in self.markers:
                raise ValueError(f"Marker '{m}' is unknown in '{self.name}'!")
        try:
            both = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in self.markers)
        except StopIteration as e:
            raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
        overlaps = dict()
        for m in (marker1, marker2):
            # TODO: clipping overlaps to 100% because of a bug with the QuPath script that counts overlapping cells as belonging to different regions
            overlaps[m] = (self.markers_data[both] / self.markers_data[m]).clip(upper=1)
            overlaps[m].metric = str(BrainMetrics.OVERLAPPING)
            overlaps[m].units = f"({marker1}+{marker2})/{m}"
        return AnimalBrain(markers_data=overlaps, areas=self.areas)
    
    def markers_jaccard_index(self, marker1: str, marker2: str) -> Self:
        # computes Jaccard's index
        if self.mode not in (BrainMetrics.SUM, BrainMetrics.MEAN):
            raise ValueError("Cannot compute the overlapping of two markers for AnimalBrains whose slices' cell count were not summed or averaged.")
        for m in (marker1, marker2):
            if m not in self.markers:
                raise ValueError(f"Marker '{m}' is unknown in '{self.name}'!")
        try:
            overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in self.markers)
        except StopIteration as e:
            raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
        similarities = self.markers_data[overlapping] / (self.markers_data[marker1]+self.markers_data[marker2]-self.markers_data[overlapping])
        similarities.metric = str(BrainMetrics.JACCARD_INDEX)
        similarities.units = f"({marker1}∩{marker2})/({marker1}∪{marker2})"
        return AnimalBrain(markers_data={overlapping: similarities}, areas=self.areas)
    
    def markers_similarity_index(self, marker1: str, marker2: str) -> Self:
        # computes Jaccard's index
        if self.mode not in (BrainMetrics.SUM, BrainMetrics.MEAN):
            raise ValueError("Cannot compute the overlapping of two markers for AnimalBrains whose slices' cell count were not summed or averaged.")
        for m in (marker1, marker2):
            if m not in self.markers:
                raise ValueError(f"Marker '{m}' is unknown in '{self.name}'!")
        try:
            overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in self.markers)
        except StopIteration as e:
            raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
        # NOT normalized in (0,1)
        # similarities = self.markers_data[overlapping] / (self.markers_data[marker1]*self.markers_data[marker2]) * self.areas
        # NORMALIZED
        similarities = self.markers_data[overlapping]**2 / (self.markers_data[marker1]*self.markers_data[marker2])
        similarities.metric = str(BrainMetrics.SIMILARITY_INDEX)
        similarities.units = f"({marker1}∩{marker2})²/({marker1}×{marker2})"
        return AnimalBrain(markers_data={overlapping: similarities}, areas=self.areas)

    def markers_chance_level(self, marker1: str, marker2: str) -> Self:
        # This chance level is good only if the used for the fold change.
        #
        # It is similar to our Similarity Index, as it is derived from its NOT normalized form.
        # ideally it would use the #DAPI instead of the area, as that would give an interval
        # which is easier to work with.
        # However, when:
        #  * the DAPI is not available AND
        #  * we're interested in the difference of fold change between groups
        # we can ignore the DAPI count it simplifies during the rate group1/group2
        # thus the use case of this index.
        #
        # since the areas/DAPI simplifies only when they are ~comparable between animals,
        # we force the AnimalBrain to be a result of MEAN of SlicedBrain, not SUM of SlicedBrain
        if self.mode != BrainMetrics.MEAN:
            raise ValueError("Cannot compute the overlapping of two markers for AnimalBrains whose slices' cell count were not averaged.")
        for m in (marker1, marker2):
            if m not in self.markers:
                raise ValueError(f"Marker '{m}' is unknown in '{self.name}'!")
        try:
            overlapping = next(m for m in (f"{marker1}+{marker2}", f"{marker2}+{marker1}") if m in self.markers)
        except StopIteration as e:
            raise ValueError(f"Overlapping data between '{marker1}' and '{marker2}' are not available. Are you sure you ran the QuPath script correctly?")
        chance_level = self.markers_data[overlapping] / (self.markers_data[marker1]*self.markers_data[marker2])
        chance_level.metric = str(BrainMetrics.CHANCE_LEVEL)
        chance_level.units = f"({marker1}∩{marker2})/({marker1}×{marker2})"
        return AnimalBrain(markers_data={overlapping: chance_level}, areas=self.areas)

    def markers_difference(self, marker1: str, marker2: str) -> Self:
        if self.mode == BrainMetrics.SUM:
            return self.density().markers_difference(marker1, marker2)
        elif self.mode != BrainMetrics.DENSITY:
            raise ValueError("Cannot compute the marker difference of two markers for AnimalBrains whose data is not density or sum")
        for m in (marker1, marker2):
            if m not in self.markers:
                raise ValueError(f"Marker '{m}' is unknown in '{self.name}'!")
        diff = self.markers_data[marker1] - self.markers_data[marker2]
        diff.metric = str(BrainMetrics.DENSITY_DIFFERENCE)
        diff.units = f"{marker1}-{marker2}"
        return AnimalBrain(markers_data={f"{marker1}+{marker2}": diff}, areas=self.areas)
    
    def _group_change(self, group, metric, fun: callable, symbol: str) -> BrainData: # AnimalGroup
        assert self.is_split == group.is_split, "Both AnimalBrain and AnimalGroup must either have the hemispheres split or not"
        assert set(self.markers) == set(group.markers), "Both AnimalBrain and AnimalGroup must have the same markers"
        # assert self.mode == group.metric == BrainMetrics.DENSITY, f"Both AnimalBrain and AnimalGroup must be on {BrainMetrics.DENSITY}"
        # assert set(self.get_regions()) == set(group.get_regions()), f"Both AnimalBrain and AnimalGroup must be on the same regions"
        
        markers_data = dict()
        for marker,this in self.markers_data.items():
            data = fun(this, group.mean[marker])
            data.metric = str(metric)
            data.units = f"{marker} {str(self.mode)}{symbol}{group.name} {str(group.metric)}"
            markers_data[marker] = data
        return AnimalBrain(markers_data=markers_data, areas=self.areas)

    def fold_change(self, group) -> BrainData: # AnimalGroup
        return self._group_change(group, BrainMetrics.FOLD_CHANGE, lambda animal,group: animal/group, "/")

    def diff_change(self, group) -> BrainData: # AnimalGroup
        return self._group_change(group, BrainMetrics.DIFF_CHANGE, lambda animal,group: animal-group, "-")

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
    def from_pandas(animal_name, df: pd.DataFrame) -> Self:
        if type(mode:=df.columns.name) != str:
            mode = str(df.columns.name)
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
        return AnimalBrain(markers_data=markers_data, areas=areas)
    
    @staticmethod
    def exists_csv(animal_name, root_dir, mode) -> bool:
        return os.path.exists(os.path.join(root_dir, f"{animal_name}_{str(mode)}.csv"))

    @staticmethod
    def from_csv(animal_name, root_dir, mode) -> Self:
        # read CSV
        df = pd.read_csv(os.path.join(root_dir, f"{animal_name}_{str(mode)}.csv"), sep="\t", header=0, index_col=0)
        if df.index.name == "Class":
            # is old csv
            raise ValueError("Trying to read an AnimalBrain from an outdated formatted .csv. Please re-run the analysis from the SlicedBrain!")
        df.columns.name = df.index.name
        df.index.name = None
        return AnimalBrain.from_pandas(animal_name, df)

    @staticmethod
    def from_slices(sliced_brain: SlicedBrain, mode=BrainMetrics.SUM, min_slices=0, hemisphere_distinction=True) -> Self:
        if not hemisphere_distinction:
            sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)

        name = sliced_brain.name
        markers = copy.copy(sliced_brain.markers)
        mode = BrainMetrics(mode)
        match mode:
            case BrainMetrics.SUM:
                redux = AnimalBrain.sum_slices_detections(sliced_brain, min_slices)
            case BrainMetrics.DENSITY | BrainMetrics.PERCENTAGE | BrainMetrics.RELATIVE_DENSITY | BrainMetrics.OVERLAPPING:
                raise NotImplementedError(f"Cannot yet build AnimalBrain(name='{name}', mode={mode}, markers={list(markers)}) from a SlicedBrain."+\
                                          "Use BrainMetrics.DENSITY.analyse(brain) method.")
            case _: # MEAN | CVAR | STD
                redux = AnimalBrain.reduce_slices_densities(sliced_brain, mode, min_slices)
        areas = BrainData(redux["area"], name=name, metric=str(mode), units="mm²")
        markers_data = {
            m: BrainData(redux[m], name=name, metric=str(mode), units=m)
            for m in markers
        }
        return AnimalBrain(markers_data=markers_data, areas=areas)

    @staticmethod
    def sum_slices_detections(sliced_brain: SlicedBrain, min_slices: int) -> pd.DataFrame:
        all_slices = sliced_brain.concat_slices()
        redux = all_slices.groupby(all_slices.index)\
                            .sum(min_count=min_slices)\
                            .dropna(axis=0, how="all")\
                            .astype({m: sliced_brain.get_marker_dtype(m) for m in sliced_brain.markers}) # dropna() changes type to float64
        return redux

    @staticmethod
    def reduce_slices_densities(sliced_brain: SlicedBrain, mode: BrainMetrics, min_slices: int) -> pd.DataFrame:
        all_slices = sliced_brain.concat_slices(densities=True)
        redux = all_slices.groupby(all_slices.index)\
                            .apply(mode.fold_slices(min_slices))\
                            .dropna(axis=0, how="all") # we want to keep float64 as the dtype, since the result of the 'mode' function is a float as well
        return redux

    @staticmethod
    def filter_selected_regions(animal_brain: Self, brain_onthology: AllenBrainHierarchy) -> Self:
        brain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.select_from_onthology(brain_onthology) for m, m_data in brain.markers_data.items()}
        brain.areas = brain.areas.select_from_onthology(brain_onthology)
        return brain

    @staticmethod
    def merge_hemispheres(animal_brain: Self) -> Self:
        brain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.merge_hemispheres() for m, m_data in brain.markers_data.items()}
        brain.areas = brain.areas.merge_hemispheres()
        return brain

def extract_name_and_units(ls) -> str:
    regex = r'(.+) \((.+)\)$'
    pattern = re.compile(regex)
    for s in ls:
        matches = re.findall(pattern, s)
        assert len(matches) == 1, f"Cannot find units in column '{s}'"
        yield matches[0]