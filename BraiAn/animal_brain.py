import copy
import numpy as np
import os
import pandas as pd

from typing import Self

from .brain_hierarchy import AllenBrainHierarchy
from .brain_slice import extract_acronym, is_split_left_right
from .sliced_brain import merge_sliced_hemispheres, SlicedBrain
from .brain_data import BrainData

def min_count(fun, min, **kwargs):
    def nan_if_less(xs):
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

class AnimalBrain:
    def __init__(self, sliced_brain, mode="sum", hemisphere_distinction=True,
                min_slices=0, markers_data: dict[BrainData]=None, areas: BrainData=None) -> None:
        if markers_data is not None:
            first_data = tuple(markers_data.values())[0]
            self.name = first_data.name
            self.mode = first_data.metric
            self.is_split = first_data.is_split
            assert all([m.name == self.name for m in markers_data.values()]), "All BrainData must be from the same animal!"
            assert all([m.metric == self.mode for m in markers_data.values()]), "All BrainData must be of the same metric!"
            assert all([m.is_split == self.is_split for m in markers_data.values()]), "All BrainData must either have split hemispheres or not!"
            self.markers = list(markers_data.keys())
            self.markers_data = markers_data
            self.areas = areas
            return
        if not hemisphere_distinction:
            sliced_brain = merge_sliced_hemispheres(sliced_brain)
        
        self.name = sliced_brain.name
        self.markers = copy.copy(sliced_brain.markers)
        self.mode = self.__simple_mode_name(mode)
        self.is_split = sliced_brain.is_split
        match mode:
            case "sum":
                redux = self.sum_slices(sliced_brain, min_slices)
            case "overlap" | "%overlapping":
                raise NotImplementedError("Can't yet build an AnimalBrain of marker overlappings from a SlicedBrain. Use AnimalBrain.overlap_markers() method.")
            # self.data = self.overlap_markers()
            case _:
                redux = self.reduce_brain_densities(sliced_brain, self.mode, min_slices)
        self.areas = BrainData(redux["area"], name=self.name, metric=self.mode, units="mm²")
        self.markers_data = {
            m: BrainData(redux[m], name=self.name, metric=self.mode, units=m)
            for m in self.markers
        }

    def sum_slices(self, sliced_brain: SlicedBrain, min_slices: int) -> pd.DataFrame:
        all_slices = sliced_brain.concat_slices()
        redux = all_slices.groupby(all_slices.index)\
                            .sum(min_count=min_slices)\
                            .dropna(axis=0, how="all")\
                            .astype({m: sliced_brain.get_marker_dtype(m) for m in sliced_brain.markers}) # dropna() changes type to float64
        return redux

    def reduce_brain_densities(self, sliced_brain: SlicedBrain, mode: str, min_slices: int) -> pd.DataFrame:
        match mode:
            case "mean" | "avg":
                reduction_fun = min_count(np.mean, min_slices, axis=0)
            case "std":
                reduction_fun = min_count(np.std, min_slices, ddof=1, axis=0)
            case "variation" | "cvar":
                reduction_fun = min_count(coefficient_variation, min_slices)
            case _:
                raise NameError("Invalid mode selected.")
        all_slices = sliced_brain.concat_slices(densities=True)
        redux = all_slices.groupby(all_slices.index)\
                            .apply(reduction_fun)\
                            .dropna(axis=0, how="all") # we want to keep float64 as the dtype, since the result of the 'mode' function is a float as well
        return redux

    def density(self) -> Self:
        assert self.mode == "sum", "Cannot compute densities for AnimalBrains whose slices' cell count were not summed."
        markers_data = dict()
        for marker in self.markers:
            data = self.markers_data[marker] / self.areas
            data.units = f"{marker}/{self.areas.units}"
            markers_data[marker] = data
        return AnimalBrain(None, markers_data=markers_data)

    def percentage(self, marker: str) -> BrainData:
        assert self.mode == "sum", "Cannot compute percentages for AnimalBrains whose slices' cell count were not summed."
        if self.is_split:
            hems = ("L", "R")
        else:
            hems = (None,)
        markers_data = dict()
        for marker in self.markers:
            brainwide_cell_counts = sum((self.markers_data[marker].root(hem) for hem in hems))
            data = self.markers_data[marker] / brainwide_cell_counts
            data.units = f"{marker}/{marker} in root"
            markers_data[marker] = data
        return AnimalBrain(None, markers_data=markers_data)
    
    def relative_density(self, marker: str) -> BrainData:
        assert self.mode == "sum", "Cannot compute relative densities for AnimalBrains whose slices' cell count were not summed."
        if self.is_split:
            hems = ("L", "R")
        else:
            hems = (None,)
        markers_data = dict()
        for marker in self.markers:
            brainwide_area = sum((self.areas.root(hem) for hem in hems))
            brainwide_cell_counts = sum((self.markers_data[marker].root(hem) for hem in hems))
            data = (self.markers_data[marker] / self.markers_data["area"]) / (brainwide_cell_counts / brainwide_area)
            data.units = f"{marker} density/root {marker} density"
            markers_data[marker] = data
        return AnimalBrain(None, markers_data=markers_data)

    def overlap_markers(self, marker1: str, marker2: str) -> Self:
        if self.mode != "sum" and self.mode != "mean":
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
            overlaps[marker1] = (self.markers_data[both] / self.markers_data[marker1]).clip(upper=1)
            overlaps[marker1].metric = "%overlapping"
            overlaps[marker1].units = f"({marker1}+{marker2})/{m}"
        return AnimalBrain(None, markers_data=overlaps)

    def __simple_mode_name(self, mode: str) -> str:
        match mode:
            case "avg":
                return "mean"
            case "variation":
                return "cvar"
            case "overlap":
                return "%overlapping"
            case _:
                return mode
    
    def to_pandas(self):
        data = pd.concat({"area": self.areas.data, **{m: m_data.data for m,m_data in self.markers_data.items()}}, axis=1)
        data.columns.name = self.mode
        return data

    def write_all_brains(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{self.name}_{self.mode}.csv")
        data = self.to_pandas()
        data.to_csv(output_path, sep="\t", mode="w", index_label=self.mode)
        print(f"AnimalBrain {self.name} reduced with mode='{self.mode}' saved to {output_path}")
    
    @staticmethod
    def from_pandas(animal_name, df: pd.DataFrame):
        mode = df.columns.name
        if "area" in df.columns:
            areas = BrainData(df["area"], animal_name, mode, None)
            df = df.loc[:, df.columns != "area"]
        else:
            areas = None
        markers_data = {marker: BrainData(data, animal_name, mode, None) for marker, data in df.items()}
        return AnimalBrain(None, markers_data=markers_data, areas=areas)

    @staticmethod
    def from_csv(animal_name, root_dir, mode):
        # read CSV
        df = pd.read_csv(os.path.join(root_dir, f"{animal_name}_{mode}.csv"), sep="\t", header=0, index_col=0)
        if df.index.name == "Class":
            # is old csv
            raise ValueError("Trying to read an AnimalBrain from an outdated formatted .csv. Please re-run the analysis from the SlicedBrain!")
        df.columns.name = df.index.name
        df.index.name = None
        return AnimalBrain.from_pandas(animal_name, df)

    @staticmethod
    def filter_selected_regions(animal_brain: Self, AllenBrain: AllenBrainHierarchy) -> Self:
        brain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.select_from_onthology(AllenBrain) for m, m_data in brain.markers_data.items()}
        return brain

    @staticmethod
    def merge_hemispheres(animal_brain: Self) -> Self:
        brain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.merge_hemispheres() for m, m_data in brain.markers_data.items()}
        return brain