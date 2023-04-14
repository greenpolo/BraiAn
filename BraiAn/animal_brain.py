import os
import numpy as np
import pandas as pd
import copy

from .sliced_brain import merge_sliced_hemispheres, SlicedBrain
from .brain_slice import find_region_abbreviation

def min_count(fun, min):
    def nan_if_less(xs):
        if len(xs) >= min:
            return fun(xs)
        else:
            return np.NaN
    return nan_if_less

# https://en.wikipedia.org/wiki/Coefficient_of_variation
def coefficient_variation(x) -> np.float64:
    avg = x.mean()
    if len(x) > 1 and avg != 0:
        return x.std(ddof=1) / avg
    else:
        return 0

class AnimalBrain:
    def __init__(self, sliced_brain, mode="sum", hemisphere_distinction=True,
                name=None, min_slices=0, data: pd.DataFrame=None) -> None:
        if name and data is not None:
            self.name = name
            self.marker = data.columns[-1]
            self.mode = mode
            self.data = data
            return
        if not hemisphere_distinction:
            sliced_brain = merge_sliced_hemispheres(sliced_brain)
        if mode == "sum": # data is a pd.DataFrame
            self.data = self.sum_slices(sliced_brain, min_slices)
        else: # data is a pd.Series
            sliced_brain.add_density()
            self.data = self.reduce_brain_densities(sliced_brain, mode, min_slices)
        self.name = sliced_brain.name
        self.marker = sliced_brain.marker
        self.mode = mode
    
    def sum_slices(self, sliced_brain: SlicedBrain, min_slices: int) -> pd.DataFrame:
        all_slices = sliced_brain.concat_slices()
        return all_slices.groupby(all_slices.index, axis=0)\
                            .sum(min_count=min_slices)\
                            .dropna(axis=0, how="all")\
                            .astype({sliced_brain.marker: sliced_brain.get_marker_dtype()}) # dropna() changes type to float64

    def reduce_brain_densities(self, sliced_brain: SlicedBrain, mode: str, min_slices: int) -> pd.Series:
        match mode:
            case "mean" | "avg":
                reduction_fun = min_count(np.mean, min_slices)
            case "std":
                reduction_fun = min_count(np.std, min_slices)
            case "variation" | "cvar":
                reduction_fun = min_count(coefficient_variation, min_slices)
            case _:
                raise NameError("Invalid mode selected.")
        all_slices = sliced_brain.concat_slices()[f"{sliced_brain.marker}_density"]
        return all_slices.groupby(all_slices.index, axis=0)\
                            .apply(reduction_fun)\
                            .dropna() # we want to keep float64 as the dtype, since the result of the 'mode' function a float as well

    def write_all_brains(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{self.name}_{self.mode}.csv")
        self.data.to_csv(output_path, sep="\t", mode="w")
        print(f"AnimalBrain {self.name} reduced with mode='{self.mode}' saved to {output_path}")
    
    @staticmethod
    def from_csv(animal_name, root_dir, mode):
        # read CSV
        df = pd.read_csv(os.path.join(root_dir, f"{animal_name}_{mode}.csv"), sep="\t", header=0, index_col=0)
        return AnimalBrain(None, mode=mode, name=animal_name, data=df)

    @staticmethod
    def filter_selected_regions(animal_brain, AllenBrain): # -> AnimalBrain:
        brain = copy.copy(animal_brain)
        selected_allen_regions = AllenBrain.get_selected_regions()
        selectable_regions = set(animal_brain.data.index).intersection(set(selected_allen_regions))
        if type(brain.data) == pd.Series:
            brain.data = animal_brain.data[list(selectable_regions)]
        else: # type == pd.DataFrame
            brain.data = animal_brain.data.loc[list(selectable_regions), :]
        return brain

    @staticmethod
    def merge_hemispheres(animal_brain): # -> AnimalBrain:
        brain = copy.copy(animal_brain)
        corresponding_region = [find_region_abbreviation(region) for region in animal_brain.data.index]
        brain.data = animal_brain.data.groupby(corresponding_region, axis=0).sum(min_count=1)
        return brain