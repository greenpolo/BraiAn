import os
import numpy as np
import pandas as pd
import copy

from .sliced_brain import merge_sliced_hemispheres
from .brain_slice import find_region_abbreviation


# https://en.wikipedia.org/wiki/Coefficient_of_variation
def coefficient_variation(x) -> np.float64:
    avg = x.mean()
    if len(x) > 1 and avg != 0:
        return x.std(ddof=1) / avg
    else:
        return 0

class AnimalBrain:
    def __init__(self, sliced_brain, mode="sum", hemisphere_distinction=True) -> None:
        if not hemisphere_distinction:
            sliced_brain = merge_sliced_hemispheres(sliced_brain)
        if mode == "sum":
            self.data = self.sum_slices(sliced_brain)
        else:
            sliced_brain.add_density()
            self.data = self.reduce_brain_densities(sliced_brain, mode)
        self.name = sliced_brain.name
        self.marker = sliced_brain.marker
        self.mode = mode
    
    def sum_slices(self, sliced_brain) -> pd.DataFrame:
        all_slices = sliced_brain.concat_slices()
        return all_slices.groupby(all_slices.index, axis=0).sum()

    def reduce_brain_densities(self, sliced_brain, mode) -> pd.Series:
        match mode:
            case "mean" | "avg":
                reduction_fun = np.mean
            case "std":
                reduction_fun = np.std
            case "variation" | "cvar":
                reduction_fun = coefficient_variation
            case _:
                raise NameError("Invalid mode selected.")
        all_slices = sliced_brain.concat_slices()[f"{sliced_brain.marker}_density"]
        return all_slices.groupby(all_slices.index, axis=0).apply(reduction_fun)

    def write_all_brains(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, f"{self.name}_{self.mode}.csv")
        self.data.to_csv(output_path, sep="\t", mode="w")
        print(f"AnimalBrain {self.name} reduced with mode='{self.mode}' saved to {output_path}")

def filter_selected_regions(animal_brain, AllenBrain) -> AnimalBrain:
    brain = copy.copy(animal_brain)
    selected_allen_regions = AllenBrain.get_selected_regions()
    selectable_regions = set(animal_brain.data.index).intersection(set(selected_allen_regions))
    if type(brain.data) == pd.Series:
        brain.data = animal_brain.data[list(selectable_regions)]
    else: # type == pd.DataFrame
        brain.data = animal_brain.data.loc[list(selectable_regions), :]
    return brain

def merge_hemispheres(animal_brain) -> AnimalBrain:
    brain = copy.copy(animal_brain)
    corresponding_region = [find_region_abbreviation(region) for region in animal_brain.data.index]
    brain.data = animal_brain.data.groupby(corresponding_region, axis=0).sum(min_count=1)
    return brain