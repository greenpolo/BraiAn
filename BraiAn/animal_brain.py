import os
import numpy as np
import pandas as pd
import copy

from .sliced_brain import merge_sliced_hemispheres, SlicedBrain
from .brain_slice import find_region_abbreviation

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
                name=None, min_slices=0, data: pd.DataFrame=None,
                use_literature_reuniens=False) -> None:
        if name and data is not None:
            self.name = name
            self.markers = data.columns[2:]
            self.mode = mode
            self.data = data
            return
        if not hemisphere_distinction:
            sliced_brain = merge_sliced_hemispheres(sliced_brain)
        if mode == "sum":
            self.data = self.sum_slices(sliced_brain, min_slices)
        else:
            sliced_brain.add_density()
            self.data = self.reduce_brain_densities(sliced_brain, mode, min_slices)
        if use_literature_reuniens:
            self.__add_literature_reuniens(sliced_brain.name, hemisphere_distinction)
        self.name = sliced_brain.name
        self.markers = sliced_brain.markers
        self.mode = mode
    
    def __add_literature_reuniens(self, animal, hemisphere_distinction):
        # subregions_ = ("PR", "RE", "Xi", "RH")
        subregions_ = ("RE", "Xi", "RH")
        if not hemisphere_distinction:
            subregions = self.data.index.intersection(subregions_)
            if len(subregions) == len(subregions_):
                self.data.loc[f"REtot"] = self.data.loc[subregions].sum()
                # self.data.drop(subregions, inplace=True)
            else:
                print(f"WARNING: Animal '{animal}' - could not find data for computing the 'REtot' region. Missing {', '.join(set(subregions_) - set(subregions))}.")
            return
        # if hemisphere_distinction:
        for hem in ("Left", "Right"):
            hem_subregions_ = [f"{hem}: {subregion}" for subregion in subregions_]
            hem_subregions = self.data.index.intersection(hem_subregions_)
            if len(hem_subregions) == len(subregions_):
                # sum with int and float columns get casted to all-float.
                # a possible fix is to add a 'string' place-holder column, sum and then remove it
                # see:
                # https://github.com/pandas-dev/pandas/issues/26219#issuecomment-905882647
                re_tot_data = self.data.loc[hem_subregions]
                re_tot_data["place_holder"] = ""
                re_tot_data = re_tot_data.sum()
                re_tot_data.drop(labels="place_holder", inplace=True)
                self.data.loc[f"{hem}: REtot"] = re_tot_data
                # self.data.drop(hem_subregions, inplace=True)
            else:
                print(f"WARNING: Animal '{animal}' - could not find data for computing the 'REtot' region. Missing {', '.join(set(hem_subregions_) - set(hem_subregions))}.")
    
    def sum_slices(self, sliced_brain: SlicedBrain, min_slices: int) -> pd.DataFrame:
        all_slices = sliced_brain.concat_slices()
        return all_slices.groupby(all_slices.index)\
                            .sum(min_count=min_slices)\
                            .dropna(axis=0, how="all")\
                            .astype({m: sliced_brain.get_marker_dtype(m) for m in sliced_brain.markers}) # dropna() changes type to float64

    def reduce_brain_densities(self, sliced_brain: SlicedBrain, mode: str, min_slices: int) -> pd.Series:
        match mode:
            case "mean" | "avg":
                reduction_fun = min_count(np.mean, min_slices, axis=0)
            case "std":
                reduction_fun = min_count(np.std, min_slices, ddof=1, axis=0)
            case "variation" | "cvar":
                reduction_fun = min_count(coefficient_variation, min_slices)
            case _:
                raise NameError("Invalid mode selected.")
        # sliced_brain.add_density() must have been called previously!
        all_slices = sliced_brain.concat_slices()[[f"{m}_density" for m in sliced_brain.markers]]
        return all_slices.groupby(all_slices.index)\
                            .apply(reduction_fun)\
                            .dropna(axis=0, how="all") # we want to keep float64 as the dtype, since the result of the 'mode' function is a float as well

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
        brain.data = animal_brain.data.groupby(corresponding_region).sum(min_count=1)
        return brain