import os
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import pearsonr

from .brain_hierarchy import AllenBrainHierarchy
from .animal_brain import AnimalBrain
from .utils import save_csv

class AnimalGroup:
    def __init__(self, name: str, \
                animals: list[AnimalBrain]=None, AllenBrain: AllenBrainHierarchy=None, \
                marker: str=None, data:pd.DataFrame=None, \
                hemisphere_distinction=False) -> None:
        self.name = name
        if marker is not None and data is not None:
            self.marker = marker
            self.data = data
            self.n = len(self.get_animals())
            return
        elif not animals or not AllenBrain:
            raise ValueError("You must specify the AnimalBrain list and the AllenBrainHierarchy.")
        assert all([brain.mode == "sum" for brain in animals]), "Can't normalize AnimalBrains whose slices' cell count were not summed."
        assert len(animals) > 0, "Inside the group there must be at least one animal."
        self.marker = animals[0].marker
        assert all([brain.marker == self.marker for brain in animals]), "All AnimalBrain composing the group must use the same marker."
        if not hemisphere_distinction:
            animals = [AnimalBrain.merge_hemispheres(animal_brain) for animal_brain in animals]
        self.data = self.normalize_animals(animals, AllenBrain)
        self.n = len(self.get_animals())
        
    
    def normalize_animals(self, animals, AllenBrain) -> pd.DataFrame:
        '''
        returns a DataFrame where, for each region and for each animal, gives:
        - the percentage
        - the density
        - the relative density
        If a brain region is not present in (one/any) animal, it fills every value with NaN

        NOTE: The brain regions are sorted by Breadth-First in the AllenBrain hierarchy
        '''
        all_animals = pd.concat({
                                    brain.name: pd.concat(
                                        (
                                            brain.data["area"],
                                            AnimalGroup.normalize_animal(brain, self.marker)
                                        ),
                                        axis=1)
                                    for brain in animals
                                }, join="outer")
        all_animals = all_animals.reorder_levels([1,0], axis=0)
        ordered_indices = product(AllenBrain.list_all_subregions("root", mode="depth"), [animal.name for animal in animals])
        return all_animals.reindex(ordered_indices, fill_value=np.nan)
    
    @staticmethod
    def normalize_animal(animal_brain, marker) -> AnimalBrain:
        '''
        Do normalization of the cell counts for one marker.
        The marker can be any column name of brain_df, e.g. "CFos".
        The output will be a dataframe with three columns: "Density", "Percentage" and "RelativeDensity".
        Each row is one of the original brain regions
        '''
            
        # Init dataframe
        columns = ["Density","Percentage","RelativeDensity"]
        norm_cell_counts = pd.DataFrame(np.nan, index=animal_brain.data.index, columns=columns)

        # Get the the brainwide area and cell counts (corresponding to the root)
        brainwide_area = animal_brain.data["area"]["root"]
        brainwide_cell_counts = animal_brain.data[marker]["root"]
            
        # Do the normalization for each column seperately.
        norm_cell_counts["Density"] = animal_brain.data[marker] / animal_brain.data["area"]
        norm_cell_counts["Percentage"] = animal_brain.data[marker] / brainwide_cell_counts 
        norm_cell_counts["RelativeDensity"] = (animal_brain.data[marker] / animal_brain.data["area"]) / (brainwide_cell_counts / brainwide_area)

        return norm_cell_counts
    
    def get_normalization_methods(self):
        return [col for col in self.data.columns if col != "area"]
    
    def get_normalized_data(self, normalization: str, regions: list[str]=None):
        assert normalization in self.get_normalization_methods(), f"Invalid normalization method '{normalization}'"
        if not regions:
            # we have to reindex the columns (brain regions) because of this bug:
            # https://github.com/pandas-dev/pandas/issues/15105
            regions_index = self.data.index.get_level_values(0).unique()
            return self.data[normalization].unstack(level=0).reindex(regions_index, axis=1)
        else:
            # if selecting a subset of regions first, the ording is retained
            return self.select(regions)[normalization].unstack(level=0)

    
    def get_animals(self):
        return {index[1] for index in self.data.index}
    
    def get_all_regions(self):
        return self.data.index.get_level_values(0)
    
    def get_regions(self):
        return list(self.get_all_regions().unique())
    
    def is_comparable(self, other) -> bool:
        if type(other) != AnimalGroup:
            return False
        return self.marker == other.marker and \
                set(self.get_regions()) == set(other.get_regions())
    
    def select(self, selected_regions: list[str], animal=None) -> pd.DataFrame:
        if animal is None:
            animal = list(self.get_animals())
        # return self.data.loc(axis=0)[selected_regions, animal].reset_index(level=1, drop=True)
        return self.data.loc(axis=0)[selected_regions, animal]
    
    def remove_smaller_subregions(self, area_threshold, selected_regions: list[str], AllenBrain: AllenBrainHierarchy) -> None:
        for animal in self.get_animals():
            self.remove_smaller_subregions_in_animal(area_threshold, animal, selected_regions, AllenBrain)

    def remove_smaller_subregions_in_animal(self, area_threshold, animal,
                                            selected_regions: list[str],
                                            AllenBrain: AllenBrainHierarchy) -> None:
        animal_areas = self.select(selected_regions, animal=animal)["area"]
        small_regions = [smaller_region for small_region    in animal_areas.index[animal_areas <= area_threshold].get_level_values(0)
                                        for smaller_region  in AllenBrain.list_all_subregions(small_region)]
        self.data.loc(axis=0)[small_regions, animal] = np.nan
    
    def group_by_region(self, method=None):
        if method is None:
            # pd.DataFrame
            data = self.data      
        else:
            # pd.Series
            data = self.data[method]
        return data.groupby(self.get_all_regions())
    
    def cross_correlation(self, normalization: str, regions: list[str]=None, min_animals=2) -> pd.DataFrame:
        assert not min_animals or (min_animals >= 2), "Invalid minimum number of animals needed for cross correlation. It must be >= 2."
        normalized_data = self.get_normalized_data(normalization, regions)
        if not min_animals:
            min_animals = len(normalized_data)
        r = normalized_data.corr(method=lambda x,y: pearsonr(x,y)[0], min_periods=min_animals)
        p = normalized_data.corr(method=lambda x,y: pearsonr(x,y)[1], min_periods=min_animals)
        return r, p
    
    def get_units(self, normalization):
        match normalization:
            case "Density":
                return f"#{self.marker}/mm²"
            case "Percentage" | "RelativeDensity":
                return "" # both are percentages between 0 and 1
            case _:
                raise ValueError(f"Normalization methods available are: {', '.join(self.get_normalization_methods())}")
    
    def get_plot_title(self, normalization):
        match normalization:
            case "Density":
                return f"[#{self.marker} / area]"
            case "Percentage":
                return f"[#{self.marker} / brain]"
            case "RelativeDensity":
                return f"[#{self.marker} / area] / [{self.marker} (brain) / area (brain)]"
            case _:
                raise ValueError(f"Normalization methods available are: {', '.join(self.get_normalization_methods())}")
    
    def to_csv(self, output_path, file_name, overwrite=False) -> None:
        saved_data = self.data.copy()
        saved_data.columns = pd.MultiIndex.from_product([[self.marker], self.data.columns])
        save_csv(saved_data, output_path, file_name, overwrite=overwrite)
    
    @staticmethod
    def from_csv(group_name, root_dir, file_name):
        # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=[0, 1], index_col=[0,1])
        # retrieve marker name
        markers = list({cols[0] for cols in df.columns})
        assert len(markers) == 1, "The CSVs are expected to have data for one marker only."
        marker = markers[0]
        return AnimalGroup(group_name, marker=marker, data=df.xs(marker, axis=1, drop_level=True))