import os
import numpy as np
import pandas as pd
from itertools import product

from .brain_hierarchy import AllenBrainHierarchy
from .animal_brain import AnimalBrain, merge_hemispheres
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
            return
        elif not animals or not AllenBrain:
            raise ValueError("You must specify the AnimalBrain list and the AllenBrainHierarchy.")
        assert all([brain.mode == "sum" for brain in animals]), "Can't normalize AnimalBrains whose slices' cell count were not summed."
        assert len(animals) > 0, "Inside the group there must be at least one animal."
        self.marker = animals[0].marker
        assert all([brain.marker == self.marker for brain in animals]), "All AnimalBrain composing the group must use the same marker."
        if not hemisphere_distinction:
            animals = [merge_hemispheres(animal_brain) for animal_brain in animals]
        self.data = self.normalize_animals(animals, AllenBrain)
        
    
    def normalize_animals(self, animals, AllenBrain) -> pd.DataFrame:
        '''
        returns a DataFrame where, for each region and for each animal, gives:
        - the percentage
        - the density
        - the relative density
        If a brain region is not present in (one/any) animal, it fills every value with NaN

        NOTE: The brain regions are sorted by Breadth-First in the AllenBrain hierarchy
        '''
        all_animals = pd.concat({brain.name: self.normalize_animal(brain, self.marker) for brain in animals})
        all_animals = pd.concat({self.marker: all_animals}, axis=1)
        all_animals = all_animals.reorder_levels([1,0], axis=0)
        ordered_indices = product(AllenBrain.full_name.keys(), [animal.name for animal in animals])
        return all_animals.reindex(ordered_indices, fill_value=np.nan)
    
    def normalize_animal(self, animal_brain, tracer) -> AnimalBrain:
        '''
        Do normalization of the cell counts for one tracer.
        The tracer can be any column name of brain_df, e.g. "CFos".
        The output will be a dataframe with three columns: "Density", "Percentage" and "RelativeDensity".
        Each row is one of the original brain regions
        '''
            
        # Init dataframe
        columns = ["Density","Percentage","RelativeDensity"]
        norm_cell_counts = pd.DataFrame(np.nan, index=animal_brain.data.index, columns=columns)

        # Get the the brainwide area and cell counts (corresponding to the root)
        brainwide_area = animal_brain.data["area"]["root"]
        brainwide_cell_counts = animal_brain.data[tracer]["root"]
            
        # Do the normalization for each column seperately.
        norm_cell_counts["Density"] = animal_brain.data[tracer] / animal_brain.data["area"]
        norm_cell_counts["Percentage"] = animal_brain.data[tracer] / brainwide_cell_counts 
        norm_cell_counts["RelativeDensity"] = (animal_brain.data[tracer] / animal_brain.data["area"]) / (brainwide_cell_counts / brainwide_area)

        return norm_cell_counts
    
    def get_normalization_methods(self):
        return self.data.columns.get_level_values(1).to_list()
    
    def get_animals(self):
        return {index[1] for index in self.data.index}
    
    def get_all_regions(self):
        return self.data.index.get_level_values(0).to_list()
    
    def get_regions(self):
        return set(self.get_all_regions())
    
    def is_comparable(self, other) -> bool:
        if type(other) != AnimalGroup:
            return False
        return self.marker == other.marker and \
                self.get_regions() == other.get_regions()
    
    def select(self, selected_regions: list[str], animal=None) -> pd.DataFrame:
        if animal is None:
            animal = list(self.get_animals())
        return self.data.loc(axis=0)[selected_regions, animal].reset_index(level=1, drop=True)[self.marker]
    
    def group_by_region(self, col=None):
        if col is None:
            # pd.DataFrame
            data = self.data      
        else:
            # pd.Series
            data = self.data[self.marker, col]
        return data.groupby(self.get_all_regions())
    
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
        save_csv(self.data, output_path, file_name, overwrite=overwrite)
    
    @staticmethod
    def from_csv(group_name, root_dir, file_name):
        # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=[0, 1], index_col=[0,1])
        # retrieve marker name
        markers = list({cols[0] for cols in df.columns})
        assert len(markers) == 1, "The CSVs are expected to have data for one marker only."
        marker = markers[0]
        return AnimalGroup(group_name, marker=marker, data=df)