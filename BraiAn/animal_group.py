import os
import numpy as np
import pandas as pd
from itertools import product

from .brain_hierarchy import AllenBrainHierarchy
from .animal_brain import AnimalBrain
from .utils import save_csv

class AnimalGroup:
    def __init__(self, name: str, \
                animals: list[AnimalBrain]=None, AllenBrain: AllenBrainHierarchy=None, \
                markers=None, data:pd.DataFrame=None, \
                hemisphere_distinction=False) -> None:
        self.name = name
        if markers is not None and data is not None:
            self.markers = [markers] if isinstance(markers, str) else markers
            self.data = data
            self.n = len(self.get_animals())
            return
        elif not animals or not AllenBrain:
            raise ValueError("You must specify the AnimalBrain list and the AllenBrainHierarchy.")
        assert all([brain.mode == "sum" for brain in animals]), "Can't normalize AnimalBrains whose slices' cell count were not summed."
        assert len(animals) > 0, "Inside the group there must be at least one animal."
        self.markers = animals[0].markers
        assert all([marker in self.markers for brain in animals for marker in brain.markers]), "All AnimalBrain composing the group must use the same markers."
        if hemisphere_distinction:
            raise NotImplementedError("AnimalGroup does not (yet) support split hemispheres!")
        animals = [AnimalBrain.merge_hemispheres(animal_brain) for animal_brain in animals]
        self.data = self.assemble_group(animals, AllenBrain)
        self.n = len(self.get_animals())

    def assemble_group(self, animals, AllenBrain) -> pd.DataFrame:
        '''
        returns a DataFrame where, for each region and for each animal, gives:
        - the percentage
        - the density
        - the relative density
        If a brain region is not present in (one/any) animal, it fills every value with NaN

        NOTE: The brain regions are sorted by Breadth-First in the AllenBrain hierarchy
        '''
        all_animals = dict()
        for brain in animals:
            all_columns = [AnimalGroup.normalize_animal(brain, marker) for marker in brain.markers]
            area_col = brain.data["area"].to_frame()
            area_col.columns = pd.MultiIndex.from_tuples([("area", "area")])
            all_animals[brain.name] = pd.concat([area_col]+all_columns, axis=1)
        all_animals = pd.concat(all_animals, join="outer") # dict to pd.DataFrame
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

        norm_cell_counts.columns = pd.MultiIndex.from_product([[marker], norm_cell_counts.columns])
        return norm_cell_counts

    def get_normalization_methods(self):
        return list(self.data[self.markers[0]].columns)

    def get_normalized_data(self, normalization: str, regions: list[str]=None, marker=None):
        assert normalization in self.get_normalization_methods(), f"Invalid normalization method '{normalization}'"
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get normalized data for marker '{marker}'!"
        if not regions:
            # we have to reindex the columns (brain regions) because of this bug:
            # https://github.com/pandas-dev/pandas/issues/15105
            regions_index = self.data.index.get_level_values(0).unique()
            return self.data[marker][normalization].unstack(level=0).reindex(regions_index, axis=1)
        else:
            # if selecting a subset of regions first, the ording is retained
            return self.select(regions)[marker][normalization].unstack(level=0)
    
    def get_animals(self):
        return {index[1] for index in self.data.index}

    def get_all_regions(self):
        return self.data.index.get_level_values(0)

    def get_regions(self):
        return list(self.get_all_regions().unique())

    def is_comparable(self, other) -> bool:
        if type(other) != AnimalGroup:
            return False
        return set(self.markers) == set(other.markers) and \
                set(self.get_regions()) == set(other.get_regions())

    def select(self, selected_regions: list[str], animal=None) -> pd.DataFrame:
        if animal is None:
            animal = list(self.get_animals())
        # return self.data.loc(axis=0)[selected_regions, animal].reset_index(level=1, drop=True)
        selected = self.data.loc(axis=0)[selected_regions, animal]
        if len(self.markers) == 0:
            selected[self.markers[0]]
        else:
            selected

    def remove_smaller_subregions(self, area_threshold, selected_regions: list[str], AllenBrain: AllenBrainHierarchy) -> None:
        for animal in self.get_animals():
            self.remove_smaller_subregions_in_animal(area_threshold, animal, selected_regions, AllenBrain)

    def remove_smaller_subregions_in_animal(self, area_threshold, animal,
                                            selected_regions: list[str],
                                            AllenBrain: AllenBrainHierarchy) -> None:
        animal_areas = self.select(selected_regions, animal=animal)["area"]["area"]
        small_regions = [smaller_region for small_region    in animal_areas.index[animal_areas <= area_threshold].get_level_values(0)
                                        for smaller_region  in AllenBrain.list_all_subregions(small_region)]
        self.data.loc(axis=0)[small_regions, animal] = np.nan

    def group_by_region(self, marker=None, method=None):
        if marker is None:
            data = self.data
        else:
            data = self.data[marker]
        if method is not None:
            # if marker != None -> pd.Series
            # if marker == None -> pd.DataFrame
            data = data[method]
        return data.groupby(level=0)

    def get_units(self, normalization, marker=None):
        if len(self.markers) == 0:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        match normalization:
            case "Density":
                return f"#{marker}/mm²"
            case "Percentage" | "RelativeDensity":
                return "" # both are percentages between 0 and 1
            case _:
                raise ValueError(f"Normalization methods available are: {', '.join(self.get_normalization_methods())}")

    def get_plot_title(self, normalization, marker=None):
        if len(self.markers) == 0:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get the plot title for marker '{marker}'!"
        match normalization:
            case "Density":
                return f"[#{marker} / area]"
            case "Percentage":
                return f"[#{marker} / brain]"
            case "RelativeDensity":
                return f"[#{marker} / area] / [{marker} (brain) / area (brain)]"
            case _:
                raise ValueError(f"Normalization methods available are: {', '.join(self.get_normalization_methods())}")

    def to_csv(self, output_path, file_name, overwrite=False) -> None:
        save_csv(self.data, output_path, file_name, overwrite=overwrite)

    @staticmethod
    def from_csv(group_name, root_dir, file_name):
        # read CSV
        df = pd.read_csv(os.path.join(root_dir, file_name), sep="\t", header=[0, 1], index_col=[0,1])
        # old CSVs
        if len(df.columns.get_level_values(0).unique()) == 1 and "area" in df.columns.get_level_values(1):
            df.columns = pd.MultiIndex.from_tuples([
                (marker, col) if col != "area" else ("area", "area")
                for (marker, col) in df.columns.to_list()
            ])
        # retrieve marker name
        markers = [col for col in df.columns.get_level_values(0).unique() if col != "area"]
        return AnimalGroup(group_name, markers=markers, data=df)