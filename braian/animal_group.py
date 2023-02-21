import os
import numpy as np
import pandas as pd
from itertools import product

from .brain_hierarchy import AllenBrainHierarchy
from .animal_brain import AnimalBrain, merge_hemispheres

class AnimalGroup:
    def __init__(self, name: str, animals: list[AnimalBrain], AllenBrain: AllenBrainHierarchy,
                hemisphere_distinction=False) -> None:
        assert all([brain.mode == "sum" for brain in animals]), "Can't normalize AnimalBrains whose slices' cell count were not summed."
        assert len(animals) > 0, "Inside the group there must be at least one animal."
        self.marker = animals[0].marker
        assert all([brain.marker == self.marker for brain in animals]), "All AnimalBrain composing the group must use the same marker."
        if not hemisphere_distinction:
            animals = [merge_hemispheres(animal_brain) for animal_brain in animals]
        self.name = name
        self.data = pd.concat({brain.name: self.normalize_cell_counts(brain, self.marker) for brain in animals})
        self.data = pd.concat({self.marker: self.data}, axis=1)
        self.data = self.data.reorder_levels([1,0], axis=0)
        ordered_indices = product(AllenBrain.brain_region_dict.keys(), [animal.name for animal in animals])
        self.data = self.data.reindex(ordered_indices, fill_value=np.nan)
    
    def normalize_cell_counts(self, animal_brain, tracer) -> AnimalBrain:
        '''
        Do normalization of the cell counts for one tracer.
        The tracer can be any column name of brain_df, e.g. 'CFos'.
        The output will be a dataframe with three columns: 'Density', 'Percentage' and 'RelativeDensity'.
        Each row is one of the original brain regions
        '''
            
        # Init dataframe
        columns = ['Density','Percentage','RelativeDensity']
        norm_cell_counts = pd.DataFrame(np.nan, index=animal_brain.data.index, columns=columns)

        # Get the the brainwide area and cell counts (corresponding to the root)
        brainwide_area = animal_brain.data["area"]["root"]
        brainwide_cell_counts = animal_brain.data[tracer]["root"]
            
        # Do the normalization for each column seperately.
        norm_cell_counts['Density'] = animal_brain.data[tracer] / animal_brain.data["area"]
        norm_cell_counts['Percentage'] = animal_brain.data[tracer] / brainwide_cell_counts 
        norm_cell_counts['RelativeDensity'] = (animal_brain.data[tracer] / animal_brain.data["area"]) / (brainwide_cell_counts / brainwide_area)

        return norm_cell_counts
    
    def save(self, output_path, filename) -> bool:
        if not(os.path.exists(output_path)):
            os.mkdir(output_path)
            print('\nCreated a new results_python folder '+output_path+' \n')
        else:
            print('\n! A results_python folder already existed in root. I am overwriting previous results!\n')
        self.data.to_csv(os.path.join(output_path, filename), sep='\t', mode='w')

        print(f"Results are saved in {output_path}")
        return True