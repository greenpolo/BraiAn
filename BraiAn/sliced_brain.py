import os
import pandas as pd
import copy

from .brain_hierarchy import AllenBrainHierarchy
from .brain_slice import BrainSlice, merge_slice_hemispheres

class SlicedBrain:
    def __init__(self, name: str, animal_dir: str, AllenBrain: AllenBrainHierarchy,
                area_key: str, tracer_key: str, marker_key, area_units="µm2") -> None:
        self.name = name
        self.marker = marker_key
        excluded_regions_dir = os.path.join(animal_dir, 'regions_to_exclude')
        csv_slices_dir = os.path.join(animal_dir, "results")
        images = self.get_image_names_in_folder(csv_slices_dir)
        self.slices = [BrainSlice(AllenBrain,
                                    os.path.join(csv_slices_dir, f"{image}_regions.txt"),
                                    os.path.join(excluded_regions_dir, f"{image}_regions_to_exclude.txt"),
                                    image,
                                    area_key, tracer_key, self.marker, area_units=area_units)
                        for image in images]
    
    def get_image_names_in_folder(self, path) -> list[str]:
        all_files = os.listdir(path)
        images = list({file.replace("_regions.txt", "") for file in all_files if "_regions.txt" in file})
        #txt_files = [file.replace("_regions.csv", "") for file in all_files if "_regions.csv" in file]
        images.sort()
        return images
    
    def add_density(self) -> None:
        for brain_slice in self.slices:
            brain_slice.add_density()
    
    def concat_slices(self) -> pd.DataFrame:
        return pd.concat([slice.data for slice in self.slices])
    
def merge_sliced_hemispheres(sliced_brain) -> SlicedBrain:
    brain = copy.copy(sliced_brain)
    brain.slices = [merge_slice_hemispheres(brain_slice) for brain_slice in brain.slices]
    return brain