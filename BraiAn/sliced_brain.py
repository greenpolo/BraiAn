import os
import pandas as pd
import copy

from .brain_hierarchy import AllenBrainHierarchy
from .brain_slice import BrainSlice, merge_slice_hemispheres,\
                        BrainSliceFileError, \
                        ExcludedRegionsNotFoundError, \
                        EmptyResultsError, \
                        InvalidResultsError, \
                        MissingResultsColumnError, \
                        InvalidRegionsHemisphereError, \
                        InvalidExcludedRegionsHemisphereError

class EmptyBrainError(Exception): pass

class SlicedBrain:
    def __init__(self, name: str, animal_dir: str, AllenBrain: AllenBrainHierarchy,
                area_key: str, tracer_key: str, marker_key, area_units="µm2") -> None:
        self.name = name
        self.marker = marker_key
        excluded_regions_dir = os.path.join(animal_dir, "regions_to_exclude")
        csv_slices_dir = os.path.join(animal_dir, "results")
        images = self.get_image_names_in_folder(csv_slices_dir)
        self.slices = []
        for image in images:
            results_file = os.path.join(csv_slices_dir, f"{image}_regions.txt")
            regions_to_exclude_file = os.path.join(excluded_regions_dir, f"{image}_regions_to_exclude.txt")
            try:
                slice = BrainSlice(AllenBrain,
                                    results_file,
                                    regions_to_exclude_file,
                                    self.name, image,
                                    area_key, tracer_key, self.marker, area_units=area_units)
            except ExcludedRegionsNotFoundError as e:
                mode = "print"
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            except EmptyResultsError as e:
                mode = "print"
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            except InvalidResultsError as e:
                mode = "print"
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            except MissingResultsColumnError as e:
                mode = "print"
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            except InvalidRegionsHemisphereError as e:
                mode = "print"
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            except InvalidExcludedRegionsHemisphereError as e:
                mode = "print"
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            else:
                self.slices.append(slice)
        if len(self.slices) == 0:
            raise EmptyBrainError(self.name)
    
    def get_image_names_in_folder(self, path) -> list[str]:
        all_files = os.listdir(path)
        images = list({file.replace("_regions.txt", "") for file in all_files if "_regions.txt" in file})
        #csv_files = [file.replace("_regions.csv", "") for file in all_files if "_regions.csv" in file]
        images.sort()
        return images
    
    def add_density(self) -> None:
        for brain_slice in self.slices:
            brain_slice.add_density()
    
    def concat_slices(self) -> pd.DataFrame:
        return pd.concat([slice.data for slice in self.slices])
    
    def handle_brainslice_error(self, exception, mode, results_file, regions_to_exclude_file):
        assert issubclass(type(exception), BrainSliceFileError), ""
        match mode:
            case "error":
                raise exception
            case "print":
                print(exception)
            case "delete":
                print(exception, "\nRemoving the corresponding result and regions_to_exclude files.")
                os.remove(results_file)
                if type(exception) != ExcludedRegionsNotFoundError:
                    os.remove(regions_to_exclude_file)
            case _:
                raise ValueError("Invalid 'mode' parameter. Supported BrainSliceFileError handling modes: 'error', 'print', 'delete'.")
    
def merge_sliced_hemispheres(sliced_brain) -> SlicedBrain:
    brain = copy.copy(sliced_brain)
    brain.slices = [merge_slice_hemispheres(brain_slice) for brain_slice in brain.slices]
    return brain