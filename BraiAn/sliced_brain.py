import os
import pandas as pd
import copy

from .brain_hierarchy import AllenBrainHierarchy
from .brain_slice import BrainSlice, merge_slice_hemispheres,\
                        BrainSliceFileError, \
                        ExcludedRegionsNotFoundError, \
                        EmptyResultsError, \
                        NanResultsError, \
                        InvalidResultsError, \
                        MissingResultsColumnError, \
                        InvalidRegionsHemisphereError, \
                        InvalidExcludedRegionsHemisphereError

global MODE_ExcludedRegionsNotFoundError
global MODE_EmptyResultsError
global MODE_NanResultsError
global MODE_InvalidResultsError
global MODE_MissingResultsColumnError
global MODE_InvalidRegionsHemisphereError
global MODE_InvalidExcludedRegionsHemisphereError
MODE_ExcludedRegionsNotFoundError = "print"
MODE_EmptyResultsError = "print"
MODE_NanResultsError = "print"
MODE_InvalidResultsError = "print"
MODE_MissingResultsColumnError = "print"
MODE_InvalidRegionsHemisphereError = "print"
MODE_InvalidExcludedRegionsHemisphereError = "print"

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
            except BrainSliceFileError as e:
                mode = self.get_default_error_mode(e)
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
            case "delete":
                print(exception, "\nRemoving the corresponding result and regions_to_exclude files.")
                os.remove(results_file)
                if type(exception) != ExcludedRegionsNotFoundError:
                    os.remove(regions_to_exclude_file)
            case "error":
                raise exception
            case "print":
                print(exception)
            case "silent":
                pass
            case _:
                raise ValueError(f"Invalid mode='{mode}' parameter. Supported BrainSliceFileError handling modes: 'delete', 'error', 'print', 'silent'.")
    
    def get_default_error_mode(self, exception):
        e_name = type(exception).__name__
        mode_var = f"MODE_{e_name}"
        if mode_var in globals():
            return globals()[mode_var]

        match type(exception):
            case ExcludedRegionsNotFoundError.__class__:
                return "print"
            case EmptyResultsError.__class__:
                return "print"
            case NanResultsError.__class__:
                return "print"
            case InvalidResultsError.__class__:
                return "print"
            case MissingResultsColumnError.__class__:
                return "print"
            case InvalidRegionsHemisphereError.__class__:
                return "print"
            case InvalidExcludedRegionsHemisphereError.__class__:
                return "print"
            case _:
                ValueError(f"Undercognized exception: {type(exception)}")

    
def merge_sliced_hemispheres(sliced_brain) -> SlicedBrain:
    brain = copy.copy(sliced_brain)
    brain.slices = [merge_slice_hemispheres(brain_slice) for brain_slice in brain.slices]
    return brain