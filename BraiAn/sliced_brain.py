import copy
import numpy as np
import os
import pandas as pd
import platform
import re
from typing import Self

from . import resolve_symlink
from .brain_hierarchy import AllenBrainHierarchy
from .brain_slice import BrainSlice,\
                        BrainSliceFileError, \
                        ExcludedAllRegionsError, \
                        ExcludedRegionsNotFoundError, \
                        EmptyResultsError, \
                        NanResultsError, \
                        InvalidResultsError, \
                        MissingResultsColumnError, \
                        InvalidRegionsHemisphereError, \
                        InvalidExcludedRegionsHemisphereError

global MODE_ExcludedAllRegionsError
global MODE_ExcludedRegionsNotFoundError
global MODE_EmptyResultsError
global MODE_NanResultsError
global MODE_InvalidResultsError
global MODE_MissingResultsColumnError
global MODE_InvalidRegionsHemisphereError
global MODE_InvalidExcludedRegionsHemisphereError
MODE_ExcludedAllRegionsError = "print"
MODE_ExcludedRegionsNotFoundError = "print"
MODE_EmptyResultsError = "print"
MODE_NanResultsError = "print"
MODE_InvalidResultsError = "print"
MODE_MissingResultsColumnError = "print"
MODE_InvalidRegionsHemisphereError = "print"
MODE_InvalidExcludedRegionsHemisphereError = "print"

class EmptyBrainError(Exception): pass

class SlicedBrain:
    def __init__(self, name: str, animal_dir: str, brain_onthology: AllenBrainHierarchy,
                area_key: str, tracers_key, markers_key, *overlapping_tracers: list[int], area_units="µm2",
                exclude_parent_regions=False) -> None:
        self.name = name
        if not isinstance(tracers_key, str) and len(overlapping_tracers) > 0:
            # QuPath specific
            qupath_channels = [remove_qupath_num(t) for t in tracers_key]
            overlapping_channels = get_overlapping_keys(qupath_channels, overlapping_tracers)
            overlapping_detections = [f"Num {c1}~{c2}" for c1,c2 in overlapping_channels]
            assert all([o not in tracers_key for o in overlapping_detections]), f"You don't have to specify the columns of the overlapping detections {overlapping_detections}.\n"+\
            "Just pass the indices of the markers you want to do an overlapping comparison with."
            tracers_key = copy.copy(tracers_key) + overlapping_detections
        self.markers = [markers_key] if isinstance(markers_key, str) else copy.copy(markers_key)
        self.markers += [f"{m1}+{m2}" for m1,m2 in get_overlapping_keys(self.markers, overlapping_tracers)]
        excluded_regions_dir = os.path.join(animal_dir, "regions_to_exclude")
        csv_slices_dir = os.path.join(animal_dir, "results")
        images = self.get_image_names_in_folder(csv_slices_dir)
        self.slices: list[BrainSlice] = []
        for image in images:
            results_file = os.path.join(csv_slices_dir, f"{image}_regions.txt")
            if not os.path.exists(results_file) and platform.system() == "Windows":
                results_file += ".lnk"
            regions_to_exclude_file = os.path.join(excluded_regions_dir, f"{image}_regions_to_exclude.txt")
            if not os.path.exists(regions_to_exclude_file) and platform.system() == "Windows":
                regions_to_exclude_file += ".lnk"
            try:
                slice = BrainSlice(brain_onthology,
                                    resolve_symlink(results_file),
                                    resolve_symlink(regions_to_exclude_file), exclude_parent_regions,
                                    self.name, image,
                                    area_key, tracers_key, self.markers, area_units=area_units)
            except BrainSliceFileError as e:
                mode = self.get_default_error_mode(e)
                self.handle_brainslice_error(e, mode, results_file, regions_to_exclude_file)
            else:
                self.slices.append(slice)
        if len(self.slices) == 0:
            raise EmptyBrainError(self.name)
        are_split = np.array([s.is_split for s in self.slices])
        assert are_split.all() or ~are_split.any(), "Slices from the same animal should either be ALL split between right/left hemisphere or not."
        self.is_split = are_split[0]
    
    def set_name(self, name):
        for slice in self.slices:
            slice.name = name
        self.name = name
    
    def get_marker_dtype(self, marker):
        assert marker in self.markers, f"Missing marker ('{marker}')!"
        return self.slices[0].data[marker].dtype
    
    def get_image_names_in_folder(self, path) -> list[str]:
        images = list({re.sub('_regions.txt[.lnk]*', '', file) for file in os.listdir(path)})
        # images = list({re.sub('_regions.csv[.lnk]*', '', file) for file in os.listdir(path)}) # csv_files
        images.sort()
        return images
    
    def concat_slices(self, densities=False) -> pd.DataFrame:
        return pd.concat([slice.data if not densities else
                          pd.concat((slice.data["area"], slice.markers_density), axis=1)
                          for slice in self.slices])
    
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
            case ExcludedAllRegionsError.__class__:
                return "print"
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

    @staticmethod
    def merge_hemispheres(sliced_brain) -> Self:
        if not sliced_brain.is_split:
            return sliced_brain
        brain = copy.copy(sliced_brain)
        brain.slices = [BrainSlice.merge_hemispheres(brain_slice) for brain_slice in brain.slices]
        brain.is_split = False
        return brain

def remove_qupath_num(detection_key: str) -> str:
    return re.compile("Num (.+)").findall(detection_key)[0]

def get_overlapping_keys(values: list[str], overlapping_tracers: list[int]) -> list[str]:
    assert all([len(idx) == 2 for idx in overlapping_tracers]), "Overlapping marker analyisis is supported only between two markers!"
    return [(values[i1], values[i2]) for i1,i2 in overlapping_tracers]