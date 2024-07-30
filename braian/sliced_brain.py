# SPDX-FileCopyrightText: 2024 Carlo Castoldi <carlo.castoldi@outlook.com>
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import copy
import numpy as np
import os
import pandas as pd
import re
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from braian.brain_hierarchy import AllenBrainHierarchy
from braian.brain_slice import BrainSlice,\
                        BrainSliceFileError, \
                        ExcludedAllRegionsError, \
                        ExcludedRegionsNotFoundError, \
                        EmptyResultsError, \
                        NanResultsError, \
                        InvalidResultsError, \
                        MissingResultsColumnError, \
                        InvalidRegionsHemisphereError, \
                        InvalidExcludedRegionsHemisphereError

__all__ = ["SlicedBrain"]

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
    __QUPATH_DIR_NAME_RESULTS = "results"
    __QUPATH_DIR_NAME_EXCLUSIONS = "regions_to_exclude"

    @staticmethod
    def from_qupath(name: str,
                    animal_dir: str|Path,
                    ch2marker: dict[str,str]|OrderedDict[str,str],
                    brain_ontology: AllenBrainHierarchy,
                    overlapping_markers: Iterable[int]=(),
                    area_units: str="µm2",
                    exclude_parent_regions: bool=False) -> Self:
        if not isinstance(animal_dir, Path):
            animal_dir = Path(animal_dir)
        csv_slices_dir = animal_dir / SlicedBrain.__QUPATH_DIR_NAME_RESULTS
        excluded_regions_dir = animal_dir / SlicedBrain.__QUPATH_DIR_NAME_EXCLUSIONS
        images = get_image_names_in_folder(csv_slices_dir)
        if len(overlapping_markers) > 0:
            assert len(overlapping_markers) == 2, "SlicedBrain currently supports overlapping at maximum between two markers from QuPath"
            assert isinstance(ch2marker, OrderedDict), "'ch2marker' should be an OrderedDict if `overlapping_markers` is specified"
            if len(ch2marker) >= 2 and len(overlapping_markers) > 0:
                overlapping_channels = qupath_class2overlap(ch2marker, [overlapping_markers])
                assert all([o not in ch2marker for o in overlapping_channels.keys()]), \
                    f"You don't have to specify the columns of the overlapping detections in 'ch2marker'. "+\
                        "Just specify the indices of the overlapping markers in 'overlapping_markers'."
                ch2marker = copy.copy(ch2marker) | overlapping_channels # we copy the dict, or we modify the structure for the caller as well
        slices: list[BrainSlice] = []
        for image in images:
            results_file = os.path.join(csv_slices_dir, f"{image}_regions.txt")
            excluded_regions_file = os.path.join(excluded_regions_dir, f"{image}_regions_to_exclude.txt")
            try:
                # Setting brain_ontology=None, we don't check that the data corresponds to real brain regions
                # we post-pone the check later in the analysis for performance reasons.
                # The assumption is that if you're creating a SlicedBrain, you will eventually do
                # group analysis. Checking against the ontology for each slice would be too time consuming.
                # We can do it afterwards, after the SlicedBrain is reduced to AnimalBrain
                slice: BrainSlice = BrainSlice.from_qupath(results_file,
                                               ch2marker.keys(), ch2marker.values(),
                                               animal=name, name=image, is_split=True,
                                               area_units=area_units, brain_ontology=None)
                exclude = BrainSlice.read_qupath_exclusions(excluded_regions_file)
                slice.exclude_regions(exclude, brain_ontology, exclude_parent_regions)
            except BrainSliceFileError as e:
                mode = SlicedBrain.__get_default_error_mode(e)
                SlicedBrain.__handle_brainslice_error(e, mode, name, results_file, excluded_regions_file)
            else:
                slices.append(slice)
        return SlicedBrain(name, slices, ch2marker.values())
        

    def __init__(self, name: str, slices: Iterable[BrainSlice], markers: Iterable[str]) -> None:
        self._name = name
        self.slices: list[BrainSlice] = list(slices)
        if len(self.slices) == 0:
            raise EmptyBrainError(self._name)
        self.markers = list(markers)
        are_split = np.array([s.is_split for s in self.slices])
        assert are_split.all() or ~are_split.any(), "Slices from the same animal should either be ALL split between right/left hemisphere or not."
        self.is_split = are_split[0]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        for slice in self.slices:
            slice.name = value
        self._name = value
    
    def get_marker_dtype(self, marker):
        assert marker in self.markers, f"Missing marker ('{marker}')!"
        return self.slices[0].data[marker].dtype
    
    def concat_slices(self, densities=False) -> pd.DataFrame:
        return pd.concat([slice.data if not densities else
                          pd.concat((slice.data["area"], slice.markers_density), axis=1)
                          for slice in self.slices])

    @staticmethod
    def merge_hemispheres(sliced_brain) -> Self:
        if not sliced_brain.is_split:
            return sliced_brain
        brain = copy.copy(sliced_brain)
        brain.slices = [BrainSlice.merge_hemispheres(brain_slice) for brain_slice in brain.slices]
        brain.is_split = False
        return brain

    @staticmethod
    def __handle_brainslice_error(exception, mode, name, results_file, regions_to_exclude_file):
        assert issubclass(type(exception), BrainSliceFileError), ""
        match mode:
            case "delete":
                print(f"Animal '{name}' -", exception, "\nRemoving the corresponding result and regions_to_exclude files.")
                os.remove(results_file)
                if type(exception) != ExcludedRegionsNotFoundError:
                    os.remove(regions_to_exclude_file)
            case "error":
                raise exception
            case "print":
                print(f"Animal '{name}' -", exception)
            case "silent":
                pass
            case _:
                raise ValueError(f"Invalid mode='{mode}' parameter. Supported BrainSliceFileError handling modes: 'delete', 'error', 'print', 'silent'.")

    @staticmethod
    def __get_default_error_mode(exception):
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
    
def get_image_names_in_folder(path: Path) -> list[str]:
    images = list({re.sub('_regions.txt[.lnk]*', '', file) for file in os.listdir(path)})
    # images = list({re.sub('_regions.csv[.lnk]*', '', file) for file in os.listdir(path)}) # csv_files
    images.sort()
    return images

def qupath_overlapping_classes(class1: str, class2: str) -> str:
    return f"{class1}~{class2}"

def overlapping_markers(marker1: str, marker2: str) -> str:
    return f"{marker1}+{marker2}"

def qupath_class2overlap(ch2marker: OrderedDict[str,str], overlapping_tracers: Iterable[tuple[int,int]]) -> dict[str,str]:
    assert all([len(idx) == 2 for idx in overlapping_tracers]), "Overlapping marker analyisis is supported only between two markers!"
    ordered_channels = list(ch2marker.keys())
    overlapping_channels = [(ordered_channels[i1], ordered_channels[i2]) for i1,i2 in overlapping_tracers]
    return {qupath_overlapping_classes(ch1, ch2): overlapping_markers(ch2marker[ch1], ch2marker[ch2]) for ch1, ch2 in overlapping_channels}