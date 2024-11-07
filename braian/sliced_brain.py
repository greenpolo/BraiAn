import copy
import numpy as np
import pandas as pd
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from braian.ontology import AllenBrainOntology
from braian.brain_data import BrainData
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
    __QUPATH_AREA_UNITS = "µm2"

    @staticmethod
    def from_qupath(name: str,
                    animal_dir: str|Path,
                    brain_ontology: AllenBrainOntology,
                    ch2marker: dict[str,str],
                    exclude_parent_regions: bool=False,
                    results_subdir: str="results",
                    results_suffix: str="_regions.tsv",
                    exclusions_subdir: str="regions_to_exclude",
                    exclusions_suffix: str="_regions_to_exclude.txt"
                    ) -> Self:
        """
        Creates a [`SlicedBrain`][braian.SlicedBrain] from all the per-image files exported with
        [`qupath-extension-braian`](https://github.com/carlocastoldi/qupath-extension-braian)
        inside `animal_dir`.\
        It assumes that cell counts and exclusions files have the following naming structure:
        `<IDENTIFIER><SUFFIX>.<EXTENSION>`. The _identifier_ must be common in files relatives
        to the same image. The _suffix_ must be common to files of the same kind (i.e. cell counts
        or exclusions). The _extension_ [defines][braian.BrainSlice.from_qupath] whether the table
        is comma-separated or tab-separated.

        Parameters
        ----------
        name
            The name of the animal.
        animal_dir
            The path to where all the reports of the brain sections were saved from QuPath. Both per-region results and exclusions.
        brain_ontology
            An ontology against whose version the brain was aligned.
        ch2marker
            A dictionary mapping QuPath channel names to markers.
        exclude_parent_regions
            `exclude_parent_regions` from [`BrainSlice.exclude_regions`][braian.BrainSlice.exclude_regions].
        results_subdir
            The name of the subfolder in `animal_dir` that contains all cell counts files of each brain section.\\
            It can be `None` if no subfolder is used.
        results_suffix
            The suffix used to identify cell counts files saved in `results_subdir`. It includes the file extension.
        exclusions_subdir
            The name of the subfolder in `animal_dir` that contains all regions to exclude from further analysis of each brain section.\\
            It can be `None` if no subfolder is used.
        exclusions_suffix
            The suffix used to identify exclusion files saved in `results_subdir`. It includes the file extension.

        Returns
        -------
        :
            A [`SlicedBrain`][braian.SlicedBrain].

        See also
        --------
        [`BrainSlice.from_qupath`][braian.BrainSlice.from_qupath]
        [`BrainSlice.exclude_regions`][braian.BrainSlice.exclude_regions]
        """
        if not isinstance(animal_dir, Path):
            animal_dir = Path(animal_dir)
        csv_slices_dir = animal_dir / results_subdir if results_subdir is not None else animal_dir
        excluded_regions_dir = animal_dir / exclusions_subdir if exclusions_subdir is not None else animal_dir
        images = get_image_names_in_folder(csv_slices_dir, results_suffix)
        slices: list[BrainSlice] = []
        for image in images:
            results_file = csv_slices_dir/(image+results_suffix)
            excluded_regions_file = excluded_regions_dir/(image+exclusions_suffix)
            try:
                # Setting brain_ontology=None, we don't check that the data corresponds to real brain regions
                # we post-pone the check later in the analysis for performance reasons.
                # The assumption is that if you're creating a SlicedBrain, you will eventually do
                # group analysis. Checking against the ontology for each slice would be too time consuming.
                # We can do it afterwards, after the SlicedBrain is reduced to AnimalBrain
                slice: BrainSlice = BrainSlice.from_qupath(results_file,
                                               ch2marker,
                                               animal=name, name=image, is_split=True,
                                               area_units=SlicedBrain.__QUPATH_AREA_UNITS, brain_ontology=None)
                exclude = BrainSlice.read_qupath_exclusions(excluded_regions_file)
                slice.exclude_regions(exclude, brain_ontology, exclude_parent_regions)
            except BrainSliceFileError as e:
                mode = SlicedBrain.__get_default_error_mode(e)
                SlicedBrain.__handle_brainslice_error(e, mode, name, results_file, excluded_regions_file)
            else:
                slices.append(slice)
        # DOES NOT PRESERVE ORDER
        # markers = {marker for slice in slices for marker in slice.markers_density.columns}
        # PRESERVES ORDER
        # all_markers = np.array((marker for slice in slices for marker in slice.markers_density.columns))
        # _, idx = np.unique(all_markers, return_index=True)
        # markers = all_markers[np.sort(idx)]
        # PRESERVES ORDER: FROM PYTHON 3.7+,
        #                  due to dict implementation details! (i.e. not guaranteed)
        markers = list(dict.fromkeys((marker for slice in slices for marker in slice.markers_density.columns)))
        return SlicedBrain(name, slices, markers)


    def __init__(self, name: str, slices: Iterable[BrainSlice], markers: Iterable[str]) -> None:
        """
        A `SlicedBrain` is a collection of [`BrainSlice`][braian.BrainSlice], and it is
        an basic structure from which [`AnimalBrain`][braian.AnimalBrain] are reconstructed.

        Parameters
        ----------
        name
            The name of the animal.
        slices
            The list of [`BrainSlice`][braian.BrainSlice] that makes up a sample of a brain.
        markers
            The list of markers in used in all `BrainSlice`s.

        Raises
        ------
        EmptyBrainError
            If `slices` is empty.
        """
        self._name = name
        self._slices: tuple[BrainSlice] = tuple(slices)
        if len(self._slices) == 0:
            raise EmptyBrainError(self._name)
        self.markers = list(markers)
        are_split = np.array([s.is_split for s in self._slices])
        assert are_split.all() or ~are_split.any(), "Slices from the same animal should either be ALL split between right/left hemisphere or not."
        self.is_split = are_split[0]
        """Whether the data of the current `SlicedBrain` makes a distinction between right and left hemisphere."""

    @property
    def name(self) -> str:
        """The name of the animal."""
        return str(self._name)

    @name.setter
    def name(self, value: str):
        for slice in self._slices:
            slice.animal = value
        self._name = value

    @property
    def slices(self) -> tuple[BrainSlice]:
        """The list of slices making up the `SlicedBrain`."""
        return self._slices

    def concat_slices(self, densities: bool=False) -> pd.DataFrame:
        """
        Combines all the [`BrainSlice`][braian.BrainSlice] making up the current
        `SlicedBrain` into a [`DataFrame`][pandas.DataFrame].

        Parameters
        ----------
        densities
            If True, the result is a [`DataFrame`][pandas.DataFrame] of slices marker densities.
            Otherwise, the result will contain the cell counts.

        Returns
        -------
        :
            A [`DataFrame`][pandas.DataFrame] of the data from all [`SlicedBrain.slices`][braian.SlicedBrain.slices].
        """
        return pd.concat([slice.data if not densities else
                          pd.concat((slice.data["area"], slice.markers_density), axis=1)
                          for slice in self._slices])

    def count(self, brain_ontology: AllenBrainOntology=None) -> BrainData:
        """
        Counts the number of slices that contains data for each brain region.

        Parameters
        ----------
        brain_ontology
            If specified, it sorts and check the regions accordingly to the given atlas ontology.

        Returns
        -------
        :
            A `BrainData` with the number of slices per region.
        """
        all_slices = self.concat_slices()
        count = all_slices.groupby(all_slices.index).count().iloc[:,0]
        return BrainData(count, self._name, "count_slices", "#slices", brain_ontology=brain_ontology, fill_nan=False)

    @staticmethod
    def merge_hemispheres(sliced_brain: Self) -> Self:
        """
        Creates a new `SlicedBrain` from all merged [`BrainSlice`][braian.BrainSlice]
        in `sliced_brain`.

        Parameters
        ----------
        sliced_brain
            A sliced brain to merge.

        Returns
        -------
        :
            A new [`SlicedBrain`][braian.SlicedBrain] with no hemisphere distinction.
            If `sliced_brain` is already merged, it return the same instance with no changes.

        See also
        --------
        [`BrainSlice.merge_hemispheres`][braian.BrainSlice.merge_hemispheres]
        """
        if not sliced_brain.is_split:
            return sliced_brain
        brain = copy.copy(sliced_brain)
        brain._slices = [BrainSlice.merge_hemispheres(brain_slice) for brain_slice in brain._slices]
        brain.is_split = False
        return brain

    @staticmethod
    def __handle_brainslice_error(exception, mode, name, results_file: Path, regions_to_exclude_file: Path):
        assert issubclass(type(exception), BrainSliceFileError), ""
        match mode:
            case "delete":
                print(f"Animal '{name}' -", exception, "\nRemoving the corresponding result and regions_to_exclude files.")
                results_file.unlink()
                if not isinstance(exception, ExcludedRegionsNotFoundError):
                    regions_to_exclude_file.unlink()
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

def get_image_names_in_folder(path: Path, exclusions_suffix: str) -> list[str]:
    assert path.is_dir(), f"'{str(path)}' is not an existing directory."
    match = re.escape(exclusions_suffix)+r"[.lnk]*$" # allow for windows symlink as well
    images = list({re.sub(match, "", file.name) for file in path.iterdir() if file.is_file()})
    images.sort()
    return images