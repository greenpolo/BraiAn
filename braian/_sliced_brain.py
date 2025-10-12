import copy
import numpy as np
import pandas as pd
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from braian import AnimalBrain, AtlasOntology, BrainData, BrainHemisphere, BrainSlice, SlicedMetric,\
                   BrainSliceFileError, \
                   ExcludedAllRegionsError, \
                   ExcludedRegionsNotFoundError, \
                   EmptyResultsError, \
                   NanResultsError, \
                   InvalidResultsError, \
                   MissingQuantificationError, \
                   InvalidRegionsHemisphereError, \
                   InvalidExcludedRegionsHemisphereError
from braian.utils import deprecated

__all__ = [
    "SlicedBrain",
    "EmptyBrainError",
]

global MODE_ExcludedAllRegionsError
global MODE_ExcludedRegionsNotFoundError
global MODE_EmptyResultsError
global MODE_NanResultsError
global MODE_InvalidResultsError
global MODE_MissingResultsColumnError
global MODE_MissingResultsMeasurementError
global MODE_InvalidRegionsHemisphereError
global MODE_InvalidExcludedRegionsHemisphereError
MODE_ExcludedAllRegionsError = "print"
MODE_ExcludedRegionsNotFoundError = "print"
MODE_EmptyResultsError = "print"
MODE_NanResultsError = "print"
MODE_InvalidResultsError = "print"
MODE_MissingResultsColumnError = "print"
MODE_MissingResultsMeasurementError = "print"
MODE_InvalidRegionsHemisphereError = "print"
MODE_InvalidExcludedRegionsHemisphereError = "print"

class EmptyBrainError(Exception):
    def __init__(self, context=None):
        context = f"'{context}': " if context is not None else ""
        super().__init__(f"{context}Inconsistent hemisphere distinction across sections of the same brain!"+\
                         +" Some report the data with split regions, while others do not.")
class InconsistentRegionsSplitError(Exception):
    def __init__(self, context=None):
        context = f"'{context}': " if context is not None else ""
        super().__init__(f"{context}Inconsistent hemisphere distinction across sections of the same brain!"+\
                         +" Some report the data with split regions, while others do not.")

class SlicedBrain:
    @staticmethod
    @deprecated(since="1.1.0", params=["exclude_parent_regions"],
                message="Quantifications in ancestor regions are now completely removed too. "+\
                        "If you want the old behaviour, you can momentarily use braian.BrainSlice.exclude with ancestors=False")
    def from_qupath(name: str,
                    animal_dir: str|Path,
                    brain_ontology: AtlasOntology,
                    ch2marker: dict[str,str],
                    exclude_parent_regions: bool=True,
                    exclude_ancestors_layer1: bool=True,
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
            `exclude_parent_regions` from [`BrainSlice.exclude`][braian.BrainSlice.exclude].
        exclude_ancestors_layer1
            `ancestors_layer1` from [`BrainSlice.exclude`][braian.BrainSlice.exclude].
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
        [`BrainSlice.exclude`][braian.BrainSlice.exclude]
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
                # Setting check=True because braian has to check the existance of every brain region
                # at read-time. This way, no following check on the existence of the brain region should be
                # necessary.
                slice: BrainSlice = BrainSlice.from_qupath(results_file, ch2marker,
                                                           animal=name, name=image,
                                                           ontology=brain_ontology, check=True)
                exclude = BrainSlice.read_qupath_exclusions(excluded_regions_file)
                slice.exclude(exclude, ontology=brain_ontology,
                              ancestors_layer1=exclude_ancestors_layer1)
            except BrainSliceFileError as e:
                mode = _get_default_error_mode(e)
                _handle_brainslice_error(e, mode, name, results_file, excluded_regions_file)
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
        markers = list(dict.fromkeys((marker for slice in slices for marker in slice.markers_density.columns
                                      if marker not in ("acronym", "hemisphere"))))
        return SlicedBrain(name, slices, markers)


    def __init__(self, name: str, slices: Iterable[BrainSlice], markers: Iterable[str]) -> None:
        """
        A `SlicedBrain` is a collection of [`BrainSlice`][braian.BrainSlice], and it is
        an basic structure from which [`AnimalBrain`][braian.AnimalBrain] are reduced.

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
        InconsistentRegionsSplitError
            If some `slices` report the data making a distinction between right and left hemispheres, while others do not.
        """
        self._name = name
        self._slices: tuple[BrainSlice] = tuple(slices)
        if len(self._slices) == 0:
            raise EmptyBrainError(context=self._name)
        self._markers: list[str] = markers
        _check_same_units(self._slices)
        self._units = self._slices[0].units
        are_split = np.array([s.is_split for s in self._slices])
        if not are_split.all() and are_split.any():
            raise InconsistentRegionsSplitError(context=self._name)
        self._is_split = are_split[0]

    @property
    def atlas(self) -> str:
        """The name of the atlas used to align the brain data."""
        return self._slices[0].atlas

    @property
    def name(self) -> str:
        """The name of the animal."""
        return str(self._name)

    @name.setter
    def name(self, value: str):
        for slice in self._slices:
            slice._animal = value
        self._name = value

    @property
    def n(self) -> int:
        """The number of brain sections within the `SlicedBrain`."""
        return len(self._slices)

    @property
    def markers(self) -> list[str]:
        """The name of the markers for which the current `SlicedBrain` has data."""
        return list(self._markers)

    @property
    def units(self) -> dict[str,str]:
        """The units of measurements for each [`marker`][braian.SlicedBrain.markers] quantification."""
        return self._units.copy()

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return self._is_split

    @property
    def regions(self) -> list[str]:
        """
        The list of region acronyms for which the current `SlicedBrain` has data. The given order is arbitrary.
        If [`SlicedBrain.is_split`][braian.SlicedBrain.is_split], it contains the acronyms of the split brain region only once.
        """
        return pd.unique(np.array([r for s in self.slices for r in s.regions]))

    @property
    def slices(self) -> tuple[BrainSlice]:
        """The list of slices making up the `SlicedBrain`."""
        return self._slices

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"SlicedBrain(name='{self.name}', slices={self.n}, is_split={self.is_split})"

    def __iter__(self) -> Iterable[BrainSlice]:
        return iter(self._slices)

    def __len__(self) -> int:
        return len(self._slices)

    def __contains__(self, name: str) -> bool:
        if not isinstance(name, str):
            return False
        for slice in self._slices:
            if slice._name == name:
                return True
        return False

    def __getitem__(self, name: str) -> BrainSlice:
        if not isinstance(name, str):
            raise TypeError("BrainSlices are identified by strings")
        try:
            return next(slice for slice in self._slices if slice._name == name)
        except StopIteration:
            pass
        raise KeyError(name)

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
        return pd.concat([slice._data if not densities else
                          pd.concat((slice._data["area"], slice.markers_density), axis=1)
                          for slice in self._slices])

    def count(self, brain_ontology: AtlasOntology=None) -> BrainData|dict[BrainHemisphere, BrainData]:
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
        count = all_slices.groupby(["acronym", "hemisphere"]).count().iloc[:,0]
        if self.is_split:
            return {hem: BrainData(count.xs(hem.value, level="hemisphere"),
                        self._name, "count_slices", "#slices", hemisphere=hem,
                        brain_ontology=brain_ontology, fill_nan=False)
                    for hem in (BrainHemisphere.LEFT, BrainHemisphere.RIGHT)}
        hem = BrainHemisphere.BOTH
        return BrainData(count.xs(hem.value, level="hemisphere"),
                        self._name, "count_slices", "#slices", hemisphere=hem,
                        brain_ontology=brain_ontology, fill_nan=False)

    def merge_hemispheres(self) -> Self:
        """
        Creates a new `SlicedBrain` from all merged [`BrainSlice`][braian.BrainSlice]
        in `sliced_brain`.

        Returns
        -------
        :
            A new [`SlicedBrain`][braian.SlicedBrain] with no hemisphere distinction.
            If `sliced_brain` is already merged, it return the same instance with no changes.

        See also
        --------
        [`BrainSlice.merge_hemispheres`][braian.BrainSlice.merge_hemispheres]
        """
        if not self.is_split:
            return self
        brain = copy.copy(self)
        brain._slices = [brain_slice.merge_hemispheres() for brain_slice in brain._slices]
        brain._is_split = False
        return brain

    def region(self,
               region: str,
               *,
               metric: str,
               hemisphere: BrainHemisphere=BrainHemisphere.BOTH,
               as_density: bool=False) -> pd.DataFrame:
        """
        Extracts all values of a brain region from all [`slices`][braian.SlicedBrain.slices] in the brain.
        If the brain [is split][braian.SlicedBrain.is_split], the resulting DataFrame
        may contain two values for the same brain slice.

        Parameters
        ----------
        region
            A brain structure identified by its acronym.
        metric
            The metric to extract from the `SlicedBrain`.
            It can either be `"area"` or any value in [`SlicedBrain.markers`][braian.SlicedBrain.markers].
        hemisphere
            The hemisphere of the brain region to extract. If [`BOTH`][braian.BrainHemisphere]
            and the brain [is split][braian.BrainSlice.is_split], it may return both hemispheric values
            of the region.
        as_density
            If `True`, it retrieves the values as densities (i.e. marker/area).

        Returns
        -------
        :
            A `DataFrame` with:

            * a [`pd.MultiIndex`][pandas.MultiIndex] of _slice_ and _hemisphere_,
            revealing the name of the slice and the [hemisphere][braian.BrainHemisphere]
            from which the region data was extracted;

            * a `metric` column, revealing the value for the specified brain region.

            If there is no data for `region`, it returns an empty `DataFrame`.
        """
        vals = [(s.name, hem, val)
                for s in self.slices
                for val,hem in zip(*s.region(region=region, metric=metric, hemisphere=hemisphere, as_density=as_density, return_hemispheres=True))
                if region in s.regions]
        df = pd.DataFrame(vals, columns=("slice", "hemisphere", metric))
        df.set_index(["slice", "hemisphere"], drop=True, append=False, inplace=True)
        return df

    def reduce(self,
               metric: SlicedMetric|str=SlicedMetric.SUM,
               *,
               min_slices: int=0,
               densities: bool=False) -> AnimalBrain:
        """
        Crates a cohesive [`AnimalBrain`][braian.AnimalBrain] from data coming from brain sections.

        Parameters
        ----------
        metric
            The metric used to reduce sections data from the same region into a single value.
        min_slices
            The minimum number of sections for a reduction to be valid.
            If a region has not enough sections, it will disappear from the dataset.
        densities
            If True, it computes the reduction on the section density (i.e., marker/area)
            instead of doing it on the raw cell counts.

        Returns
        -------
        :

        Raises
        ------
        EmptyBrainError
            when `sliced_brain` has not enough sections or when `min_slices` filters out all brain regions.

        See also
        --------
        [`SlicedGroup.reduce`][braian.SlicedGroup.reduce]
        [`SlicedExperiment.reduce`][braian.SlicedExperiment.reduce]
        """
        name = self.name
        markers = copy.copy(self.markers)
        metric = SlicedMetric(metric)
        if len(self.slices) < min_slices:
            raise EmptyBrainError(self.name)
        all_slices = self.concat_slices(densities=densities)
        all_slices = all_slices.groupby(by=["acronym", "hemisphere"]).filter(lambda g: len(g) >= min_slices)
        redux = metric(all_slices.groupby(by=["acronym", "hemisphere"]))
        if redux.shape[0] == 0: # TODO: could change to len(redux) == 0?
            raise EmptyBrainError(self.name)
        metric = f"{str(metric)}_densities" if densities else str(metric)
        if self.is_split:
            hemispheres = (BrainHemisphere.LEFT, BrainHemisphere.RIGHT)
        else:
            hemispheres = (BrainHemisphere.BOTH,)
        # areas = BrainData(redux["area"], name=name, metric=metric, units=sliced_brain.units["area"])
        areas = tuple(BrainData(redux["area"].xs(hem.value, level=1),
                                name=name, metric=metric,
                                units=self.units["area"],
                                hemisphere=hem)
                for hem in hemispheres)
        markers_data = {
            m: tuple(
                BrainData(redux[m].xs(hem.value, level=1),
                                      name=name, metric=metric,
                                      units=self.units[m],
                                      hemisphere=hem)
                for hem in hemispheres)
            for m in markers
        }
        return AnimalBrain(markers_data=markers_data, sizes=areas)

def _check_same_units(slices: Iterable[BrainSlice]):
    units = pd.DataFrame([s.units for s in slices])
    units_np = units.to_numpy()
    if not all(same_units:=(units_np[0] == units_np).all(0)):
        raise ValueError("Some measurements do not have the same unit of measurement for all slices: "+\
                            ", ".join(units.columns[~same_units]))

def _handle_brainslice_error(exception, mode, name, results_file: Path, regions_to_exclude_file: Path):
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

def _get_default_error_mode(exception):
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
        case MissingQuantificationError.__class__:
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