import pandas as pd
import re
from pathlib import Path
from typing import Self
from collections.abc import Iterable, Generator

from braian.brain_data import _is_split_left_right, extract_acronym, _sort_by_ontology, UnkownBrainRegionsError
from braian.ontology import AllenBrainOntology
from braian.utils import search_file_or_simlink

__all__ = ["BrainSlice"]

# global MODE_PathAnnotationObjectError
global MODE_ExcludedRegionNotRecognisedError
# MODE_PathAnnotationObjectError = "print"
MODE_RegionsWithNoCountError = "silent"
MODE_ExcludedRegionNotRecognisedError = "print"

class BrainSliceFileError(Exception):
    def __init__(self, file: str|Path, *args: object) -> None:
        self.file_path = file
        super().__init__(*args)
class ExcludedRegionsNotFoundError(BrainSliceFileError):
    def __str__(self):
        return f"Could not read the expected regions to exclude: {self.file_path}. "+\
                "Make sure it has the following naming scheme: '<IDENTIFIER><SUFFIX>.<EXTENSION>'"
class ExcludedAllRegionsError(BrainSliceFileError):
    def __str__(self):
        return f"The corresponding regions_to_exclude excludes everything: {self.file_path}"
class EmptyResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Empty file: {self.file_path}"
class NanResultsError(BrainSliceFileError):
    def __str__(self):
        return f"NaN-filled file: {self.file_path}"
class InvalidResultsError(BrainSliceFileError):
    def __str__(self):
        return f"Could not read results file: {self.file_path}"
class MissingResultsColumnError(BrainSliceFileError):
    def __init__(self, column, **kargs: object) -> None:
        self.column = column
        super().__init__(**kargs)
    def __str__(self):
        return f"Column '{self.column}' is missing in: {self.file_path}"
class RegionsWithNoCountError(BrainSliceFileError):
    def __init__(self, tracer, regions, **kargs: object) -> None:
        self.tracer = tracer
        self.regions = regions
        super().__init__(**kargs)
    def __str__(self) -> str:
        return f"There are {len(self.regions)} region(s) with no count of tracer '{self.tracer}' in file: {self.file_path}"
class InvalidRegionsHemisphereError(BrainSliceFileError):
    def __str__(self):
        return f"Slice {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"
class InvalidExcludedRegionsHemisphereError(BrainSliceFileError):
    def __str__(self):
        return f"Exclusions for Slice {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"

def overlapping_markers(marker1: str, marker2: str) -> str:
    return f"{marker1}+{marker2}"

class BrainSlice:
    __QUPATH_AREA = "Area um^2"

    @staticmethod
    def from_qupath(csv_file: str|Path, ch2marker: dict[str,str],
                    *args, **kwargs) -> Self:
        """
        Creates a [`BrainSlice`][braian.BrainSlice] from a file exported with
        [`qupath-extension-braian`](https://github.com/carlocastoldi/qupath-extension-braian).
        Additional arguments are passed to `BrainSlice`'s constructor.

        Parameters
        ----------
        csv_file
            The path to file exported with [`AtlasManager.saveResults()`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AtlasManager.html#saveResults(java.util.List,java.io.File)).
            If it ends with ".csv", it treats the file as a comma-separated table. Otherwise, it assuems it is tab-separated.
        ch2marker
            A dictionary mapping the QuPath channel names to markers.
            A cell segmentation algorithm must have previously run on each of the given channels.
        *args
            Other arguments are passed to [`BrainSlice`][braian.BrainSlice] constructor.
        **kwargs
            Other keyword arguments are passed to [`BrainSlice`][braian.BrainSlice] constructor.

        Returns
        -------
        :
            A [`BrainSlice`][braian.BrainSlice]

        Raises
        ------
        EmptyResultsError
            If the given `csv_file` is empty
        InvalidResultsError
            If the given `csv_file` is an invalid file. Probably because it was not exported through
            [`qupath-extension-braian`](https://github.com/carlocastoldi/qupath-extension-braian).
        NanResultsError
            If the number of total detections in all brain regions, exported from QuPath, is undefined
        MissingResultsColumnError
            If `csv_file` is missing reporting the number of detection in at least one `detected_channels`.
        """
        cols2marker = {BrainSlice.__column_from_qupath_channel(ch): m for ch,m in ch2marker.items()}
        data = BrainSlice.__read_qupath_data(search_file_or_simlink(csv_file))
        BrainSlice.__check_columns(data, [BrainSlice.__QUPATH_AREA, *cols2marker.keys()], csv_file)
        for column, ch1, ch2 in BrainSlice.__find_overlapping_channels(data, cols2marker.keys()):
            try:
                m1 = ch2marker[ch1]
                m2 = ch2marker[ch2]
            except KeyError:
                continue
            cols2marker[column] = overlapping_markers(m1, m2)
        data = pd.DataFrame(data, columns=[BrainSlice.__QUPATH_AREA, *cols2marker.keys()])
        # NOTE: not needed since qupath-extension-braian>=1.0.1
        BrainSlice.__fix_nan_countings(data, cols2marker.keys())
        data.rename(columns={BrainSlice.__QUPATH_AREA: "area"} | cols2marker, inplace=True)
        return BrainSlice(data, *args, **kwargs)

    @staticmethod
    def __column_from_qupath_channel(channel: str) -> str:
        return f"Num {channel}"

    @staticmethod
    def __read_qupath_data(csv_file: Path) -> pd.DataFrame:
        sep = "," if csv_file.suffix.lower() == ".csv" else "\t"
        try:
            data = pd.read_csv(csv_file, sep=sep).drop_duplicates()
        except Exception as e:
            if csv_file.stat().st_size == 0:
                raise EmptyResultsError(file=csv_file)
            else:
                raise InvalidResultsError(file=csv_file)
        if "Class" in data.columns:
            raise ValueError("You are analyising results file exported with QuPath <0.5.x. Such files are no longer supported!")
        BrainSlice.__check_columns(data, ["Name", "Classification", "Num Detections",], csv_file)
        # data = self.clean_rows(data, csv_file)
        if len(data) == 0:
            raise EmptyResultsError(file=csv_file)
        if data["Num Detections"].count() == 0:
            raise NanResultsError(file=csv_file)

        # There may be one region/row with Name == "Root" and Classification == NaN indicating the whole slice.
        # We remove it, as we want the distinction between hemispheres in the regions' acronym given by Classification column
        match (data["Classification"].isnull()).sum():
            case 0:
                data = data.set_index("Classification")
            case 1:
                data["Classification"] = data["Classification"].fillna("wholebrain")
                data = data.set_index("Classification")
                data = data.drop("wholebrain", axis=0)
            case _:
                raise InvalidResultsError(file=csv_file)
        data.index.name = None
        return data

    @staticmethod
    def __check_columns(data: pd.DataFrame, columns: Iterable[str],
                        csv_file: str|Path) -> bool:
        for column in columns:
            if column not in data.columns:
                raise MissingResultsColumnError(file=csv_file, column=column)
        return True

    @staticmethod
    def __find_overlapping_channels(data: pd.DataFrame,
                                    detections_cols: Iterable[str]) -> Generator[tuple[str], None, None]:
        other_cols = [col for col in data.columns if col not in (*detections_cols, BrainSlice.__QUPATH_AREA)]
        if len(other_cols) == 0:
            return
        for col_name in other_cols:
            match = re.match(r"Num (.+)\~(.+)", col_name)
            if match is None:
                continue
            overlapping_channels = match.groups()
            if len(overlapping_channels) == 2:
                yield (col_name, *overlapping_channels)
        return

    @staticmethod
    def __fix_nan_countings(data, detection_columns):
        # old version (<1.0.1) of qupath.ext.braian.AtlasManager.saveResults()
        # fills with NaNs columns when there is< no detection in the whole image
        # This, for instance, can happen when there is no overlapping detection.
        # The counting are thus set to zero
        are_detections_missing = data[detection_columns].isna().all()
        missing_detections = are_detections_missing.index[are_detections_missing]
        data[missing_detections] = 0

    @staticmethod
    def read_qupath_exclusions(file_path: Path|str) -> list[str]:
        """
        Reads the regions to exclude from the analysis of a [`BrainSlice`][braian.BrainSlice]
        from a file exported with [`qupath-extension-braian`](https://github.com/carlocastoldi/qupath-extension-braian).

        Parameters
        ----------
        file_path
            The path to file exported with [`AtlasManager.saveExcludedRegions()`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AtlasManager.html#saveExcludedRegions(java.io.File)).

        Returns
        -------
        :
            A list of acronyms of brain regions.

        Raises
        ------
        ExcludedRegionsNotFoundError
            If the `file_path` was not found.
        """
        try:
            with open(search_file_or_simlink(file_path), mode="r", encoding="utf-8") as file:
                excluded_regions = file.readlines()
        except FileNotFoundError:
            raise ExcludedRegionsNotFoundError(file_path)
        return [line.strip() for line in excluded_regions]

    # NOTE: since switching to qupath-extension-braian, this function is useless
    # def clean_rows(self, data: pd.DataFrame, csv_file: str):
    #     if (data["Name"] == "Exclude").any():
    #         # some rows have the Name==Exclude because the cell counting script was run AFTER having done the exclusions
    #         data = data.loc[data["Name"] != "Exclude"]
    #     if (data["Name"] == "PathAnnotationObject").any() and \
    #         not (data["Name"] == "PathAnnotationObject").all():
    #         global MODE_PathAnnotationObjectError
    #         if MODE_PathAnnotationObjectError != "silent":
    #             print(f"WARNING: there are rows with column 'Name'=='PathAnnotationObject' in animal '{self.animal}', file: {csv_file}\n"+\
    #                     "\tPlease, check on QuPath that you selected the right Class for every exclusion.")
    #         data = data.loc[data["Name"] != "PathAnnotationObject"]
    #     return data
    # NOTE: checking whether a region has no detection at all is useless at this stage
    # def check_zero_rows(self, csv_file: str, markers: list[str]) -> bool:
    #     for marker in markers:
    #         zero_rows = self.data[marker] == 0
    #         if sum(zero_rows) > 0:
    #             err = RegionsWithNoCountError(slice=self, file=csv_file,
    #                         tracer=marker, regions=self.data.index[zero_rows].to_list())
    #             if MODE_RegionsWithNoCountError == "error":
    #                 raise err
    #             elif MODE_RegionsWithNoCountError == "print":
    #                 print(err)
    #             return False
    #     return True

    def __init__(self, data: pd.DataFrame, animal:str, name: str, is_split: bool,
                 area_units: str="µm2", brain_ontology: AllenBrainOntology|None=None) -> None:
        """
        Creates a `BrainSlice` from a [`DataFrame`][pandas.DataFrame]. Each row representes the data
        of a single brain region, whose acronym is used as index. If the data was collected
        distinguishing between the two hemispheres, the index is expected to be either `Left: <ACRONYM>`
        or `Right: <ACRONYM>`. The DataFrame is expected to have at least two columns: one named `"area"`
        corresponding to the size in `area_units`, and the others corresponding to the markers used to
        measure brain activity.

        Parameters
        ----------
        data
            The data extracted from a brain slice.
        animal
            The name of the animal from which the slice was cut.
        name
            The name of hte brain slice.
        is_split
            Whether the data was extracted distinguishing between the two hemispheres or not.
        area_units
            The units of measurements used by `"area"` column in `data`.
            Accepted values are: `"µm2"`, `"um2"` or `mm2`.
        brain_ontology
            If specified, it checks the brain regions in `data` against the given ontology
            and it sorts the rows in depth-first order in ontology's hierarchy.

        Raises
        ------
        InvalidRegionsHemisphereError
            If `is_split=True` but some rows' indices don't start with `"Left: "` or with `"Right: "`.
        ValueError
            If the specified `area_units` is unknown.
        """
        self.animal: str = animal
        """The name of the animal from which the current `BrainSlice` is from."""
        self.name: str = name
        """The name of the image that captured the section from which the data of the current `BrainSlice` are from."""
        self.data = data
        self.is_split: bool = _is_split_left_right(self.data.index)
        """Whether the data of the current `BrainSlice` make a distinction between right and left hemisphere."""
        if is_split and not self.is_split:
            raise InvalidRegionsHemisphereError(file=self.name)
        BrainSlice.__check_columns(self.data, ("area",), self.name)
        assert self.data.shape[1] >= 2, "'data' should have at least one column, apart from 'area', containing per-region data"
        match area_units:
            case "µm2" | "um2":
                self.__area_µm2_to_mm2()
            case "mm2":
                pass
            case _:
                raise ValueError("A brain slice's area can only be expressed in µm² or mm²!")
        assert (self.data["area"] > 0).any(), f"All region areas are zero or NaN for animal={self.animal} slice={self.name}"
        self.data = self.data[self.data["area"] > 0]
        if brain_ontology is not None:
            self.data = _sort_by_ontology(self.data, brain_ontology, fill=False)

        self.markers_density: pd.DataFrame = self.__marker_density()

    @property
    def markers(self) -> list[str]:
        """The name of the markers for which the current `BrainSlice` has data."""
        return list(self.data.columns[self.data.columns != "area"])


    def exclude_regions(self,
                        excluded_regions: Iterable[str],
                        brain_ontology: AllenBrainOntology,
                        exclude_parent_regions: bool):
        """
        Takes care of the regions to be excluded from the analysis.\\
        If `exclude_parent_regions` is `False`, for each region:

         1. the cell counts of that region is subtracted fromm all its parent regions
         2. its cell counts are deleted from the current `BrainSlice`, along with all of its subregions

        If `exclude_parent_regions` is `True`, the parent regions of an excluded region is also
        deleted from the current `BrainSlice` since its cell count is determined also by that same region.

        NOTE: Netherless, if a Layer 1 region is _explicitly_ excluded, it won't
        impact (i.e. remove) the parent regions.
        This decision was taken because Layer 1 is often mis-aligned but with
        few detection. We don't want to delete too much data and we reckon that
        this exception does not impact too much on the data of the whole Isocortex.

        Parameters
        ----------
        excluded_regions
            a list of acronyms of the regions to be excluded from the analysis.
        brain_ontology
            an ontology against whose version the brain section was aligned.
        exclude_parent_regions
            Whether the cell counts of the parent regions are excluded or subtracted.

        Raises
        ------
        InvalidExcludedRegionsHemisphereError
            if [`BrainSlice.is_split`][braian.BrainSlice.is_split] but a region in
            `excluded_regions` is not considering left/right hemisphere distinction.
        UnkownBrainRegionsError
            if a region in `excluded_regions` is not recognised from `brain_ontology`.
        ExcludedAllRegionsError
            if there is no cell count left after the exclusion is done.
        """
        try:
            layer1 = set(brain_ontology.get_layer1())
            for exclusion in excluded_regions:
                if self.is_split:
                    if ": " not in exclusion:
                        if MODE_ExcludedRegionNotRecognisedError != "silent":
                            print(f"WARNING: Class '{exclusion}' is not recognised as a brain region. It was skipped from the regions_to_exclude in animal '{self.animal}', file: {self.name}_regions_to_exclude.txt")
                            continue
                        elif MODE_ExcludedRegionNotRecognisedError == "error":
                            raise InvalidExcludedRegionsHemisphereError(file=self.name)
                    hem, region = exclusion.split(": ")
                else:
                    region = exclusion
                if not brain_ontology.is_region(region):
                    raise UnkownBrainRegionsError((region,))

                # Step 1: subtract counting results of the regions to be excluded
                # from their parent regions.
                regions_above = brain_ontology.get_regions_above(region)
                for parent_region in regions_above:
                    row = hem+": "+parent_region if self.is_split else parent_region
                    if row in self.data.index:
                        if exclude_parent_regions and region not in layer1:
                            self.data.drop(row, inplace=True)
                        elif exclusion in self.data.index:
                            # Subtract the counting results from the parent region.
                            # Use fill_value=0 to prevent "3-NaN=NaN".
                            self.data.loc[row] = self.data.loc[row].subtract(self.data.loc[exclusion], fill_value=0)

                # Step 2: Remove the regions that should be excluded
                # together with their daughter regions.
                subregions = brain_ontology.list_all_subregions(region)
                for subregion in subregions:
                    row = hem+": "+subregion if self.is_split else subregion
                    if row in self.data.index:
                        self.data.drop(row, inplace=True)
        except Exception:
            raise Exception(f"Animal '{self.animal}': failed to exclude regions for in slice '{self.name}'")
        finally:
            self.markers_density = self.__marker_density()
        if len(self.data) == 0:
            raise ExcludedAllRegionsError(file=self.name)

    def __area_µm2_to_mm2(self) -> None:
        self.data.area = self.data.area * 1e-06

    def __marker_density(self) -> pd.DataFrame:
        return self.data[self.markers].div(self.data["area"], axis=0)

    @staticmethod
    def merge_hemispheres(slice: Self) -> Self:
        """
        For each brain region, sums the data of left and right hemispheres into one single datum

        Parameters
        ----------
        slice
            A brain section to merge

        Returns
        -------
        :
            A new [`BrainSlice`][braian.BrainSlice] with no hemisphere distinction.
            If `slice` is already merged, it return the same instance with no changes.
        """
        if not slice.is_split:
            return slice
        corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in slice.data.index]
        data = slice.data.groupby(corresponding_region).sum(min_count=1)
        return BrainSlice(data, slice.animal, slice.name, is_split=False, area_units="mm2")