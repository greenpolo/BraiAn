import itertools
import numpy as np
import pandas as pd
import re

from dataclasses import dataclass
from enum import Enum
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from typing import Self
from collections.abc import Iterable, Sequence

from braian import AtlasOntology, BrainHemisphere, UnknownBrainRegionsError, InvalidRegionsHemisphereError
from braian._brain_data import extract_legacy_hemispheres, sort_by_ontology
from braian.utils import deprecated, search_file_or_simlink

__all__ = [
    "BrainSlice",
    "QuPathMeasurementType",
    "BrainSliceFileError",
    "ExcludedRegionsNotFoundError",
    "ExcludedAllRegionsError",
    "EmptyResultsError",
    "NanResultsError",
    "InvalidResultsError",
    "MissingResultsMeasurementError",
    "RegionsWithNoCountError",
    "InvalidRegionsHemisphereError",
    "InvalidExcludedRegionsHemisphereError",
]

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
class MissingResultsMeasurementError(BrainSliceFileError):
    def __init__(self, channel, **kargs: object) -> None:
        self.channel = channel
        super().__init__(**kargs)
    def __str__(self):
        return f"Measurements missing for channel '{self.channel}' in: {self.file_path}"
class InvalidRegionsSplitError(Exception):
    def __init__(self, context=None):
        context = f"'{context}': " if context is not None else ""
        super().__init__(f"{context}The data should be split between right and left hemispheres for ALL regions.")
class RegionsWithNoCountError(BrainSliceFileError):
    def __init__(self, tracer, regions, **kargs: object) -> None:
        self.tracer = tracer
        self.regions = regions
        super().__init__(**kargs)
    def __str__(self) -> str:
        return f"There are {len(self.regions)} region(s) with no count of tracer '{self.tracer}' in file: {self.file_path}"
class InvalidExcludedRegionsHemisphereError(BrainSliceFileError):
    def __str__(self):
        return f"Exclusions for Slice {self.file_path}"+" is badly formatted. Each row is expected to be of the form '{Left|Right}: <region acronym>'"

def overlapping_markers(marker1: str, marker2: str) -> str:
    return f"{marker1}+{marker2}"

class QuPathMeasurementType(Enum):
    r"""
    Enum of the supported types of measurements available from QuPath.

    Attributes
    ----------
    AREA
        The associated measurements report per-region _coverages_ (e.g. area covered by axonal projections).
    CELL_COUNT
        The associated measurements report per-region _cell counts_ (e.g. cFos positive quantifications).
    """
    AREA = 0
    CELL_COUNT = 1

@dataclass
class QuPathMeasurement:
    r"""Class for identifying measurements extracted with QuPath."""
    key: str
    measurement: str
    type: QuPathMeasurementType

    def colabelled_channels(self) -> tuple[str,str]:
        match = re.match(r"(.+)\~(.+)", self.measurement)
        if match is None:
            return None
        channels = match.groups()
        if len(channels) == 2:
            return channels

    def unit(self) -> str:
        match self.type:
            case QuPathMeasurementType.AREA:
                return "µm²"
            case QuPathMeasurementType.CELL_COUNT:
                assert "name" in self.__dict__
                return self.name
            case _:
                raise ValueError(f"Unknown QuPath measurement type: '{self.type}'")

QUPATH_REGEX_MEASUREMENT_AREA = re.compile(r"(.+) area [u,µ]m\^2")
QUPATH_REGEX_MEASUREMENT_COUNT = re.compile(r"Num (.+)")
QUPATH_AREA = "Area um^2"

def unique_parse(regex: re.Pattern, s: str):
    result = regex.findall(s)
    match len(result):
        case 0:
            return None
        case 1:
            return result[0]
        case _:
            raise ValueError(f"Unexpected parsing of '{s}': {result}")

def extract_qupath_measurement_type(measurement: str) -> tuple[str,QuPathMeasurementType]:
    if measurement == QUPATH_AREA:
        return "area",QuPathMeasurementType.AREA
    if channel:=unique_parse(QUPATH_REGEX_MEASUREMENT_AREA, measurement):
        return channel,QuPathMeasurementType.AREA
    if channel:=unique_parse(QUPATH_REGEX_MEASUREMENT_COUNT, measurement):
        return channel,QuPathMeasurementType.CELL_COUNT
    raise ValueError(f"Cannot extract units for measurement '{measurement}'")

def extract_qupath_measurement_types(data: pd.DataFrame) -> list[QuPathMeasurement]:
    measurements = [c for c in data.columns if c not in ("Image Name", "Name", "Num Detections")]
    return [QuPathMeasurement(m, *extract_qupath_measurement_type(m)) for m in measurements]

class BrainSlice:
    __QUPATH_ATLAS_CONTAINER = "Root"

    @staticmethod
    def from_qupath(csv: str|Path|pd.DataFrame, ch2marker: dict[str,str],
                    *,
                    animal:str, name: str,
                    ontology: AtlasOntology, check: bool) -> Self:
        """
        Creates a [`BrainSlice`][braian.BrainSlice] from a file exported with
        [`qupath-extension-braian`](https://github.com/carlocastoldi/qupath-extension-braian).
        Additional arguments are passed to `BrainSlice`'s constructor.

        Parameters
        ----------
        csv
            The path to file exported with [`AtlasManager.saveResults()`](https://carlocastoldi.github.io/qupath-extension-braian/docs/qupath/ext/braian/AtlasManager.html#saveResults(java.util.List,java.io.File)).
            If it ends with ".csv", it treats the file as a comma-separated table. Otherwise, it assuems it is tab-separated.
            It can also be a DataFrame.
        ch2marker
            A dictionary mapping the QuPath channel names to markers.
            A cell segmentation algorithm must have previously run on each of the given channels.
        animal
            The name of the animal from which the slice was cut.
        name
            The name of the brain slice.
        ontology
            The atlas ontology used to align the section.
        check
            If True, it checks that all quantifications read in `csv` are from regions in the `atlas`,
            and then sorts them in depth-first order.
            Otherwise, it skips the check and no sorting is applied, but it's _faster_.

        Returns
        -------
        :
            A [`BrainSlice`][braian.BrainSlice]

        Raises
        ------
        EmptyResultsError
            If the given `csv` is empty
        InvalidResultsError
            If the given `csv` is an invalid file. Probably because it was not exported through
            [`qupath-extension-braian`](https://github.com/carlocastoldi/qupath-extension-braian).
        NanResultsError
            If the number of total detections in all brain regions, exported from QuPath, is undefined
        MissingResultsColumnError
            If `csv` is missing reporting the number of detection in at least one `detected_channels`.
        """
        if isinstance(csv, pd.DataFrame):
            data = csv.copy(deep=True)
            csv = "unkown file"
            data_atlas = None
        else:
            data_atlas, data = BrainSlice._read_qupath_data(search_file_or_simlink(csv))
            assert data_atlas is None or ontology.is_compatible(data_atlas),\
                f"Brain slice was expected to be aligned against '{ontology.name}', but instead found '{data_atlas}': {csv}"
        BrainSlice._check_columns(data, (QUPATH_AREA,), csv)
        all_measurements = extract_qupath_measurement_types(data)
        unique_channels = {m.measurement for m in all_measurements}
        if len(unique_channels) != len(all_measurements):
            raise ValueError("No support for multiple measurements on the same QuPath channel, even if of the diffent type (e.g. area and detection count).")
        for channel in ch2marker.keys():
            if channel not in unique_channels:
                raise MissingResultsMeasurementError(file=csv, channel=channel)
        measurements = BrainSlice._rename_selected_measurements(all_measurements, ch2marker, data)
        if any(m.type is QuPathMeasurementType.CELL_COUNT for m in measurements):
            if data["Num Detections"].count() == 0:
                raise NanResultsError(file=csv)
        for m in measurements:
            if m.type is QuPathMeasurementType.AREA and data[m.key].count() == 0:
                raise(NanResultsError(file=csv))
        BrainSlice._extract_qupath_hemispheres(data, csv)
        data = pd.DataFrame(data, columns=["Name", "hemisphere"]+[m.key for m in measurements])
        # NOTE: not needed since qupath-extension-braian>=1.0.1
        BrainSlice._fix_nan_countings(data, [m.key for m in measurements if m.type is QuPathMeasurementType.CELL_COUNT])
        data.rename(columns={"Name": "acronym"}|{m.key: m.name for m in measurements}, inplace=True)
        units = {m.name: m.unit() for m in measurements}
        return BrainSlice(data=data, units=units,
                          animal=animal, name=name,
                          ontology=ontology, check=check)

    @staticmethod
    def _extract_qupath_hemispheres(data: pd.DataFrame, csv_file: str):
        try:
            extract_legacy_hemispheres(data, reindex=False, inplace=True)
        except InvalidRegionsHemisphereError as e:
            raise InvalidRegionsHemisphereError(f"'{csv_file}': {str(e)}")
        # if (unknown_classes:=match_groups[2] != data["Name"]).any():
        #     raise ValueError("Unknown regions: '"+"', '".join(match_groups.index[unknown_classes])+"'")

    @staticmethod
    def _rename_selected_measurements(measurements: Iterable[QuPathMeasurement],
                                      ch2marker: dict[str,str],
                                      data: pd.DataFrame) -> list[QuPathMeasurement]:
        selected_measurements = []
        for m in measurements:
            if (colabelled_channels:=m.colabelled_channels()) is not None:
                ch1,ch2 = colabelled_channels
                BrainSlice._fix_double_positive_bug(data, m.key, ch1, ch2)
                try:
                    m1 = ch2marker[ch1]
                    m2 = ch2marker[ch2]
                except KeyError:
                    continue
                m.name = overlapping_markers(m1, m2)
            elif m.measurement == "area":
                m.name = "area"
            elif m.measurement not in ch2marker:
                continue
            else:
                m.name = ch2marker[m.measurement]
            selected_measurements.append(m)
        return selected_measurements

    @staticmethod
    def _fix_double_positive_bug(data: pd.DataFrame, dp_col: str, ch1: str, ch2: str):
        # ~fixes: https://github.com/carlocastoldi/qupath-extension-braian/issues/2
        # NOTE: this solution MAY reduce the total number of double positive,
        # but no better solution was found to the above issue
        ch1_col = BrainSlice._column_from_qupath_channel(ch1)
        ch2_col = BrainSlice._column_from_qupath_channel(ch2)
        maximum_overlap = data[[ch1_col, ch2_col]].min(axis=1)
        data[dp_col] = data[dp_col].clip(upper=maximum_overlap)

    @staticmethod
    def _column_from_qupath_channel(channel: str) -> str:
        return f"Num {channel}"

    @staticmethod
    def _read_qupath_data(csv_file: Path) -> tuple[str,pd.DataFrame]:
        sep = "," if csv_file.suffix.lower() == ".csv" else "\t"
        try:
            data = pd.read_csv(csv_file, sep=sep).drop_duplicates()
        except Exception:
            if csv_file.stat().st_size == 0:
                raise EmptyResultsError(file=csv_file)
            else:
                raise InvalidResultsError(file=csv_file)
        if "Class" in data.columns:
            raise ValueError("You are analyising results file exported with QuPath <0.5.x. Such files are no longer supported!")
        BrainSlice._check_columns(data, ["Name", "Classification", "Num Detections",], csv_file)
        # data = self.clean_rows(data, csv_file)
        if len(data) == 0:
            raise EmptyResultsError(file=csv_file)

        # There may be one region/row with Name == "Root" and Classification == NaN indicating the whole slice.
        # We remove it, as we want the distinction between hemispheres in the regions' acronym given by Classification column
        match (atlas_container:=data["Name"] == BrainSlice.__QUPATH_ATLAS_CONTAINER).sum():
            case 0:
                data.set_index("Classification", inplace=True)
                atlas = None
            case 1:
                root_class = data["Classification"][atlas_container].values[0]
                data.drop(data.index[atlas_container], axis=0, inplace=True, errors="raise")
                data.set_index("Classification", inplace=True)
                if not pd.isna(root_class): # QP BraiAn<1.0.4 exported data without the atlas name
                    atlas = str(root_class)
                else:
                    atlas = None
            case _:
                raise InvalidResultsError(file=csv_file)
        data.index.name = None
        return atlas, data

    @staticmethod
    def _check_columns(data: pd.DataFrame, columns: Iterable[str],
                        csv_file: str|Path) -> bool:
        for column in columns:
            if column not in data.columns:
                raise MissingResultsMeasurementError(file=csv_file, channel=column)
        return True

    @staticmethod
    def _fix_nan_countings(data, detection_columns):
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

    def __init__(self, data: pd.DataFrame, units: dict,
                 *,
                 animal:str, name: str,
                 ontology: AtlasOntology|str, check: bool) -> None:
        """
        Creates a `BrainSlice` from a [`DataFrame`][pandas.DataFrame]. Each row representes the data
        of a single brain region, whose acronym is used as index. If the data was collected
        distinguishing between the two hemispheres, the index is expected to be either `Left: <ACRONYM>`
        or `Right: <ACRONYM>`. The DataFrame is expected to have at least two columns: one named `"area"`
        corresponding to the size specified in `units`, and the others corresponding to the markers used to
        measure brain activity.

        Parameters
        ----------
        data
            The data extracted from a brain slice.
        units
            The units of measurement corresponding to each columns in `data`.
        animal
            The name of the animal from which the slice was cut.
        name
            The name of the brain slice.
        ontology
            The atlas ontology to which the data was registered to.
        check
            If True, it checks that all quantifications in `data` are from regions in the `atlas`,
            and then sorts them in depth-first order.
            Otherwise, it skips the check and no sorting is applied, but it's _faster_.

        Raises
        ------
        ValueError
            If a specified unit of measurment in `units` is unknown.
        UnknownBrainRegionsError
            If `check=True` and `ontology` is an [`AtlasOntology`][braian.AtlasOntology], but `data`
            contains structures not present in `ontology`.
        """
        self.animal: str = animal
        """The name of the animal from which the current `BrainSlice` is from."""
        self.name: str = name
        """The name of the image that captured the section from which the data of the current `BrainSlice` are from."""
        self.data = data
        self.atlas = str(ontology) if isinstance(ontology, str) else ontology.name # AtlasOntology
        """The name of the brain atlas used to align the section. If None, it means that the cell-segmented data didn't specify it."""
        BrainSlice._check_columns(self.data, ("acronym", "hemisphere", "area"), self.name)
        assert self.data.shape[1] >= 4, "'data' should have at least one column, apart from 'acronym', 'hemisphere' and 'area', containing per-region data"
        hemispheres = self.data["hemisphere"].unique()
        hemispheres = {BrainHemisphere(v) for v in hemispheres} # if not hemispheres.issubset(BrainHemisphere), it already raises an error
        if len(hemispheres) > 2 or (len(hemispheres) == 2 and BrainHemisphere.BOTH in hemispheres):
            raise InvalidRegionsSplitError(context=self.name)
        self.is_split: bool = len(hemispheres) == 2
        """Whether the data of the current `BrainSlice` make a distinction between right and left hemisphere."""
        for column, unit in units.items():
            if column == unit: # it's a cell count
                continue
            match unit:
                case "µm2" | "um2" | "um^2" | "µm^2" | "um²" | "µm²" :
                    self._µm2_to_mm2(column)
                case "mm2" | "mm²":
                    pass
                case _:
                    raise ValueError(f"Unknown unit of measurement '{unit}' for '{column}'!")
            units[column] = "mm²"
        self.units: dict[str,str] = units.copy()
        """The units of measurements corresponding to each [`marker`][braian.BrainSlice.markers] of the current `BrainSlice`."""
        assert (self.data["area"] > 0).any(), f"All region areas are zero or NaN for animal={self.animal} slice={self.name}"
        self.data = self.data[self.data["area"] > 0]
        if check:
            if not self.is_split:
                self.data.reset_index(inplace=True)
                self.data.set_index("acronym", inplace=True)
                self.data = sort_by_ontology(self.data, ontology, fill=False)
                self.data.reset_index(inplace=True)
                self.data.set_index("index", inplace=True)
            else:
                hem1 = self.data[self.data["hemisphere"] == BrainHemisphere.LEFT.value].reset_index().set_index("acronym")
                hem2 = self.data[self.data["hemisphere"] == BrainHemisphere.RIGHT.value].reset_index().set_index("acronym")
                hem1 = sort_by_ontology(hem1, ontology, fill=True)
                hem2 = sort_by_ontology(hem2, ontology, fill=True)
                self.data.reindex([*hem2["index"], *hem1["index"]], copy=False)

        self.markers_density: pd.DataFrame = self._marker_density()

    @property
    def markers(self) -> list[str]:
        """The name of the markers for which the current `BrainSlice` has data."""
        return list(self.data.columns[~self.data.columns.isin(("acronym", "hemisphere", "area"))])

    @property
    def regions(self) -> list[str]:
        """
        The list of region acronyms for which the current `BrainSlice` has data. The given order is arbitrary.
        If [`BrainSlice.is_split`][braian.BrainSlice.is_split], it contains the acronyms of the split brain region only once.
        """
        return list(pd.unique(self.data["acronym"].values))

    @deprecated(since="1.1.0", alternatives=["braian.BrainSlice.exclude"])
    def exclude_regions(self,
                        excluded_regions: Iterable[str],
                        brain_ontology: AtlasOntology,
                        exclude_parent_regions: bool):
        self.exclude(regions=excluded_regions,
                     ontology=brain_ontology,
                     ancestors=exclude_parent_regions,
                     ancestors_layer1=False)

    @deprecated(since="1.1.0", params=["ancestors"],
                message="Quantifications in ancestor regions should be completely removed too. "+\
                        "That's because the data from a region is the data from ALL of it's subregions.")
    def exclude(self,
                regions: Iterable[str],
                ontology: AtlasOntology,
                *,
                ancestors: bool=True,
                ancestors_layer1: bool=True):
        """
        Takes care of the regions to be excluded from the analysis due to issues like
        mis-alignments to the atlas, broken tissue, aquisition problems,...\\
        If `ancestors` is `True`, all regional quantifications in `regions`,
        including subregions and _ancestors_, are deleted from the current `BrainSlice`.

        If `ancestors` is `False`, instead, the quantifications are only subtracted
        from the ancestor regions of the quantifications found in the excluded `regions`.

        Parameters
        ----------
        regions
            a list of acronyms of the regions to be excluded from the analysis.
        ontology
            an ontology against whose version the brain section was aligned.
        ancestors
            Whether to completely exclude the quantifications in the parent regions too, or not.\
            If False, it keeps them after subtracting the quantifications of the excluded subregion.
        ancestors_layer1
            If False, the exclusions in layer 1 of the cortical regions will **not**
            exclude completely the quantifications in all the ancestor regions.\
            This might be useful when cortical layer 1 is expected to have _few_
            quantifications (e.g. not many detected neurons). In this case, one might
            prefer not to discard the data for the whole cortical regions in case of of
            mis-alignments of layer 1 to the atlas. Especially so if layer 1 quantifications
            don't have lots of impact on the quantifications in the cortical regions.\
            NOTE: this option is currently available only for `allen_mouse` atlases.

        Raises
        ------
        InvalidExcludedRegionsHemisphereError
            if [`BrainSlice.is_split`][braian.BrainSlice.is_split] but a region in
            `excluded_regions` is not considering left/right hemisphere distinction.
        UnknownBrainRegionsError
            if a region in `excluded_regions` is not recognised from `brain_ontology`.
        ExcludedAllRegionsError
            if there is no cell count left after the exclusion is done.
        """
        if ancestors and not ancestors_layer1:
            assert re.match(r"(allen|silvalab)_mouse_(10|25|50|100)um", ontology.name), \
                    f"Could not extract layer 1 of the cortex. Incompatible atlas: '{ontology.name}'."+\
                    " If you think think BraiAn should support this atlas, please open an issue on https://codeberg.org/SilvaLab/BraiAn"
            layer1 = [ontology.subregions(s, blacklisted=True, unreferenced=False) for s in
                        itertools.chain(ontology.subregions("Isocortex"), ontology.subregions("OLF"))
                        if s.endswith("1")]
            layer1 = {s for layer1_subs in layer1 for s in layer1_subs}
        try:
            for exclusion in regions:
                if self.is_split:
                    if ": " not in exclusion:
                        if MODE_ExcludedRegionNotRecognisedError != "silent":
                            print(f"WARNING: Class '{exclusion}' is not recognised as a brain region. It was skipped from the regions_to_exclude in animal '{self.animal}', file: {self.name}_regions_to_exclude.txt")
                            continue
                        elif MODE_ExcludedRegionNotRecognisedError == "error":
                            raise InvalidExcludedRegionsHemisphereError(file=self.name)
                    hem, region = exclusion.split(": ")
                    # hem = BrainHemisphere(hem)
                else:
                    region = exclusion
                if not ontology.is_region(region):
                    raise UnknownBrainRegionsError((region,))

                # Step 1: subtract counting results of the regions to be excluded
                # from their parent regions.
                _ancestors = ontology.ancestors(region, key="acronym")
                for parent_region in _ancestors:
                    row = hem+": "+parent_region if self.is_split else parent_region
                    if row in self.data.index:
                        if ancestors and (ancestors_layer1 or region not in layer1):
                            self.data.drop(row, inplace=True)
                        elif exclusion in self.data.index:
                            # Subtract the quantification results from the parent region.
                            # Use fill_value=0 to prevent "3-NaN=NaN".
                            for column in self.data.columns:
                                if not is_numeric_dtype(self.data[column]) or column == "hemisphere":
                                    continue
                                if pd.isna(self.data.loc[row,column]) or pd.isna(self.data.loc[exclusion,column]):
                                    # NOTE: remove if never used.
                                    # This check is a leftover of the old <row>.subtract(<exclusion>, fill_value=True)
                                    raise RuntimeError(f"{self.name}: quantification '{column}' is NaN for '{row}' or for '{exclusion}'!")
                                self.data.loc[row,column] -= self.data.loc[exclusion,column]

                # Step 2: Remove the regions that should be excluded
                # together with their daughter regions.
                subregions = ontology.subregions(region, blacklisted=True, unreferenced=True, key="acronym")
                for subregion in subregions:
                    row = hem+": "+subregion if self.is_split else subregion
                    if row in self.data.index:
                        self.data.drop(row, inplace=True)
        except Exception:
            raise Exception(f"Animal '{self.animal}': failed to exclude regions for in slice '{self.name}'")
        finally:
            self.markers_density = self._marker_density()
        if len(self.data) == 0:
            raise ExcludedAllRegionsError(file=self.name)

    def _µm2_to_mm2(self, column) -> None:
        self.data[column] = self.data[column] * 1e-06

    def _marker_density(self) -> pd.DataFrame:
        densities = self.data[self.markers].div(self.data["area"], axis=0)
        densities["acronym"] = self.data["acronym"]
        densities["hemisphere"] = self.data["hemisphere"]
        return densities

    def merge_hemispheres(self) -> Self:
        """
        For each brain region, sums the data of left and right hemispheres into one single datum

        Returns
        -------
        :
            A new [`BrainSlice`][braian.BrainSlice] with no hemisphere distinction.
            If `slice` is already merged, it return the same instance with no changes.
        """
        if not self.is_split:
            return self
        # corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in self.data.index]
        data = self.data.groupby("acronym").sum(min_count=1)
        data["hemisphere"] = BrainHemisphere.BOTH.value
        data["acronym"] = data.index
        data.index = data.index.set_names(None)
        return BrainSlice(data=data, units=self.units,
                          animal=self.animal, name=self.name,
                          ontology=self.atlas, check=False)

    def region(self, acronym: str, metric: str, as_density: bool=False) -> Sequence:
        """
        Extract the values of a brain region from the current `BrainSlice`.

        Parameters
        ----------
        acronym
            The acronym of a brain region.
        metric
            The metric to extract from the current `SlicedBrain`.
            It can either be `"area"` or any value in [`SlicedBrain.markers`][braian.SlicedBrain.markers].
        as_density
            If `True`, it retrieves the values as densities instead (i.e. marker/area).

        Returns
        -------
        :
            If the current `BrainSlice` is not split between right and left hemisphers, it returns a sequence of length 1.
            Else, it returns a sequence of length 2.

            If there is no value of `region`, it returns an empty sequence.

        Raises
        ------
        ValueError
            If `marker` is not recognised as a source of data for the current `BrainSlice`.
        ValueError
            If `marker="area"` and `as_density=True`.
        """
        if metric != "area" and metric not in self.markers:
            raise ValueError(f"Invalid metric: '{metric}'. Valid metrics are: {['area', *self.markers]}")
        if metric == "area" and as_density:
            raise ValueError("Cannot request to retrieve values as densities when metric='area'. Choose a metric within the available markers.")
        df = self.markers_density if as_density else self.data
        return np.array(df[df["acronym"] == acronym][metric].values)
