import igraph as ig
import itertools
import pandas as pd
import re

from dataclasses import dataclass, field
from enum import Enum
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from typing import Self
from collections.abc import Iterable, Sequence

from braian import AtlasOntology, BrainHemisphere, UnknownBrainRegionsError, InvalidRegionsHemisphereError
from braian._animal_brain import colabelled_markers
from braian._brain_data import extract_legacy_hemispheres
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
    "MissingQuantificationError",
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
class MissingQuantificationError(BrainSliceFileError):
    def __init__(self, measure=None, **kargs: object) -> None:
        self.quantification = "a marker" if measure is None else f"'{measure}'"
        super().__init__(**kargs)
    def __str__(self):
        return f"Quantification missing for {self.quantification} in: {self.file_path}"
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

class QuPathMeasurementType(Enum):
    """
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

@dataclass(frozen=True)
class QuPathMeasurement:
    r"""Class for identifying measurements extracted with QuPath."""
    key: str
    measurement: str
    type: QuPathMeasurementType
    name: str = field(default=None, init=False)
    channels: list[str] = field(init=False)

    def __post_init__(self):
        channels = self.measurement.split("~")
        object.__setattr__(self, "channels", channels)

    def label(self, name: str):
        object.__setattr__(self, "name", name)

    # def colabelled_channels(self) -> tuple[str,str]:
    #     match = re.match(r"(.+)\~(.+)", self.measurement)
    #     if match is None:
    #         return None
    #     channels = match.groups()
    #     if len(channels) == 2:
    #         return channels

    def unit(self) -> str:
        match self.type:
            case QuPathMeasurementType.AREA:
                return "µm²"
            case QuPathMeasurementType.CELL_COUNT:
                assert self.name is not None, "Measurement is not labelled"
                return self.name
            case _:
                raise ValueError(f"Unknown QuPath measurement type: '{self.type}'")


class ColabellingHierarchy:
    def __init__(self, measurements: list[QuPathMeasurement]):
        assert all(m.type is QuPathMeasurementType.CELL_COUNT for m in measurements), "Colabelling supported among cell count measurements only"
        self._g: ig.Graph = ig.Graph(directed=True)
        # first add the 'primitive' measurements
        measurements = sorted(measurements, key=lambda m: len(m.channels))
        for m in measurements:
            n = len(m.channels)
            primitives = self._g.vs.select(n_eq=n-1) if n > 1 else []
            if n > 1 and len(primitives) == 0:
                print(f"WARNING: no primitive channels found for colabelled quantifications '{m.measurement}'")
                continue
            child = self._g.add_vertex(m.measurement, n=n, data=m)
            for parent in primitives:
                # assumes there is already a measurement for each primitive measurements
                if not set(parent["data"].channels).issubset(m.channels):
                    continue
                self._g.add_edge(parent, child)

    def __getitem__(self, value: str) -> QuPathMeasurement:
        return self._g.vs.find(name_eq=value)["data"]

    def prime(self) -> list[QuPathMeasurement]:
        return [v["data"] for v in self._g.vs.select(n_eq=1)]

    def primitives(self, colabel: str|QuPathMeasurement|ig.Vertex) -> list[QuPathMeasurement]:
        if isinstance(colabel, QuPathMeasurement):
            colabel = colabel.measurement
        if isinstance(colabel, str):
            colabel = self._g.vs.find(name_eq=colabel)
        components = self._g.subcomponent(colabel, mode="in")
        return [primitive["data"] for primitive in self._g.vs[components] if primitive != colabel]

    def inherit_colabellings(self, data: pd.DataFrame):
        # BraiAn for QuPath re-classifies double-positive detections that are also triple-positive (or more).
        # This means that triple-positive (or more) quantifications have to be added to those that are only double-positive.
        # However, colabellings of rank 1 (i.e. those with no overlap) don't need to inherit from rank 2
        for n in range(max(self._g.vs["n"]), 2, -1):
            for colabel in self._g.vs.select(n_eq=n):
                r2_primitives = [p.key for p in self.primitives(colabel) if len(p.channels) > 1]
                data[r2_primitives] = data[r2_primitives].add(data[colabel["data"].key], axis=0)

    def fix_colabelling_bug(self, data: pd.DataFrame):
        # ~fixes: https://github.com/carlocastoldi/qupath-extension-braian/issues/2
        # NOTE: this solution MAY reduce the total number of colabellings,
        # but no better solution was found to the above issue
        for n in range(2, max(self._g.vs["n"])+1):
            for colabel in self._g.vs.select(n_eq=n):
                qupath_col = colabel["data"].key
                primitives_vs = self._g.vs[self._g.subcomponent(colabel, mode="in")]
                primitives = [primitive["data"].key for primitive in primitives_vs if primitive != colabel]
                max_colabelled = data[primitives].min(axis=1)
                data[qupath_col] = data[qupath_col].clip(upper=max_colabelled)

    def label(self, ch2marker: dict[str,str]):
        primes = self._g.vs.select(name_in=ch2marker)
        for prime in primes:
            prime["data"].label(ch2marker[prime["name"]])
        for colabel in self._g.vs.select(n_ne=1):
            colabel: QuPathMeasurement = colabel["data"]
            primes = colabel.channels
            prime_names = [self[ch].name for ch in primes]
            if any(name is None for name in prime_names):
                continue
            colabel.label(colabelled_markers(*prime_names))

    def labelled(self) -> list[QuPathMeasurement]:
        return [m["data"] for m in self._g.vs if m["data"].name is not None]

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
                    ontology: AtlasOntology) -> Self:
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
                raise MissingQuantificationError(file=csv, measure=channel)
        m_cellcounts   = [m for m in all_measurements if m.type is QuPathMeasurementType.CELL_COUNT]
        m_others       = [m for m in all_measurements if m.type is not QuPathMeasurementType.CELL_COUNT]
        mhierarchy = ColabellingHierarchy(m_cellcounts)
        mhierarchy.inherit_colabellings(data)
        mhierarchy.fix_colabelling_bug(data)
        mhierarchy.label(ch2marker) # modifies 'all_measurements'
        BrainSlice._label_measurements(m_others, ch2marker)
        # only select the measurements for which a label was given
        measurements = [m for m in all_measurements if m.name is not None]
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
                          ontology=ontology, check=True) # we're reading from unknown sources, so we should always check

    @staticmethod
    def _extract_qupath_hemispheres(data: pd.DataFrame, csv_file: str):
        try:
            extract_legacy_hemispheres(data, reindex=False, inplace=True)
            return
        except InvalidRegionsHemisphereError as e:
            message = str(e) # add path to the CSV file
        raise InvalidRegionsHemisphereError(f"'{csv_file}': {message}")
        # if (unknown_classes:=match_groups[2] != data["Name"]).any():
        #     raise ValueError("Unknown regions: '"+"', '".join(match_groups.index[unknown_classes])+"'")

    @staticmethod
    def _label_measurements(measurements: Iterable[QuPathMeasurement],
                             ch2marker: dict[str,str]) -> list[QuPathMeasurement]:
        for m in measurements:
            if m.measurement == "area":
                m.label("area")
            elif m.measurement not in ch2marker:
                continue
            else:
                m.label(ch2marker[m.measurement])

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
        """
        Raises
        ------
        MissingResultsMeasurementError
            When `data` misses at least column in `columns`
        """
        for column in columns:
            if column not in data.columns:
                raise MissingQuantificationError(file=csv_file, measure=column)
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
    #         zero_rows = self._data[marker] == 0
    #         if sum(zero_rows) > 0:
    #             err = RegionsWithNoCountError(slice=self, file=csv_file,
    #                         tracer=marker, regions=self._data.index[zero_rows].to_list())
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
        self._animal: str = str(animal)
        self._name: str = str(name)
        self._data = data
        BrainSlice._check_columns(self._data, ("acronym", "hemisphere", "area"), self._name)
        if self._data.shape[1] < 4:
            raise MissingQuantificationError(file=self._name)
        hemispheres = self._data["hemisphere"].unique()
        hemispheres = {BrainHemisphere(v) for v in hemispheres} # if not hemispheres.issubset(BrainHemisphere), it already raises an error
        if len(hemispheres) > 2 or (len(hemispheres) == 2 and BrainHemisphere.MERGED in hemispheres):
            raise InvalidRegionsSplitError(context=self._name)
        self._is_split: bool = len(hemispheres) == 2 or BrainHemisphere.MERGED not in hemispheres
        for column in self._data.columns:
            if column in ("acronym", "hemisphere"):
                continue
            if column not in units:
                raise ValueError(f"Missing unit of measurement for '{column}'")
            unit = units[column]
            if column == unit: # it's a cell count
                continue
            match unit:
                case "µm2" | "um2" | "um^2" | "µm^2" | "um²" | "µm²" :
                    self._µm2_to_mm2(column)
                case "mm2" | "mm^2" | "mm²":
                    pass
                case _:
                    raise ValueError(f"Unknown unit of measurement '{unit}' for '{column}'")
            units[column] = "mm²"
        self._units: dict[str,str] = units.copy()
        assert (self._data["area"] > 0).any(), f"All region areas are zero or NaN for animal={self._animal} slice={self._name}"
        self._data = self._data[self._data["area"] > 0]
        if check:
            if not isinstance(ontology, AtlasOntology):
                raise TypeError(type(ontology))
            elif not self.is_split:
                self._data.reset_index(inplace=True)
                self._data.set_index("acronym", inplace=True)
                self._data = ontology.sort(self._data, mode="depth", fill=False)
                self._data.reset_index(inplace=True)
                self._data.set_index("index", inplace=True)
            else:
                hem1 = self._data[self._data["hemisphere"] == BrainHemisphere.LEFT.value].reset_index().set_index("acronym")
                hem2 = self._data[self._data["hemisphere"] == BrainHemisphere.RIGHT.value].reset_index().set_index("acronym")
                hem1 = ontology.sort(hem1, mode="depth", fill=False)
                hem2 = ontology.sort(hem2, mode="depth", fill=False)
                self._data.reindex([*hem2["index"], *hem1["index"]], copy=False)
        self._atlas = str(ontology) if isinstance(ontology, str) else ontology.name # : AtlasOntology

        self.markers_density: pd.DataFrame = self._marker_density()

    @property
    def atlas(self) -> str:
        """The name of the brain atlas used to align the section. If None, it means that the cell-segmented data didn't specify it."""
        return str(self._atlas)

    @property
    def name(self) -> str:
        """The name of the image that captured the section from which the data of the current `BrainSlice` are from."""
        return str(self._name)

    @property
    def animal(self) -> str:
        """The name of the animal from which the current `BrainSlice` is from."""
        return self._animal

    @property
    def n(self) -> int:
        """The number of unique regions in the slice, without distinction between hemispheres."""
        return len(pd.unique(self._data["acronym"].values))

    @property
    def is_split(self) -> bool:
        """Whether or not the regional brain data have distinction between right and left hemisphere."""
        return self._is_split

    @property
    def markers(self) -> list[str]:
        """The name of the markers for which the current `BrainSlice` has data."""
        return list(self._data.columns[~self._data.columns.isin(("acronym", "hemisphere", "area"))])

    @property
    def units(self) -> dict[str,str]:
        """The units of measurements corresponding to each [`marker`][braian.BrainSlice.markers] of the current `BrainSlice`."""
        return self._units.copy()

    @property
    def regions(self) -> list[str]:
        """
        The list of region acronyms for which the current `BrainSlice` has data. The given order is arbitrary.
        If [`BrainSlice.is_split`][braian.BrainSlice.is_split], it contains the acronyms of the split brain region only once.
        """
        return list(pd.unique(self._data["acronym"].values))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"BrainSlice(name='{self._name}', regions={self.n}, is_split={self._is_split})"

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
            prefer not to discard the data for the whole cortical regions in case of
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
                            print(f"WARNING: Class '{exclusion}' is not recognised as a brain region. It was skipped from the regions_to_exclude in animal '{self._animal}', file: {self._name}_regions_to_exclude.txt")
                            continue
                        elif MODE_ExcludedRegionNotRecognisedError == "error":
                            raise InvalidExcludedRegionsHemisphereError(file=self._name)
                    hem, region = exclusion.split(": ")
                    # hem = BrainHemisphere(hem)
                else:
                    region = exclusion
                if not ontology.is_region(region):
                    raise UnknownBrainRegionsError((region,), ontology)

                # Step 1: subtract counting results of the regions to be excluded
                # from their parent regions.
                _ancestors = ontology.ancestors(region, key="acronym")
                for parent_region in _ancestors:
                    row = hem+": "+parent_region if self.is_split else parent_region
                    if row in self._data.index:
                        if ancestors and (ancestors_layer1 or region not in layer1):
                            self._data.drop(row, inplace=True)
                        elif exclusion in self._data.index:
                            # Subtract the quantification results from the parent region.
                            # Use fill_value=0 to prevent "3-NaN=NaN".
                            for column in self._data.columns:
                                if not is_numeric_dtype(self._data[column]) or column == "hemisphere":
                                    continue
                                if pd.isna(self._data.loc[row,column]) or pd.isna(self._data.loc[exclusion,column]):
                                    # NOTE: remove if never used.
                                    # This check is a leftover of the old <row>.subtract(<exclusion>, fill_value=True)
                                    raise RuntimeError(f"{self._name}: quantification '{column}' is NaN for '{row}' or for '{exclusion}'!")
                                self._data.loc[row,column] -= self._data.loc[exclusion,column]

                # Step 2: Remove the regions that should be excluded
                # together with their daughter regions.
                subregions = ontology.subregions(region, blacklisted=True, unreferenced=True, key="acronym")
                for subregion in subregions:
                    row = hem+": "+subregion if self.is_split else subregion
                    if row in self._data.index:
                        self._data.drop(row, inplace=True)
        except Exception:
            raise Exception(f"Animal '{self._animal}': failed to exclude regions for in slice '{self._name}'")
        finally:
            self.markers_density = self._marker_density()
        if len(self._data) == 0:
            raise ExcludedAllRegionsError(file=self._name)

    def _µm2_to_mm2(self, column) -> None:
        self._data[column] = self._data[column] * 1e-06

    def _marker_density(self) -> pd.DataFrame:
        densities = self._data[self.markers].div(self._data["area"], axis=0)
        densities["acronym"] = self._data["acronym"]
        densities["hemisphere"] = self._data["hemisphere"]
        return densities

    def merge_hemispheres(self) -> Self:
        """
        Merges data from left and right hemisheres into a single value, by sum.\\
        If it's already merged, it returns the same instance.

        Returns
        -------
        :
            The regional data from `d` with no distinction between hemispheres.

        See also
        --------
        [`merge.merge_hemispheres`][braian.merge_hemispheres]
        [`BrainData.merge`][braian.BrainData.merge]
        [`AnimalBrain.merge_hemispheres`][braian.AnimalBrain.merge_hemispheres]
        [`SlicedBrain.merge_hemispheres`][braian.SlicedBrain.merge_hemispheres]
        """
        if not self.is_split:
            return self
            # raise ValueError("Data already have no distinction between right/left hemispheres")
        # corresponding_region = [extract_acronym(hemisphered_region) for hemisphered_region in self._data.index]
        data = self._data.groupby("acronym").sum(min_count=1)
        data["hemisphere"] = BrainHemisphere.MERGED.value
        data["acronym"] = data.index
        data.index = data.index.set_names(None)
        return BrainSlice(data=data, units=self._units,
                          animal=self._animal, name=self._name,
                          ontology=self._atlas, check=False)

    def region(self,
               region: str,
               *,
               metric: str|Sequence[str],
               hemisphere: BrainHemisphere=BrainHemisphere.MERGED,
               as_density: bool=False,
               return_hemispheres: bool=False
               ) -> int|float|pd.DataFrame:
        """
        Extract the values of a brain region from a `BrainSlice`.

        Parameters
        ----------
        region
            A brain structure identified by its acronym.
        metric
            The metric to extract from the `BrainSlice`.
            It can be any value within the slice's [`markers`][braian.BrainSlice.markers] and `"area"`.
        hemisphere
            The hemisphere of the brain region to extract. If [`MERGED`][braian.BrainHemisphere]
            and the brain [is split][braian.BrainSlice.is_split], it may return both hemispheric values
            of the region.
        as_density
            If `True`, it retrieves the values as densities (i.e. marker/area).

        Returns
        -------
        : pd.DataFrame
            A `DataFrame` with `metric` as column(s). If `return_hemispheres=True` or if the slice is
            [split][braian.BrainSlice.is_split] between right and left hemispheres, there is also
            a `"hemisphere"` column.

            If there is no data for `region`, it always returns an empty `DataFrame`.

        : int|float
            A single value is returned if a single `metric` is given, and data can only be extracted for one hemisphere
            (either because the slice is not [split][braian.BrainSlice.is_split] between right and left hemispheres,
            or because only one of the two `hemisphere`s is specified).

        Raises
        ------
        ValueError
            If `marker` is not recognised as a source of data for the current `BrainSlice`.
        ValueError
            If `marker="area"` and `as_density=True`.
        """
        if isinstance(metric, str):
            metrics = (metric,)
        else: # it's a Sequence[str]
            metrics = metric
        for metric in metrics:
            if metric != "area" and metric not in self.markers:
                raise ValueError(f"Invalid metric: '{metric}'. Valid metrics are: {['area', *self.markers]}")
            if metric == "area" and as_density:
                raise ValueError("Cannot request to retrieve values as densities for metric='area'")
        if self.is_split and hemisphere is BrainHemisphere.MERGED:
            hems = (BrainHemisphere.RIGHT.value, BrainHemisphere.LEFT.value)
        else:
            hems = (hemisphere.value,)
        df = self.markers_density if as_density else self._data
        filtered = df[(df["acronym"] == region) & (df["hemisphere"].isin(hems))]
        columns = ["hemisphere", *metrics] if self.is_split or return_hemispheres else list(metrics)
        if not return_hemispheres and len(filtered) > 0 and len(metrics) == 1 and len(hems) == 1:
            return filtered[metric].values[0]
        else:
            return filtered.reset_index(drop=True)[columns]
