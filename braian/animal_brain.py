import copy
import numpy as np
import pandas as pd
import re

from collections.abc import Sequence
from enum import Enum, auto
from pandas.core.groupby import DataFrameGroupBy
from pathlib import Path
from typing import Generator, Self

from braian.ontology import AllenBrainOntology
from braian.sliced_brain import SlicedBrain, EmptyBrainError
from braian.brain_data import BrainData
from braian.utils import save_csv

__all__ = ["AnimalBrain", "SliceMetrics"]

# https://en.wikipedia.org/wiki/Coefficient_of_variation
def coefficient_variation(x: np.ndarray) -> np.float64:
    if x.ndim == 1:
        avg = x.mean()
        if len(x) > 1 and avg != 0:
            return x.std(ddof=1) / avg
        else:
            return 0
    else: # compute it for each column of the DataFrame and return a Series
        return x.apply(coefficient_variation, axis=0)

class SliceMetrics(Enum):
    r"""
    Enum of the metrics used to reduce region data from [`SlicedBrain`][braian.SlicedBrain]
    into a [`AnimalBrain`][braian.AnimalBrain].

    Attributes
    ----------
    SUM
        Computes the sum of all the sections data from the same region into a single value
    MEAN
        Computes the average $\mu$ of all the sections data from the same region into a single value
    STD
        Computes the standard deviation $\sigma$ between all the sections data from the same region into a single value
    CVAR
        Computes the [coefficient of variation](https://en.wikipedia.org/wiki/Coefficient_of_variation)
        $\frac \mu \sigma$ between all the sections data from the same region into a single value
    """
    SUM = auto()
    MEAN = auto()
    STD = auto()
    CVAR = auto()

    @property
    def _raw(self) -> bool:
        return self in (SliceMetrics.SUM, SliceMetrics.MEAN)

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f'<{cls_name}.{self.name}>'

    def __str__(self):
        return self.name.lower()

    def __format__(self, format_spec: str):
        return repr(self)

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None
        match value.lower():
            case "sum":
                return SliceMetrics.SUM
            case "avg" | "mean":
                return SliceMetrics.MEAN
            case "variation" | "cvar" | "coefficient of variation":
                return SliceMetrics.CVAR
            case "std" | "standard deviation":
                return SliceMetrics.STD

    def apply(self, grouped_by_region: DataFrameGroupBy):
        match self:
            case SliceMetrics.SUM:
                return grouped_by_region.sum()
            case SliceMetrics.MEAN:
                return grouped_by_region.mean()
            case SliceMetrics.STD:
                return grouped_by_region.std(ddof=1)
            case SliceMetrics.CVAR:
                return grouped_by_region.apply(coefficient_variation)
            case _:
                raise ValueError(f"{self} does not support BrainSlices reductions")

    def __call__(self, sliced_brain: SlicedBrain, min_slices: int, densities: bool):
        all_slices = sliced_brain.concat_slices(densities=densities)
        all_slices = all_slices.groupby(all_slices.index).filter(lambda g: len(g) >= min_slices)
        raw = not densities and self._raw
        return self.apply(all_slices.groupby(all_slices.index)), raw

class AnimalBrain:
    @staticmethod
    def from_slices(sliced_brain: SlicedBrain,
                    metric: SliceMetrics|str=SliceMetrics.SUM, min_slices: int=0,
                    hemisphere_distinction: bool=True, densities: bool=False) -> Self:
        """
        Crates a cohesive [`AnimalBrain`][braian.AnimalBrain] from data coming from brain sections.

        Parameters
        ----------
        sliced_brain
            A sectioned brain.
        metric
            The metric used to reduce sections data from the same region into a single value.
        min_slices
            The minimum number of sections for a reduction to be valid.
            If a region has not enough sections, it will disappear from the dataset.
        hemisphere_distinction
            if False and `sliced_brain` is split between left/right hemispheres,
            it first merges, for each section, the hemispheric data.
        densities
            If True, it computes the reduction on the section density (i.e., marker/area)
            instead of doing it on the raw cell counts.

        Returns
        -------
        :
            An `AnimalBrain`.

        Raises
        ------
        EmptyBrainError
            when `sliced_brain` has not enough sections or when `min_slices` filters out all brain regions.
        """
        if not hemisphere_distinction:
            sliced_brain = SlicedBrain.merge_hemispheres(sliced_brain)

        name = sliced_brain.name
        markers = copy.copy(sliced_brain.markers)
        metric = SliceMetrics(metric)
        if len(sliced_brain.slices) < min_slices:
            raise EmptyBrainError(sliced_brain.name)
        redux, raw = metric(sliced_brain, min_slices, densities=densities)
        if redux.shape[0] == 0:
            raise EmptyBrainError(sliced_brain.name)
        metric = f"{str(metric)}_densities" if densities else str(metric)
        areas = BrainData(redux["area"], name=name, metric=metric, units="mm²")
        markers_data = {
            m: BrainData(redux[m], name=name, metric=metric, units=m)
            for m in markers
        }
        return AnimalBrain(markers_data=markers_data, areas=areas, raw=raw)

    def __init__(self, markers_data: dict[str,BrainData], areas: BrainData, raw: bool=False) -> None:
        """
        Associates [`BrainData`][braian.BrainData] coming from a single subject,
        for each marker and for each brain region.

        Parameters
        ----------
        markers_data
            A dictionary that associates the name of a marker to a `BrainData`
        areas
            A `BrainData` with the size of the subject's brain regions.
        raw
            Whether the data can be considered _raw_ (e.g., contains simple cell positive counts) or not.
        """
        assert len(markers_data) > 0 and areas is not None, "You must provide both a dictionary of BrainData (markers) and an additional BrainData for the areas/volumes of each region"
        self.markers: tuple[str] = tuple(markers_data.keys())
        """The name of the markers for which the current `AnimalBrain` has data."""
        self.markers_data = markers_data
        self.areas: BrainData = areas
        """The data corresponding to the size of each brain region of the current AnimalBrain."""
        self.raw: bool = raw
        """Whether the data can be considered _raw_ (e.g., contains simple cell positive counts) or not."""
        assert all([m.data_name == self.name for m in markers_data.values()]), "All markers' BrainData must be from the same animal!"
        assert all([m.metric == self.metric for m in markers_data.values()]), "All markers' BrainData must have the same metric!"
        assert all([m.is_split == self.is_split for m in markers_data.values()]), "Markers' BrainData must either all have split hemispheres or none!"
        assert self.is_split == areas.is_split, "Markers' and areas' BrainData must either both have split hemispheres or none!"
        return

    @property
    def metric(self) -> str:
        """
        The name of the metric used to compute current data.
        Equals to [`RAW_TYPE`][braian.BrainData.RAW_TYPE] if no previous normalization was preformed.
        """
        return self.markers_data[self.markers[0]].metric

    @property
    def is_split(self) -> bool:
        """Whether the data of the current `AnimalBrain` makes a distinction between right and left hemisphere."""
        return self.markers_data[self.markers[0]].is_split

    @property
    def name(self) -> str:
        """The name of the animal."""
        return self.markers_data[self.markers[0]].data_name

    @property
    def regions(self) -> list[str]:
        """The list of region acronyms for which the current `AnimalBrain` has data."""
        # assumes areas' and all markers' BrainData are synchronized
        return self.areas.regions

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AnimalBrain(name='{self.name}', metric={self.metric}, markers={list(self.markers)})"

    def __getitem__(self, marker: str):
        """
        Get the [`BrainData`][braian.BrainData] associated to `marker`.
        Fails if there is no data for the the given marker

        Parameters
        ----------
        marker
            The marker to extract the data for.

        Returns
        -------
        :
            The data associated to `marker`.
        """
        return self.markers_data[marker]

    def remove_region(self, region: str, *regions, fill_nan: bool=True) -> None:
        """
        Removes the data from all the given regions in the current `AnimalBrain`

        Parameters
        ----------
        region
            The acronyms of the regions to exclude from the data.
        fill_nan
            If True, instead of removing the regions completely, it fills their value to [`NA`][pandas.NA].
        """
        regions = (region, *regions)
        for data in self.markers_data.values():
            data.remove_region(*regions, inplace=True, fill_nan=fill_nan)
        self.areas.remove_region(*regions, inplace=True, fill_nan=fill_nan)

    def remove_missing(self) -> None:
        """
        Removes the regions for which there is no data about the size.
        """
        self.remove_region(*self.areas.missing_regions(), fill_nan=False)

    def sort_by_ontology(self, brain_ontology: AllenBrainOntology,
                         fill_nan: bool=False, inplace: bool=False) -> Self:
        """
        Sorts the data in depth-first search order with respect to `brain_ontology`'s hierarchy.

        Parameters
        ----------
        brain_ontology
            The ontology to which the current data was registered against.
        fill_nan
            If True, it sets the value to [`NA`][pandas.NA] for all the regions in
            `brain_ontology` missing in the current `AnimalBrain`.
        inplace
            If True, it applies the sorting to the current instance.

        Returns
        -------
        :
            A brain with data sorted accordingly to `brain_ontology`.
            If `inplace=True` it returns the same instance.
        """
        markers_data = {marker: m_data.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=inplace)
                        for marker, m_data in self.markers_data.items()}
        areas = self.areas.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=inplace)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, areas=areas, raw=self.raw)
        else:
            return self

    def select_from_list(self, regions: Sequence[str], fill_nan: bool=False, inplace: bool=False) -> Self:
        """
        Filters the data from a given list of regions.

        Parameters
        ----------
        regions
            The acronyms of the regions to select from the data.
        fill_nan
            If True, the regions missing from the current data are filled with [`NA`][pandas.NA].
            Otherwise, if the data from some regions are missing, they are ignored.
        inplace
            If True, it applies the filtering to the current instance.

        Returns
        -------
        :
            A brain with data filtered accordingly to the given `regions`.
            If `inplace=True` it returns the same instance.

        See also
        --------
        [`AnimalBrain.select_from_ontology`][braian.AnimalBrain.select_from_ontology]
        """
        markers_data = {marker: m_data.select_from_list(regions, fill_nan=fill_nan, inplace=inplace)
                        for marker, m_data in self.markers_data.items()}
        areas = self.areas.select_from_list(regions, fill_nan=fill_nan, inplace=inplace)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, areas=areas, raw=self.raw)
        else:
            return self

    def select_from_ontology(self, brain_ontology: AllenBrainOntology, fill_nan: bool=False, inplace: bool=False) -> Self:
        """
        Filters the data from a given ontology, accordingly to a non-overlapping list of regions
        previously selected in `brain_ontology`.\\
        It fails if no selection method was called on the ontology.

        Parameters
        ----------
        brain_ontology
            The ontology to which the current data was registered against.
        fill_nan
            If True, the regions missing from the current data are filled with [`NA`][pandas.NA].
            Otherwise, if the data from some regions are missing, they are ignored.
        inplace
            If True, it applies the filtering to the current instance.

        Returns
        -------
        :
            A brain with data filtered accordingly to the given ontology selection.
            If `inplace=True` it returns the same instance.

        See also
        --------
        [`AnimalBrain.select_from_list`][braian.AnimalBrain.select_from_list]
        [`AllenBrainOntology.get_selected_regions`][braian.AllenBrainOntology.get_selected_regions]
        [`AllenBrainOntology.unselect_all`][braian.AllenBrainOntology.unselect_all]
        [`AllenBrainOntology.add_to_selection`][braian.AllenBrainOntology.add_to_selection]
        [`AllenBrainOntology.select_at_depth`][braian.AllenBrainOntology.select_at_depth]
        [`AllenBrainOntology.select_at_structural_level`][braian.AllenBrainOntology.select_at_structural_level]
        [`AllenBrainOntology.select_leaves`][braian.AllenBrainOntology.select_leaves]
        [`AllenBrainOntology.select_summary_structures`][braian.AllenBrainOntology.select_summary_structures]
        [`AllenBrainOntology.select_regions`][braian.AllenBrainOntology.select_regions]
        [`AllenBrainOntology.get_regions`][braian.AllenBrainOntology.get_regions]
        """
        assert brain_ontology.has_selection(), "No selection found in the given ontology."
        selected_allen_regions = brain_ontology.get_selected_regions()
        if not fill_nan:
            selectable_regions = set(self.regions).intersection(set(selected_allen_regions))
        else:
            selectable_regions = selected_allen_regions
        return self.select_from_list(list(selectable_regions), fill_nan=fill_nan, inplace=inplace)

    def get_units(self, marker:str|None=None) -> str:
        """
        Returns the units of measurment of a marker.

        Parameters
        ----------
        marker
            The marker to get the units for. It can be omitted, if the current brain has only one marker.

        Returns
        -------
        :
            A string representing the units of measurement of `marker`.
        """
        if len(self.markers) == 1:
            marker = self.markers[0]
        else:
            assert marker in self.markers, f"Could not get units for marker '{marker}'!"
        return self.markers_data[marker].units

    def to_pandas(self, units: bool=False) -> pd.DataFrame:
        """
        Converts the current `AnimalBrain` to a DataFrame. T

        Parameters
        ----------
        units
            Whether the columns should include the units of measurement or not.

        Returns
        -------
        :
            A DataFrame where the rows are the brain regions, the first column is the size
            of the regions, while the other columns contains the data for each marker.
            The columns' name is the name of the metric used.

        See also
        --------
        [`from_pandas`][braian.AnimalBrain.from_pandas]
        """
        data = pd.concat({f"area ({self.areas.units})" if units else "area": self.areas.data,
                          **{f"{m} ({m_data.units})" if units else m: m_data.data for m,m_data in self.markers_data.items()}}, axis=1)
        data.columns.name = str(self.metric)
        return data

    def to_csv(self, output_path: Path|str, sep: str=",", overwrite: bool=False) -> str:
        """
        Write the current `AnimalBrain` to a comma-separated values (CSV) file in `output_path`.

        Parameters
        ----------
        output_path
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        sep
            Character to treat as the delimiter.
        overwrite
            If True, it overwrite any conflicting file in `output_path`.

        Returns
        -------
        :
            The file path to the saved CSV file.

        Raises
        ------
        FileExistsError
            If `overwrite=False` and there is a conflicting file in `output_path`.

        See also
        --------
        [`from_csv`][braian.AnimalBrain.from_csv]
        """
        df = self.to_pandas(units=True)
        file_name = f"{self.name}_{self.metric}.csv"
        return save_csv(df, output_path, file_name, overwrite=overwrite, sep=sep, index_label=df.columns.name)

    @staticmethod
    def is_raw(metric: str) -> bool:
        """
        Test whether the given string can be associated to a raw metric or not.

        Parameters
        ----------
        metric
            A string representing the name of a metric.

        Returns
        -------
        :
            True, if the given string is associated to a raw metric. Otherwise, False.
        """
        try:
            return SliceMetrics(metric)._raw
        except ValueError:
            return metric == BrainData.RAW_TYPE

    @staticmethod
    def from_pandas(df: pd.DataFrame, animal_name: str) -> Self:
        """
        Creates an instance of [`AnimalBrain`][braian.AnimalBrain] from a `DataFrame`.

        Parameters
        ----------
        df
            A [`to_pandas`][braian.AnimalBrain.to_pandas]-compatible `DataFrame`.
        animal_name
            The name of the animal associated to the data in `df`.

        Returns
        -------
        :
            An instance of `AnimalBrain` that corresponds to the data in `df`.

        See also
        --------
        [`to_pandas`][braian.AnimalBrain.to_pandas]
        """
        if type(metric:=df.columns.name) != str:
            metric = str(df.columns.name)
        raw = AnimalBrain.is_raw(metric)
        markers_data = dict()
        areas = None
        regex = r'(.+) \((.+)\)$'
        pattern = re.compile(regex)
        for column, data in df.items():
            # extracts name and units from the column's name. E.g. 'area (mm²)' -> ('area', 'mm²')
            matches = re.findall(pattern, column)
            name, units = matches[0] if len(matches) == 1 else (column, None)
            if name == "area":
                areas = BrainData(data, animal_name, metric, units)
            else: # it's a marker
                markers_data[name] = BrainData(data, animal_name, metric, units)
        return AnimalBrain(markers_data=markers_data, areas=areas, raw=raw)

    @staticmethod
    def from_csv(filepath: Path|str, name: str, sep: str=",") -> Self:
        """
        Reads a comma-separated values (CSV) file into `AnimalBrain`.

        Parameters
        ----------
        filepath
            Any valid string path is acceptable. It also accepts any [os.PathLike][].
        name
            Name of the animal associated to the data.
        sep
            Character or regex pattern to treat as the delimiter.

        Returns
        -------
        :
            An instance of `AnimalBrain` that corresponds to the data in the CSV file

        See also
        --------
        [`to_csv`][braian.AnimalBrain.to_csv]
        """
        # read CSV
        # filename = f"{name}.csv" if metric is None else f"{name}_{str(metric)}.csv"
        df = pd.read_csv(filepath, sep=sep, header=0, index_col=0)
        if df.index.name == "Class":
            # is old csv
            raise ValueError("Trying to read an AnimalBrain from an outdated formatted .csv. Please re-run the analysis from the SlicedBrain!")
        df.columns.name = df.index.name
        df.index.name = None
        return AnimalBrain.from_pandas(df, name)

    @staticmethod
    def merge_hemispheres(animal_brain: Self) -> Self:
        """
        Creates a new `AnimalBrain` from `animal_brain` with no hemisphere distinction.

        Parameters
        ----------
        animal_brain
            A brain to merge.

        Returns
        -------
        :
            A new [`AnimalBrain`][braian.AnimalBrain] with no hemisphere distinction.
            If `animal_brain` is already merged, it return the same instance with no changes.

        See also
        --------
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        """
        brain: AnimalBrain = copy.copy(animal_brain)
        brain.markers_data = {m: m_data.merge_hemispheres() for m, m_data in brain.markers_data.items()}
        brain.areas = brain.areas.merge_hemispheres()
        return brain

def _extract_name_and_units(ls) -> Generator[str, None, None]:
    regex = r'(.+) \((.+)\)$'
    pattern = re.compile(regex)
    for s in ls:
        matches = re.findall(pattern, s)
        assert len(matches) == 1, f"Cannot find units in column '{s}'"
        yield matches[0]