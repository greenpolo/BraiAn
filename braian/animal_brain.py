import copy
import functools
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
from braian.brain_data import BrainData, BrainHemisphere, sort_by_ontology
from braian.utils import save_csv, deprecated

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

def _combined_regions(*bd: BrainData) -> list[str]:
    all_regions = [set(bd_.regions) for bd_ in bd]
    return list(functools.reduce(set.__or__, all_regions))

def _to_legacy_series(bd: BrainData) -> pd.Series:
    if bd.hemisphere is not BrainHemisphere.BOTH:
        return bd.data.add_prefix(bd.hemisphere.name.capitalize()+": ")
    return bd.data

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
        all_slices = all_slices.groupby(by=["acronym", "hemisphere"]).filter(lambda g: len(g) >= min_slices)
        raw = not densities and self._raw
        return self.apply(all_slices.groupby(by=["acronym", "hemisphere"])), raw

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
            sliced_brain = sliced_brain.merge_hemispheres()

        name = sliced_brain.name
        markers = copy.copy(sliced_brain.markers)
        metric = SliceMetrics(metric)
        if len(sliced_brain.slices) < min_slices:
            raise EmptyBrainError(sliced_brain.name)
        redux, raw = metric(sliced_brain, min_slices, densities=densities)
        if redux.shape[0] == 0:
            raise EmptyBrainError(sliced_brain.name)
        metric = f"{str(metric)}_densities" if densities else str(metric)
        if sliced_brain.is_split:
            hemispheres = (BrainHemisphere.LEFT, BrainHemisphere.RIGHT)
        else:
            hemispheres = (BrainHemisphere.BOTH,)
        areas = tuple(BrainData(redux["area"].xs(hem.value, level=1),
                                name=name, metric=metric,
                                units=sliced_brain.units["area"],
                                hemisphere=hem)
                for hem in hemispheres)
        # areas = BrainData(redux["area"], name=name, metric=metric, units=sliced_brain.units["area"])
        markers_data = {
            m: tuple(
                BrainData(redux[m].xs(hem.value, level=1),
                                      name=name, metric=metric,
                                      units=sliced_brain.units[m],
                                      hemisphere=hem)
                for hem in hemispheres)
            for m in markers
        }
        return AnimalBrain(markers_data=markers_data, sizes=areas, raw=raw)

    def __init__(self,
                 markers_data: dict[str,BrainData|tuple[BrainData,BrainData]],
                 sizes: BrainData|tuple[BrainData,BrainData],
                 raw: bool=False) -> None:
        """
        Associates [`BrainData`][braian.BrainData] coming from a single subject,
        for each marker and for each brain region.

        Parameters
        ----------
        markers_data
            A dictionary that associates the name of a marker to a `BrainData`
        sizes
            A `BrainData` with the size of the subject's brain regions.\
            A tuple of two `BrainData` is required if the associated data is split between right and left [hemispheres][braian.BrainHemisphere].
        raw
            Whether the data can be considered _raw_ (e.g., contains simple cell positive counts) or not.
        """
        assert len(markers_data) > 0 and sizes is not None, "You must provide both a dictionary of BrainData (markers) and an additional BrainData for the size of each region"
        markers_data = {m: (ds,) if isinstance(ds, BrainData) else tuple(sorted(ds, key=lambda d: d.hemisphere.value))
                        for m,ds in markers_data.items()}
        are_merged = [m.hemisphere is BrainHemisphere.BOTH for ms in markers_data.values() for m in ms]
        assert all(are_merged) or not any(are_merged), "You must provide BrainData that has merged hemispheres for all markers or for none."
        if all(are_merged):
            if isinstance(sizes, BrainData):
                sizes = (sizes,)
            assert (isinstance(sizes, tuple) and len(sizes) == 1 and\
                all(map(lambda x: type(x) is BrainData, sizes))), "'sizes' shoud be a BrainData object."
        else:
            assert isinstance(sizes, tuple) and len(sizes) == 2 and\
                 all(map(lambda x: type(x) is BrainData, sizes)), "'sizes' shoud be a tuple of one or two BrainData objects."
        self._markers: tuple[str] = tuple(markers_data.keys())
        self._markers_data: dict[str,tuple[BrainData]|tuple[BrainData,BrainData]] = markers_data
        self._sizes: tuple[BrainData]|tuple[BrainData,BrainData] = sizes
        self.raw: bool = raw
        """Whether the data can be considered _raw_ (e.g., contains simple cell positive counts) or not."""
        assert all([m.data_name == self.name for ms in markers_data.values() for m in ms]), "All markers' BrainData must be from the same animal!"
        assert all([m.metric == self.metric for ms in markers_data.values() for m in ms]), "All markers' BrainData must have the same metric!"
        return

    @property
    def hemispheres(self) -> tuple[BrainData]|tuple[BrainData,BrainData]:
        return tuple(s.hemisphere for s in self._sizes)

    @property
    def sizes(self) -> BrainData:
        """
        The data corresponding to the size of each brain region of the current AnimalBrain.
        Only available if the brain is not split between left and right hemispheres.
        """
        if self.is_split:
            raise ValueError("Cannot get a single size for all brain regions because the brain is split between left and right hemispheres."+\
                             "Use AnimalBrain.hemisizes.")
        return self._sizes[0]

    @property
    def hemisizes(self) -> tuple[BrainData,BrainData]|tuple[BrainData]:
        """
        The data corresponding to the size of each brain region of the current AnimalBrain.
        If the AnimalBrain has data for two hemispheres, it is a tuple of size two.
        Else, it is a tuple of size one.
        """
        return self._sizes

    @property
    @deprecated("Use sizes instead.")
    def areas(self) -> BrainData:
        return self.sizes

    @property
    def markers(self) -> tuple[str]:
        """
        The name of the markers for which the current `AnimalBrain` has data.
        """
        return tuple(self._markers)

    @property
    def metric(self) -> str:
        """
        The name of the metric used to compute current data.
        Equals to [`RAW_TYPE`][braian.BrainData.RAW_TYPE] if no previous normalization was preformed.
        """
        return self._markers_data[self._markers[0]][0].metric

    @property
    def is_split(self) -> bool:
        """Whether the data of the current `AnimalBrain` makes a distinction between right and left hemisphere."""
        return len(self._sizes) == 2

    @property
    def name(self) -> str:
        """The name of the animal."""
        return self._markers_data[self._markers[0]][0].data_name

    @property
    def regions(self) -> list[str]:
        """
        The list of region acronyms for which the current `AnimalBrain` has data.
        Only available if the brain is not split between left and right hemispheres.
        """
        # assumes sizes' and all markers' BrainData are synchronized
        if self.is_split:
            raise ValueError("Cannot get a single list for all brain regions because the brain is split between left and right hemispheres."+\
                             "Use AnimalBrain.hemiregions.")
        return self._sizes[0].regions

    @property
    def hemiregions(self) -> dict[BrainHemisphere,list[str]]:
        """
        The brain regions corresponding to each possible [`BrainHemisphere`][braian.BrainHemisphere].
        """
        return {s.hemisphere: s.regions for s in self._sizes}

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AnimalBrain(name='{self.name}', metric={self.metric}, markers={list(self._markers)})"

    def __getitem__(self, key) -> BrainData:
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
        if isinstance(key,str):
            if self.is_split:
                raise ValueError("Cannot get a single marker data for all brain regions because the brain is split between left and right hemispheres."+\
                                 "You should also specify a braian.BrainHemisphere.")
            return self._markers_data[key][0] # there is no hemisphere distinction
        elif isinstance(key,tuple) and len(key) == 2 and isinstance(key[0],str):
            marker,hemi = key
            hemi = BrainHemisphere(hemi)
            if not self.is_split and hemi is not BrainHemisphere.BOTH:
                ValueError(f"The brain has no hemisphere distinction. You cannot select data for '{hemi.name}' hemisphere.")
            try:
                return next(d for d in self._markers_data[marker] if d.hemisphere is hemi)
            except StopIteration | KeyError:
                pass
            msg = f"Cannot find '{marker}' data"
            if self.is_split:
                msg += f" for {hemi.name} hemisphere"
            raise KeyError(msg+".")
        raise TypeError(f"Unknown marker data selection: '{key}'")
        

    def remove_region(self, region: str, *regions, fill_nan: bool=True,
                      hemisphere: BrainHemisphere=BrainHemisphere.BOTH) -> None:
        """
        Removes the data from all the given regions in the current `AnimalBrain`

        Parameters
        ----------
        region
            The acronyms of the regions to exclude from the data.
        fill_nan
            If True, instead of removing the regions completely, it fills their value to [`NA`][pandas.NA].
        hemisphere
            If `BOTH`, it completely removes all `regions` from the brain.\
            Else, it remove only the `regions` from the specified hemisphere.
        """
        regions = (region, *regions)
        hemisphere = BrainHemisphere(hemisphere)
        for hemidata in self._markers_data.values():
            for data in hemidata:
                if data.hemisphere is not hemisphere:
                    continue
                data.remove_region(*regions, inplace=True, fill_nan=fill_nan)
        for sizes in self._sizes:
            if sizes.hemisphere is not hemisphere:
                continue
            sizes.remove_region(*regions, inplace=True, fill_nan=fill_nan)

    def remove_missing(self, kind:str="any") -> None:
        """
        Removes the regions for which there is no data about the size.

        Parameters
        ----------
        kind
            Choice to select which missing brain regions to remove:
                * `'both'`: completely removes the brain regions missing from both the hemispheres,
                * `'any'`: completely removes the brain regions missing from any of the two hemispheres,
                * `'same'`: removes just the brain regions missing within the same hemisphere.

            If the brain has no hemisphere distinction, the result will be the same.
        """
        match kind:
            case "any":
                missing_regions = {}
                for sizes in self._sizes:
                    missing_regions |= set(sizes.missing_regions())
                self.remove_region(*missing_regions, fill_nan=False, hemisphere=BrainHemisphere.BOTH)
            case "both":
                missing_regions = {}
                for sizes in self._sizes:
                    missing_regions &= set(sizes.missing_regions())
                self.remove_region(*missing_regions, fill_nan=False, hemisphere=BrainHemisphere.BOTH)
            case "same":
                for sizes in self._sizes:
                    missing_regions = sizes.missing_regions()
                    self.remove_region(*missing_regions, fill_nan=False, hemisphere=sizes.hemisphere)
            case _:
                raise ValueError(f"Unknown kind '{kind}'.")

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
        # TODO: add option to sync markers and hemispheres
        
        # NOTE: this adds the regions present in only one of the two hemispheres
        # if self.is_split:
        #     combined_regions = _combined_regions(self[self._markers[0], 1], self[self._markers[0], 2])
        #     combined_regions = sort_by_ontology(combined_regions, atlas_ontology, fill=False)
        #     combined_hemidata = {m: tuple(hemidata.select_from_list(combined_regions, fill_nan=True, inplace=False) for hemidata in data)
        #                         for m,data in self._markers_data.items()}
        #     combined_hemisizes = tuple(hemisize.select_from_list(combined_regions, fill_nan=True, inplace=False) for hemisize in self.hemisizes)
        markers_data = {marker: tuple(m_data.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=inplace) for m_data in hemidata)
                        for marker, hemidata in self._markers_data.items()}
        sizes = tuple(s.sort_by_ontology(brain_ontology, fill_nan=fill_nan, inplace=inplace) for s in self._sizes)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, sizes=sizes, raw=self.raw)
        else:
            self._markers_data = markers_data
            self._sizes = sizes
            return self

    def select_from_list(self, regions: Sequence[str],
                         fill_nan: bool=False, inplace: bool=False,
                         hemisphere: BrainHemisphere=BrainHemisphere.BOTH) -> Self:
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
        hemisphere
            If not [`BOTH`][braian.BrainHemisphere] and the brain [is split][braian.AnimalBrain.is_split],
            it only selects the brain regions from the given hemisphere.

        Returns
        -------
        :
            A brain with data filtered accordingly to the given `regions`.
            If `inplace=True` it returns the same instance.

        See also
        --------
        [`AnimalBrain.select_from_ontology`][braian.AnimalBrain.select_from_ontology]
        """
        hemisphere = BrainHemisphere(hemisphere)
        markers_data = {marker: tuple(
                                m_data if m_data.hemisphere not in (BrainHemisphere.BOTH, hemisphere)\
                                else m_data.select_from_list(regions, fill_nan=fill_nan, inplace=inplace)
                            for m_data in hemidata)
                        for marker, hemidata in self._markers_data.items()}
        sizes = tuple(
                s if s.hemisphere not in (BrainHemisphere.BOTH, hemisphere)\
                else s.select_from_list(regions, fill_nan=fill_nan, inplace=inplace)
            for s in self._sizes)
        if not inplace:
            return AnimalBrain(markers_data=markers_data, sizes=sizes, raw=self.raw)
        else:
            self._markers_data = markers_data
            self._sizes = sizes
            return self

    def select_from_ontology(self, brain_ontology: AllenBrainOntology,
                             fill_nan: bool=False, inplace: bool=False,
                             hemisphere: BrainHemisphere=BrainHemisphere.BOTH) -> Self:
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
        hemisphere
            If not [`BOTH`][braian.BrainHemisphere] and the brain [is split][braian.AnimalBrain.is_split],
            it only selects the brain regions from the given hemisphere.

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
        return self.select_from_list(list(selectable_regions), fill_nan=fill_nan, inplace=inplace, hemisphere=hemisphere)

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
        if len(self._markers) == 1:
            marker = self._markers[0]
        assert marker is not None, "Missing marker to get units of."
        assert marker in self._markers, f"Could not get units of marker '{marker}'!"
        return self._markers_data[marker][0].units


    def merge_hemispheres(self) -> Self:
        """
        Creates a new `AnimalBrain` from the current instance with no hemisphere distinction.

        Returns
        -------
        :
            A new [`AnimalBrain`][braian.AnimalBrain] with no hemisphere distinction.
            If it is already merged, it return the same instance with no changes.

        See also
        --------
        [`BrainData.merge_hemispheres`][braian.BrainData.merge_hemispheres]
        """
        if not self.is_split:
            return self
        brain: AnimalBrain = copy.copy(self)
        brain._markers_data = {m: tuple(m_data.merge_hemispheres() for m_data in hemidata) for m, hemidata in brain._markers_data.items()}
        brain._sizes = tuple(s.merge_hemispheres() for s in self._sizes)
        return brain

    def to_pandas(self, units: bool=False, missing_as_nan: bool=False) -> pd.DataFrame:
        """
        Converts the current `AnimalBrain` to a DataFrame. T

        Parameters
        ----------
        units
            Whether the columns should include the units of measurement or not.
        missing_as_nan
            If True, it converts missing values [`NA`][pandas.NA] as [`NaN`][numpy.nan].
            Note that if the corresponding brain data is integer-based, it converts them to float.

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
        brain_dict = dict()
        hemisizes = sorted(self._sizes, key=lambda bd: bd.hemisphere.name) # first LEFT and then RIGHT hemispheres
        index_size = f"size ({self._sizes[0].units})" if units else "size"
        brain_dict[index_size] = pd.concat(map(_to_legacy_series, hemisizes))
        for marker, hemidata in self._markers_data.items():
            hemidata = sorted(hemidata, key=lambda bd: bd.hemisphere.name) # first LEFT and then RIGHT hemispheres
            index_marker = f"{marker} ({hemidata[0].units})" if units else marker
            brain_dict[index_marker] = pd.concat(map(_to_legacy_series, hemidata))
        data = pd.concat(brain_dict, axis=1)
        data.columns.name = str(self.metric)
        if missing_as_nan:
            data = data.astype(float)
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
        raise NotImplementedError()
        if isinstance(metric:=df.columns.name, str):
            metric = str(df.columns.name)
        raw = AnimalBrain.is_raw(metric)
        markers_data = dict()
        sizes = None
        regex = r'(.+) \((.+)\)$'
        pattern = re.compile(regex)
        for column, data in df.items():
            # extracts name and units from the column's name. E.g. 'size (mm²)' -> ('size', 'mm²')
            matches = re.findall(pattern, column)
            name, units = matches[0] if len(matches) == 1 else (column, None)
            if name == "area" or name == "size": # braian <= 1.0.3 called sizes "area"
                sizes = BrainData(data, animal_name, metric, units)
            else: # it's a marker
                markers_data[name] = BrainData(data, animal_name, metric, units)
        return AnimalBrain(markers_data=markers_data, sizes=sizes, raw=raw)

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

def _extract_name_and_units(ls) -> Generator[str, None, None]:
    regex = r'(.+) \((.+)\)$'
    pattern = re.compile(regex)
    for s in ls:
        matches = re.findall(pattern, s)
        assert len(matches) == 1, f"Cannot find units in column '{s}'"
        yield matches[0]